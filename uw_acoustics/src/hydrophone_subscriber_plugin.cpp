#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Events.hh>
#include <ignition/math/Vector3.hh>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>

namespace gazebo {

class HydrophoneSubscriberPlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model_->GetWorld();

    // Topics and model/link naming
    source_audio_topic_ = sdf->Get<std::string>("source_audio_topic", "/acoustics/source_audio").first;
    out_topic_          = sdf->Get<std::string>("out_topic", "/hydrophones/left").first;

    source_model_name_  = sdf->Get<std::string>("source_model", "acoustic_source").first;
    source_link_name_   = sdf->Get<std::string>("source_link", "link").first;
    hydro_link_name_    = sdf->Get<std::string>("hydro_link", "link").first;

    // Signal params
    fs_             = sdf->Get<double>("fs", 48000.0).first;
    block_size_     = sdf->Get<int>("block_size", 256).first;
    speed_of_sound_ = sdf->Get<double>("speed_of_sound", 1500.0).first;
    loss_model_     = sdf->Get<std::string>("loss_model", "none").first;
    amplitude_      = sdf->Get<double>("amplitude", 1.0).first;
    noise_std_      = sdf->Get<double>("noise_std", 0.0).first;
    buffer_seconds_ = sdf->Get<double>("buffer_seconds", 3.0).first;
    start_time_     = sdf->Get<double>("start_time", 0.0).first;

    // ROS init
    if (!ros::isInitialized()) {
      int argc = 0; char** argv = nullptr;
      ros::init(argc, argv, "gazebo_hydrophone", ros::init_options::NoSigintHandler);
    }
    nh_.reset(new ros::NodeHandle(""));

    pub_ = nh_->advertise<std_msgs::Float32MultiArray>(out_topic_, 50);
    pub_r_   = nh_->advertise<std_msgs::Float32>(out_topic_ + "/range_m", 50);
    pub_D_   = nh_->advertise<std_msgs::Float32>(out_topic_ + "/delay_samples", 50);
    pub_miss_= nh_->advertise<std_msgs::Float32>(out_topic_ + "/miss_frac", 20);

    // Subscribe to source audio
    sub_ = nh_->subscribe(source_audio_topic_, 200, &HydrophoneSubscriberPlugin::OnSourceBlock, this);

    // Cache hydro link
    hydro_link_ = model_->GetLink(hydro_link_name_);
    if (!hydro_link_) {
      ROS_WARN("[Hydrophone] model='%s' link '%s' not found; using model pose fallback",
               model_->GetName().c_str(), hydro_link_name_.c_str());
    }

    // Ring buffer alloc
    ring_size_ = std::max<long long>( (long long)std::llround(buffer_seconds_ * fs_), (long long)block_size_ * 4 );
    ring_.assign((size_t)ring_size_, 0.0f);
    stamp_.assign((size_t)ring_size_, (long long)-9e18);

    // noise
    rng_.seed(std::random_device{}());
    dist_norm_ = std::normal_distribution<float>(0.0f, 1.0f);

    last_out_block_pub_ = -1;

    update_conn_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&HydrophoneSubscriberPlugin::OnUpdate, this));

    ROS_INFO("[Hydrophone] out=%s  src_topic=%s  fs=%.1f N=%d buffer=%.2fs ring=%lld",
             out_topic_.c_str(), source_audio_topic_.c_str(), fs_, block_size_,
             buffer_seconds_, (long long)ring_size_);
  }

private:
  // --- Source buffering ---
  inline size_t slotIndex(long long k) const {
    long long m = k % ring_size_;
    if (m < 0) m += ring_size_;
    return (size_t)m;
  }

  inline bool hasSample(long long k) const {
    size_t idx = slotIndex(k);
    return stamp_[idx] == k;
  }

  inline float getSample(long long k) const {
    size_t idx = slotIndex(k);
    return (stamp_[idx] == k) ? ring_[idx] : 0.0f;
  }

  // linear interpolation fetch s[kd]
  float getSampleFrac(double kd, int &miss_count) const {
    long long k0 = (long long)std::floor(kd);
    double a = kd - (double)k0;

    bool ok0 = hasSample(k0);
    bool ok1 = hasSample(k0 + 1);
    if (!ok0 || !ok1) {
      miss_count++;
      // fallback to whichever exists
      if (ok0) return getSample(k0);
      if (ok1) return getSample(k0 + 1);
      return 0.0f;
    }
    float s0 = getSample(k0);
    float s1 = getSample(k0 + 1);
    return (float)((1.0 - a) * (double)s0 + a * (double)s1);
  }

  void OnSourceBlock(const std_msgs::Float32MultiArray::ConstPtr &msg) {
    // Absolute start sample index
    const long long k_start = (long long)msg->layout.data_offset;
    const size_t N = msg->data.size();

    // Store into ring
    for (size_t n = 0; n < N; ++n) {
      long long k = k_start + (long long)n;
      size_t idx = slotIndex(k);
      ring_[idx]  = (float)msg->data[n];
      stamp_[idx] = k;
    }

    // Track buffered range (best effort; not strictly required for correctness)
    if (N > 0) {
      const long long k_end = k_start + (long long)N - 1;
      if (!have_range_) {
        k_min_ = k_start; k_max_ = k_end; have_range_ = true;
      } else {
        k_min_ = std::min(k_min_, k_start);
        k_max_ = std::max(k_max_, k_end);
      }
    }
  }

  // --- Geometry helpers ---
  double gainFromRange(double r) const {
    if (loss_model_ == "inverse") {
      return (r > 1e-6) ? (amplitude_ / r) : amplitude_;
    }
    if (loss_model_ == "inverse_square") {
      return (r > 1e-6) ? (amplitude_ / (r * r)) : amplitude_;
    }
    return amplitude_; // "none"
  }

  void resolveSourceLink() {
    if (source_link_) return;
    auto src_model = world_->ModelByName(source_model_name_);
    if (!src_model) return;
    source_link_ = src_model->GetLink(source_link_name_);
    // If missing, we will fall back to src_model pose in OnUpdate; but warn there
  }

  ignition::math::Vector3d getHydPos() const {
    return hydro_link_ ? hydro_link_->WorldPose().Pos()
                       : model_->WorldPose().Pos();
  }

  ignition::math::Vector3d getSrcPos() const {
    auto src_model = world_->ModelByName(source_model_name_);
    if (!src_model) return ignition::math::Vector3d(0,0,0);
    if (source_link_) return source_link_->WorldPose().Pos();
    return src_model->WorldPose().Pos();
  }

  void OnUpdate() {
    if (!pub_) return;

    const double t = world_->SimTime().Double() - start_time_;
    if (t < 0.0) return;

    resolveSourceLink();

    // Output block scheduler (sim-time based)
    const long long k_now = (long long)std::floor(t * fs_);
    const long long b_now = k_now / block_size_;
    if (last_out_block_pub_ < 0) last_out_block_pub_ = b_now - 1;

    if (b_now <= last_out_block_pub_) return;

    const long long b_pub   = b_now - 1;
    const long long k_start = b_pub * (long long)block_size_;
    last_out_block_pub_ = b_now;

    // Geometry (sample-hold per block; fine for now)
    auto Psrc = getSrcPos();
    auto Phyd = getHydPos();

    const double dx = Psrc.X() - Phyd.X();
    const double dy = Psrc.Y() - Phyd.Y();
    const double dz = Psrc.Z() - Phyd.Z();
    const double r  = std::sqrt(dx*dx + dy*dy + dz*dz);

    const double tau = (r > 0.0) ? (r / speed_of_sound_) : 0.0;
    const double D   = tau * fs_;               // fractional samples
    const double g   = gainFromRange(r);

    // Output message
    std_msgs::Float32MultiArray out;
    out.layout.dim.resize(1);
    out.layout.dim[0].label  = "samples";
    out.layout.dim[0].size   = (uint32_t)block_size_;
    out.layout.dim[0].stride = (uint32_t)block_size_;
    out.layout.data_offset   = (uint32_t)k_start;   // absolute sample index for this hydro block
    out.data.resize((size_t)block_size_);

    int miss = 0;

    for (int n = 0; n < block_size_; ++n) {
      const long long k = k_start + (long long)n;
      const double kd = (double)k - D;  // source index to sample
      float s = getSampleFrac(kd, miss);
      float y = (float)(g) * s;
      if (noise_std_ > 0.0) y += (float)noise_std_ * dist_norm_(rng_);
      out.data[(size_t)n] = y;
    }

    // Debug pubs
    std_msgs::Float32 r_msg; r_msg.data = (float)r;
    std_msgs::Float32 D_msg; D_msg.data = (float)D;
    std_msgs::Float32 m_msg; m_msg.data = (float)miss / (float)block_size_;
    pub_r_.publish(r_msg);
    pub_D_.publish(D_msg);
    pub_miss_.publish(m_msg);

    // If the source link wasn't found, warn (throttled)
    if (!source_link_) {
      ROS_WARN_THROTTLE(2.0, "[Hydrophone %s] source_link '%s' not found on model '%s' (using model pose)",
                        model_->GetName().c_str(), source_link_name_.c_str(), source_model_name_.c_str());
    }

    pub_.publish(out);
  }

  // Members
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  event::ConnectionPtr update_conn_;

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher pub_;
  ros::Publisher pub_r_, pub_D_, pub_miss_;
  ros::Subscriber sub_;

  // naming / topics
  std::string source_audio_topic_;
  std::string out_topic_;
  std::string source_model_name_;
  std::string source_link_name_;
  std::string hydro_link_name_;

  // links
  physics::LinkPtr hydro_link_;
  physics::LinkPtr source_link_;

  // params
  double fs_{48000.0};
  int block_size_{256};
  double speed_of_sound_{1500.0};
  std::string loss_model_{"none"};
  double amplitude_{1.0};
  double noise_std_{0.0};
  double buffer_seconds_{3.0};
  double start_time_{0.0};

  // ring buffer
  long long ring_size_{0};
  std::vector<float> ring_;
  std::vector<long long> stamp_;
  bool have_range_{false};
  long long k_min_{0}, k_max_{0};

  // scheduling
  long long last_out_block_pub_{-1};

  // noise
  std::mt19937 rng_;
  std::normal_distribution<float> dist_norm_;
};

GZ_REGISTER_MODEL_PLUGIN(HydrophoneSubscriberPlugin)

} // namespace gazebo

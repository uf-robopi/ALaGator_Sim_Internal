#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Events.hh>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

#include <sndfile.h>

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

namespace gazebo {

class AcousticSourcePlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {

    model_ = model;
    world_ = model_->GetWorld();

    out_topic_   = sdf->Get<std::string>("out_topic", "/acoustics/source_audio").first;
    audio_path_  = sdf->Get<std::string>("audio_path", "").first;
    loop_        = sdf->Get<bool>("loop", true).first;
    fs_          = sdf->Get<double>("fs", 48000.0).first;
    block_size_  = sdf->Get<int>("block_size", 256).first;
    start_time_  = sdf->Get<double>("start_time", 0.0).first;

    if (audio_path_.empty()) {
      gzerr << "[AcousticSource] audio_path is empty.\n";
      return;
    }

    // ROS init
    if (!ros::isInitialized()) {
      int argc = 0; char** argv = nullptr;
      ros::init(argc, argv, "gazebo_acoustic_source", ros::init_options::NoSigintHandler);
    }
    nh_.reset(new ros::NodeHandle(""));
    pub_ = nh_->advertise<std_msgs::Float32MultiArray>(out_topic_, 50);

    if (!loadAudio(audio_path_)) {
      gzerr << "[AcousticSource] Failed to load: " << audio_path_ << "\n";
      return;
    }

    last_block_pub_ = -1;
    update_conn_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&AcousticSourcePlugin::OnUpdate, this));

    ROS_INFO("[AcousticSource] topic=%s file=%s file_fs=%d file_ch=%d target_fs=%.1f N=%d loop=%d",
             out_topic_.c_str(), audio_path_.c_str(), file_fs_, file_ch_,
             fs_, block_size_, (int)loop_);
  }

private:
  bool loadAudio(const std::string& path) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    SNDFILE* sf = sf_open(path.c_str(), SFM_READ, &info);
    if (!sf) {
      ROS_ERROR("[AcousticSource] libsndfile: %s", sf_strerror(nullptr));
      return false;
    }
    file_fs_ = info.samplerate;
    file_ch_ = info.channels;
    if (info.frames <= 0) {
      sf_close(sf);
      ROS_ERROR("[AcousticSource] no frames in file.");
      return false;
    }

    std::vector<float> interleaved((size_t)info.frames * (size_t)info.channels);
    sf_count_t nread = sf_readf_float(sf, interleaved.data(), info.frames);
    sf_close(sf);
    if (nread != info.frames) {
      ROS_WARN("[AcousticSource] read %lld/%lld frames", (long long)nread, (long long)info.frames);
    }

    // Convert to mono float buffer
    wav_.resize((size_t)nread);
    if (file_ch_ == 1) {
      for (sf_count_t i = 0; i < nread; ++i) wav_[(size_t)i] = interleaved[(size_t)i];
    } else {
      for (sf_count_t i = 0; i < nread; ++i) {
        double s = 0.0;
        for (int c = 0; c < file_ch_; ++c) {
          s += interleaved[(size_t)i * (size_t)file_ch_ + (size_t)c];
        }
        wav_[(size_t)i] = (float)(s / (double)file_ch_);
      }
    }

    // Hard policy: require file samplerate == fs_ (keeps this simple and correct)
    if ((int)std::llround(fs_) != file_fs_) {
      ROS_ERROR("[AcousticSource] File fs=%d but plugin fs=%.1f. Please resample file to match.",
                file_fs_, fs_);
      return false;
    }

    // Normalize lightly if needed (optional)
    float mx = 0.0f;
    for (float v : wav_) mx = std::max(mx, std::abs(v));
    if (mx > 1.5f) {
      for (float& v : wav_) v /= mx;
      ROS_WARN("[AcousticSource] normalized audio (max=%.3f)", mx);
    }

    wav_len_ = (long long)wav_.size();
    return wav_len_ > 0;
  }

  inline float sampleAt(long long k) const {
    if (wav_len_ <= 0) return 0.0f;
    if (k < 0) return 0.0f;
    if (loop_) {
      long long kk = k % wav_len_;
      return wav_[(size_t)kk];
    } else {
      if (k >= wav_len_) return 0.0f;
      return wav_[(size_t)k];
    }
  }

  void OnUpdate() {
    if (!pub_) return;

    const double t = world_->SimTime().Double() - start_time_;
    if (t < 0.0) return;

    const long long k_now = (long long)std::floor(t * fs_);
    const long long b_now = k_now / block_size_;
    if (last_block_pub_ < 0) last_block_pub_ = b_now - 1;

    if (b_now <= last_block_pub_) return;

    const long long b_pub   = b_now - 1;
    const long long k_start = b_pub * (long long)block_size_;
    last_block_pub_ = b_now;

    std_msgs::Float32MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label  = "samples";
    msg.layout.dim[0].size   = (uint32_t)block_size_;
    msg.layout.dim[0].stride = (uint32_t)block_size_;
    msg.layout.data_offset   = (uint32_t)k_start;  // absolute sample index
    msg.data.resize((size_t)block_size_);

    for (int n = 0; n < block_size_; ++n) {
      msg.data[(size_t)n] = sampleAt(k_start + n);
    }

    pub_.publish(msg);
  }

  // Members
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  event::ConnectionPtr update_conn_;

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher pub_;

  std::string out_topic_;
  std::string audio_path_;
  bool loop_{true};
  double fs_{48000.0};
  int block_size_{256};
  double start_time_{0.0};

  int file_fs_{0};
  int file_ch_{0};

  std::vector<float> wav_;
  long long wav_len_{0};
  long long last_block_pub_{-1};
};

GZ_REGISTER_MODEL_PLUGIN(AcousticSourcePlugin)

} // namespace gazebo

// =====================================
// Hydrophone Gazebo plugin with Doppler
// Author : Adnan Abdullah
// =====================================

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Events.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float32.h>

#include <random>
#include <cmath>
#include <memory>
#include <string>
#include <algorithm>

#include "uw_acoustics/FractionalDelay.hpp"

namespace gazebo {

class HydrophonePlugin : public ModelPlugin {
public:
  // Deterministic white-ish noise [-1,1]
  static inline float prn_from_k(long long k) {
    uint64_t x = static_cast<uint64_t>(k);
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    const float scale = 1.0f / 2147483648.0f; // 2^31
    int32_t hi = static_cast<int32_t>(x >> 33);
    return static_cast<float>(hi) * scale;
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;
    world_ = model_->GetWorld();

    // --- Parameters ---
    source_model_name_ = sdf->Get<std::string>("source_model", "acoustic_source").first;
    out_topic_         = sdf->Get<std::string>("out_topic", "/hydrophones/left").first;

    // Names of links (optional; we fall back to model pose/vel if missing)
    source_link_name_  = sdf->Get<std::string>("source_link", "link").first;
    hydro_link_name_   = sdf->Get<std::string>("hydro_link",  "link").first;

    signal_type_ = sdf->Get<std::string>("signal_type", "tone").first; // "tone" | "noise"
    freq_        = sdf->Get<double>("freq", 150.0).first;
    fs_          = sdf->Get<double>("fs", 100000.0).first;
    amplitude_   = sdf->Get<double>("amplitude", 1.0).first;
    start_time_  = sdf->Get<double>("start_time", 0.0).first;

    speed_of_sound_ = sdf->Get<double>("speed_of_sound", 1482.0).first;
    block_size_     = sdf->Get<int>("block_size", 2048).first;
    loss_model_     = sdf->Get<std::string>("loss_model", "none").first; // "none"|"inverse"|"inverse_square"
    noise_std_      = sdf->Get<double>("noise_std", 0.0).first;

    if (!ros::isInitialized()) {
      int argc = 0; char** argv = nullptr;
      ros::init(argc, argv, "gazebo_hydrophone", ros::init_options::NoSigintHandler);
    }
    nh_.reset(new ros::NodeHandle(""));   // global ns

    pub_           = nh_->advertise<std_msgs::Float32MultiArray>(out_topic_, 50);
    pub_block_     = nh_->advertise<std_msgs::Int64>(out_topic_ + "/block_idx", 10);
    pub_debug_tau_ = nh_->advertise<std_msgs::Float32>(out_topic_ + "/delay_samples", 50);
    pub_debug_r_   = nh_->advertise<std_msgs::Float32>(out_topic_ + "/range_m", 50);
    pub_vr_        = nh_->advertise<std_msgs::Float32>(out_topic_ + "/vr_mps", 50); // radial velocity (debug)

    rng_.seed(std::random_device{}());
    dist_norm_ = std::normal_distribution<float>(0.0f, 1.0f);

    // Fractional delay line capacity: allow several seconds of delay (generous).
    // Capacity in samples = 2 * fs 
    fdline_.reset(new FractionalDelay(static_cast<size_t>(2 * fs_)));

    // Cache hydro link (may be null -> we’ll fall back to model pose & vel)
    hydro_link_ = model_->GetLink(hydro_link_name_);
    if (!hydro_link_) {
      ROS_WARN("[Hydrophone] model='%s' link '%s' not found; using model pose/vel",
               model_->GetName().c_str(), hydro_link_name_.c_str());
    }

    last_block_idx_ = -1;  // ensures first publish uses b_now-1
    update_conn_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&HydrophonePlugin::OnUpdate, this));

    ROS_INFO_STREAM("[Hydrophone] '" << model_->GetName()
                    << "' source='" << source_model_name_
                    << "' source_link='" << source_link_name_
                    << "' hydro_link='" << hydro_link_name_
                    << "' -> " << out_topic_
                    << " fs=" << fs_ << " N=" << block_size_
                    << " type=" << signal_type_
                    << " c=" << speed_of_sound_ << " m/s");
  }

private:
  void OnUpdate() {
    if (!pub_) return;

    const double sim_t = world_->SimTime().Double() - start_time_;
    if (sim_t < 0.0) return;

    // Global sample + block indices
    const long long k_now  = static_cast<long long>(std::floor(sim_t * fs_));
    const long long b_now  = k_now / block_size_;
    if (last_block_idx_ < 0) last_block_idx_ = b_now - 1;

    if (b_now <= last_block_idx_) return;
    const long long b_pub   = b_now - 1;
    const long long k_start = b_pub * block_size_;
    last_block_idx_ = b_now;

    // Find source model
    auto src_model = world_->ModelByName(source_model_name_);
    if (!src_model) {
      ROS_WARN_THROTTLE(2.0, "[Hydrophone] source model '%s' not found",
                        source_model_name_.c_str());
      return;
    }

    // Resolve source link once (lazy)
    if (!source_link_) {
      source_link_ = src_model->GetLink(source_link_name_);
      if (!source_link_) {
        ROS_WARN_THROTTLE(2.0, "[Hydrophone] source link '%s' not found on model '%s'; using model pose/vel",
                          source_link_name_.c_str(), source_model_name_.c_str());
      }
    }

    // World positions
    const ignition::math::Vector3d Psrc = source_link_ ? source_link_->WorldPose().Pos()
                                                       : src_model->WorldPose().Pos();
    const ignition::math::Vector3d Phyd = hydro_link_  ? hydro_link_->WorldPose().Pos()
                                                       : model_->WorldPose().Pos();

    // World angle
    const ignition::math::Quaterniond Qsrc = source_link_ ? source_link_->WorldPose().Rot()
                                                       : src_model->WorldPose().Rot();
    const ignition::math::Quaterniond Qhyd = source_link_ ? hydro_link_->WorldPose().Rot()
                                                       : model_->WorldPose().Rot();

    // World linear velocities
    const ignition::math::Vector3d Vsrc = source_link_ ? source_link_->WorldLinearVel()
                                                       : src_model->WorldLinearVel();
    const ignition::math::Vector3d Vhyd = hydro_link_  ? hydro_link_->WorldLinearVel()
                                                       : model_->WorldLinearVel();

    // Get World frame x vector direction
    const ignition::math::Vector3d Xsrc = Qsrc.RotateVector(ignition::math::Vector3d::UnitX);
    const ignition::math::Vector3d Xhyd = Qhyd.RotateVector(ignition::math::Vector3d::UnitX);

    // Range and LOS unit vector (hydro -> source)
    const double dx = Psrc.X() - Phyd.X();
    const double dy = Psrc.Y() - Phyd.Y();
    const double dz = Psrc.Z() - Phyd.Z();
    const double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
    const double tau = (r > 0.0 ? r / speed_of_sound_ : 0.0);
    const double D0  = tau * fs_; // initial delay (samples)

    double ux = 0.0, uy = 0.0, uz = 0.0;
    if (r > 1e-12) {
      ux = dx / r; uy = dy / r; uz = dz / r;
    }

    // Spherical gradient scaler
    double rate = 1;                        // Arbitrary (1 is sphere)
    double alpha = acos(Xsrc.Dot(Xhyd));
    double gradient_scaler = (cos(rate*alpha)+1)/2;

    // Radial relative velocity (positive if separating)
    const double vr = (Vsrc.X() - Vhyd.X())*ux +
                      (Vsrc.Y() - Vhyd.Y())*uy +
                      (Vsrc.Z() - Vhyd.Z())*uz;

    // Per-sample delay change (samples/sample)
    const double dD = - vr / speed_of_sound_;

    // Amplitude / spreading loss
    const double gain = (loss_model_=="inverse"        ? (r>1e-6 ? 1.0/r     : 1.0) :
                         loss_model_=="inverse_square" ? (r>1e-6 ? 1.0/(r*r) : 1.0) :
                                                         1.0) * gradient_scaler * amplitude_;

    // Debug pubs
    {
      std_msgs::Int64 bix; bix.data = b_pub; pub_block_.publish(bix);
      std_msgs::Float32 tau_samp_msg, r_msg, vr_msg;
      tau_samp_msg.data = static_cast<float>(D0);
      r_msg.data        = static_cast<float>(r);
      vr_msg.data       = static_cast<float>(vr);
      pub_debug_tau_.publish(tau_samp_msg);
      pub_debug_r_.publish(r_msg);
      pub_vr_.publish(vr_msg);
    }

    // Output block
    const int N = block_size_;
    std_msgs::Float32MultiArray msg;
    msg.layout.dim.resize(1);
    msg.layout.dim[0].label  = "samples";
    msg.layout.dim[0].size   = N;
    msg.layout.dim[0].stride = N;
    msg.layout.data_offset   = static_cast<int32_t>(b_pub);
    msg.data.resize(N);

    if (signal_type_ == "tone") {
      // === Tone with Doppler via phase increment ===
      // phase_inc = 2π f0/fs * (1 - vr/c)
      const double two_pi_f0_over_fs = 2.0 * M_PI * freq_ / fs_;
      const double phase_inc = two_pi_f0_over_fs * (1.0 - vr / speed_of_sound_);

      // phase_ -= two_pi_f0_over_fs * D0; // uncomment to add absolute phase shift

      double phase = phase_;
      for (int n = 0; n < N; ++n) {
        double y = std::sin(phase);
        if (noise_std_ > 0.0) y += noise_std_ * dist_norm_(rng_);
        msg.data[n] = static_cast<float>(gain * y);
        phase += phase_inc;
        // keep phase bounded (optional)
        if (phase > 1e6) phase = std::fmod(phase, 2.0 * M_PI);
      }
      phase_ = std::fmod(phase, 2.0 * M_PI);

    } else {
      // === Wideband/noise with time-varying fractional delay ===
      double Dn = D0;                           // start delay (samples)
      const double Dmin = 0.0;
      const double Dmax = std::max(0.0, 2.0*fs_ - 8.0);

      for (int n = 0; n < N; ++n) {
        const long long k_abs = k_start + n;

        // drive the delay line with a coherent source sequence
        const float src_sample = prn_from_k(k_abs);
        fdline_->push(src_sample);

        // clamp and read delayed sample at current Dn
        if (Dn < Dmin) Dn = Dmin;
        if (Dn > Dmax) Dn = Dmax;

        float y = fdline_->readDelayed(Dn);
        if (noise_std_ > 0.0) y += static_cast<float>(noise_std_) * dist_norm_(rng_);
        msg.data[n] = static_cast<float>(gain) * y;

        // update delay for next sample to inject Doppler
        Dn += dD;
      }
    }

    pub_.publish(msg);
  }

  // --- Members ---
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  event::ConnectionPtr update_conn_;

  std::unique_ptr<ros::NodeHandle> nh_;
  ros::Publisher pub_, pub_block_, pub_debug_tau_, pub_debug_r_, pub_vr_;

  // Links
  physics::LinkPtr hydro_link_;
  physics::LinkPtr source_link_;
  std::string hydro_link_name_{"link"};
  std::string source_link_name_{"link"};

  // Params
  std::string source_model_name_;
  std::string out_topic_;
  std::string signal_type_;
  std::string loss_model_;
  double freq_{150.0};
  double fs_{100000.0};
  double amplitude_{1.0};
  double start_time_{0.0};
  double speed_of_sound_{1482.0};
  int    block_size_{2048};
  double noise_std_{0.0};

  long long last_block_idx_{-1};

  std::mt19937 rng_;
  std::normal_distribution<float> dist_norm_;
  std::unique_ptr<FractionalDelay> fdline_;

  // Tone phase continuity
  double phase_{0.0};
};

GZ_REGISTER_MODEL_PLUGIN(HydrophonePlugin)

} // namespace gazebo

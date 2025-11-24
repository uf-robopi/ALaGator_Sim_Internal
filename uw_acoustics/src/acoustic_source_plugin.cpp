// =====================================
// Author : Adnan Abdullah
// Email: adnanabdullah@ufl.edu
// =====================================

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Events.hh>
#include <ros/ros.h>

namespace gazebo {

class AcousticSourcePlugin : public ModelPlugin {
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override {
    model_ = model;

    // Parameters (readable by hydrophones via SDF on their side)
    signal_type_ = sdf->Get<std::string>("signal_type", "tone").first; // "tone" or "noise"
    freq_        = sdf->Get<double>("freq", 150.0).first;
    fs_          = sdf->Get<double>("fs", 100000.0).first;
    amplitude_   = sdf->Get<double>("amplitude", 1.0).first;
    start_time_  = sdf->Get<double>("start_time", 0.0).first;

    ROS_INFO_STREAM("[AcousticSource] '" << model_->GetName()
                    << "' type=" << signal_type_ << " f=" << freq_
                    << " fs=" << fs_ << " amp=" << amplitude_
                    << " start=" << start_time_);

    // No periodic work needed; hydrophones query pose & params via their own SDF.
  }

  // Optional getters for other plugins (not used directly here)
  const std::string& SignalType() const { return signal_type_; }
  double Freq() const { return freq_; }
  double Fs() const { return fs_; }
  double Amplitude() const { return amplitude_; }
  double StartTime() const { return start_time_; }

private:
  physics::ModelPtr model_;
  std::string signal_type_;
  double freq_, fs_, amplitude_, start_time_;
};

GZ_REGISTER_MODEL_PLUGIN(AcousticSourcePlugin)
} // namespace gazebo

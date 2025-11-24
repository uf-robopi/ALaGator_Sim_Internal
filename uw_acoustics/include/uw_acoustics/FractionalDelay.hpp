#pragma once
#include <vector>
#include <cmath>

class FractionalDelay {
public:
  explicit FractionalDelay(size_t maxDelaySamples = 100000)
  : buf_(maxDelaySamples + 16, 0.0f), idx_(0) {}

  inline void push(float x) {
    buf_[idx_] = x;
    idx_ = (idx_ + 1) % buf_.size();
  }

  // D can be non-integer (e.g., 12.37 samples)
  inline float readDelayed(double D) const {
    // 4-tap Lagrange, centered near D
    int iD = static_cast<int>(std::floor(D));
    double mu = D - iD; // fractional
    // Lagrange coeffs for 4 taps (order-3)
    // indices: -1, 0, 1, 2 around the target
    double c0 = -mu*(mu-1)*(mu-2)/6.0;
    double c1 =  (mu+1)*(mu-1)*(mu-2)/2.0;
    double c2 = - (mu+1)*mu*(mu-2)/2.0;
    double c3 =  (mu+1)*mu*(mu-1)/6.0;

    auto at = [&](int n)->float {
      long ofs = static_cast<long>(idx_) - (iD + n);
      while (ofs < 0) ofs += buf_.size();
      ofs %= buf_.size();
      return buf_[ofs];
    };
    return static_cast<float>(c0*at( -1 ) + c1*at( 0 ) + c2*at( 1 ) + c3*at( 2 ));
  }

private:
  std::vector<float> buf_;
  size_t idx_;
};

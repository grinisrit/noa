#pragma once

#include <algorithm>

#include "spline.h"
#include "configs.h"

inline const double EPS = 1E-3;
inline const double TOL = 1E-9;

using VolMatrix = std::vector<std::vector<double>>;

inline double normal_cdf(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}


class VolSurface {

  double clip_ttm(double ttm) const {
    ttm = std::max(ttm, ttms_[0]); // assuming first quotes are EOD current date 
    ttm = std::min(ttm, ttms_.back());
    return ttm;
  }

  double clip_F(double F) const {
    F = std::max(F, min_K_); 
    F = std::min(F, max_K_);
    return F;
  }

  std::tuple<size_t, size_t> get_ttm_interval(double ttm) const {
    ttm = std::min(ttm, ttms_.back());
    size_t t1 = 0; 
    size_t t2 = 1;

    for(size_t i = 1; i < n_ttms_; i++) {
      const auto& ttm_ = ttms_[i];
      if (ttm_ > ttm) {
        t1 = i-1;
        t2 = i;
        break;
      }
    }
    return std::make_tuple(t1, t2);
  }

  double compute_sigma(double ttm, size_t t1, size_t t2, double F, const tk::spline& strike_splines) const {
    const double sigma1 = std::max(strike_splines(t1, F), TOL); // const fwd yield
    const double sigma2 = std::max(strike_splines(t2, F), TOL);
    const double ttm1 = ttms_[t1]; 
    const double ttm2 = ttms_[t2];
    const double dttm = ttm2 - ttm1;


    const double tot_var = (ttm2 * sigma2 * sigma2) * (ttm2 - ttm) / dttm + 
      (ttm1 * sigma1 * sigma1) * (ttm - ttm1) / dttm;

    return sqrt(tot_var / ttm);
  }

  double bsm_pv(double ttm, double K, double sigma) const {
    const double d1 = (log(fwd_ / K) + (sigma*sigma / 2) * ttm) / (sigma * sqrt(ttm));
    const double d2 = d1 - sigma * sqrt(ttm);
    return fwd_ * normal_cdf(d1) - K * normal_cdf(d2);
  }

  double compute_C(double ttm, double F, const tk::spline& strike_splines) const {
    const auto& [t1, t2] = get_ttm_interval(ttm);
    const double sigma = compute_sigma(ttm, t1, t2, F, strike_splines);
    return bsm_pv(ttm, F, sigma);
  }

  double finite_diff_dupire(double ttm, double F, const tk::spline& strike_splines) const {
    const double C = compute_C(ttm, F, strike_splines);
    const double C_Kl = compute_C(ttm, clip_F(F - EPS), strike_splines);
    const double C_Ku = compute_C(ttm, clip_F(F + EPS), strike_splines);
    const double d2C = C_Ku - 2 * C + C_Kl;
    const double d2C_dK2 = std::max(d2C / (EPS*EPS), TOL); // butterly arbitrage

    const double C_tl = compute_C(clip_ttm(ttm - EPS), F, strike_splines);
    const double C_tu = compute_C(clip_ttm(ttm + EPS), F, strike_splines);
    const double dC = C_tu - C_tl;
    const double dC_dT = std::max(dC / EPS, TOL); // calendar arbitrage 

    const double loc_vol2 = (2*dC_dT) / (F*F*d2C_dK2);
    
    return sqrt(loc_vol2);
  }

  public:
  VolSurface(
    const MarketDataConfig& market_config, 
    const VolMatrix& sigmas) : fwd_(market_config.fwd), n_ttms_(market_config.ttms.size()) {
  
  ttms_.reserve(n_ttms_);
  for(const auto& ttm: market_config.ttms) {
    ttms_.push_back(ttm);
  }

  max_K_ = market_config.strikes.back();
  min_K_ = market_config.strikes.front();
  }

  double get_local_vol(double K, double ttm, const tk::spline& strike_splines) const {
    ttm = clip_ttm(ttm);
    K = clip_F(K);
    return finite_diff_dupire(ttm, K, strike_splines);
  }

  double flat_forward() const {
    return fwd_;
  }

  private:
  std::vector<double> ttms_;
  double fwd_;
  double max_K_;
  double min_K_;
  size_t n_ttms_;
};

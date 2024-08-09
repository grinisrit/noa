#pragma once

#include <vector>
#include <numeric>
#include <variant>

#include "configs.h"
#include "vol_surface.h"

#include "spline.h"
#include "deviates.h"

template<class PayoffImpl> 
struct Payoff {

  double premium(double S) const {
    return static_cast< const PayoffImpl*>(this)->premium_impl(S);
  }

  size_t expiry() const {
    return static_cast<const PayoffImpl*>(this)->expiry_impl();
  }

};

class EuropeanCall : public Payoff<EuropeanCall>{
  double strike_;
  int64_t notional_;
  size_t ttm_;
public:
  EuropeanCall(double strike, int64_t notional, size_t ttm) : strike_(strike), notional_(notional),  ttm_(ttm) {}
  double premium_impl(double S) const {
    return notional_ * std::max(S - strike_, 0.);
  }

  size_t expiry_impl() const {
    return ttm_;
  }
};

class EuropeanPut : public Payoff<EuropeanPut>{
  double strike_;
  int64_t notional_;
  size_t ttm_;
public:
  EuropeanPut(double strike, int64_t notional, size_t ttm) : strike_(strike), notional_(notional), ttm_(ttm) {}
  double premium_impl(double S) const {
    return notional_ * std::max(strike_ - S, 0.);
  }

  size_t expiry_impl() const {
    return ttm_;
  }
};

using Vanilla = std::variant<EuropeanCall, EuropeanPut>;
using Trade = std::vector<Vanilla>;


inline const double DT = 2.7E-3;


double calc_pv(const VolMatrix& sigmas, 
  const MarketDataConfig& market_config, 
  const ModelConfig& model_config,
  const Trade& trade)
{
  tk::mat_double strikes;
  strikes.reserve(1);
  strikes.push_back(market_config.strikes);

  const tk::spline strike_splines = tk::spline{
    strikes, sigmas
    };

  const VolSurface vol_surface = VolSurface(market_config, sigmas);

  Normaldev normal_dev{0., 1., 10};

  std::vector<double> pv(model_config.N, 0.);
  for (size_t n = 0; n < model_config.N; n++) {
    double S = vol_surface.flat_forward();
    double T = 0.;
    for(size_t ttm = 0; ttm < model_config.TTM; ttm++) {
      T = T + DT;
      S = S + vol_surface.get_local_vol(S,T, strike_splines) * S * sqrt(DT) * normal_dev.dev();

      // Cashflows
      for(const auto& vanilla: trade) {
        std::visit(
          [&](const auto& payoff) {
            if (payoff.expiry() == ttm){
              pv[n] += payoff.premium(S);
            }
          },
          vanilla
        );
      }
    }    
  }


  return std::accumulate(pv.begin(), pv.end(), 0.) /model_config.N;              

}

#include <iostream>
#include <vector>

#include "rapidcsv.h"
#include "euler_calc.h"
#include "c-enzyme.h"


auto main(int argc, char *argv[]) -> int {
  size_t N_PATHS = 1000;
  size_t N_DAYS = 300;
  if (argc > 1) {
    N_PATHS = (size_t) std::atoi(argv[1]);
    if (argc > 2) {
      N_DAYS = std::min((size_t) std::atoi(argv[2]), N_DAYS);
    }
  }

  rapidcsv::Document fwd_csv("fwd.csv", rapidcsv::LabelParams(-1, -1));
  std::vector<double> ttms = fwd_csv.GetColumn<double>(0);
  const size_t n_ttms = ttms.size();

  double fwd = fwd_csv.GetColumn<double>(1).at(0);

  rapidcsv::Document impl_vol_csv("impl_vol.csv", rapidcsv::LabelParams(-1, -1));

  std::vector<double> strikes = impl_vol_csv.GetColumn<double>(0);
  const size_t n_strikes = strikes.size();
  
  VolMatrix sigmas;
  sigmas.reserve(n_ttms);
  for (size_t col = 1; col < n_ttms + 1; col++) {
    std::vector<double> smile = impl_vol_csv.GetColumn<double>(col);
    sigmas.push_back(smile);
  }

  const MarketDataConfig market_config = MarketDataConfig{ttms, strikes, fwd};
  
  double Kc = 1.1 * fwd;
  double Kp = 0.9 * fwd;
  size_t TTMc = 2 * (N_DAYS / 3);
  size_t TTMp = 4 * (N_DAYS / 5);

  Trade trade; trade.reserve(2);
  trade.push_back(EuropeanCall(Kc, 10., TTMc));
  trade.push_back(EuropeanPut(Kc, 10., TTMp));

  const auto model_config = ModelConfig{N_PATHS, N_DAYS};

  double pv = calc_pv(sigmas, market_config, model_config, trade);
  std::cout << "PV: " << pv << std::endl;

  VolMatrix vegas;
  vegas.reserve(n_ttms);
  for (size_t i = 0; i < n_ttms; i++) {
    vegas.push_back(std::vector<double>(n_strikes, 0.));
  }

  
  __enzyme_autodiff(calc_pv, 
      enzyme_dup, &sigmas, &vegas,
      enzyme_const, &market_config, 
      enzyme_const, &model_config,
      enzyme_const, &trade);
  
  
  std::cout << "Vegas:\n" << 
    vegas[0][0] << " " <<  vegas[0][1] <<  "...\n" <<
    vegas[1][0] << " " <<  vegas[1][1] <<  "..." << std::endl;

  std::ofstream vegas_csv("vegas.csv");
  for(const auto& vega_t: vegas) {
    for(const auto& vega: vega_t) {
      vegas_csv << vega << ",";
    }
    vegas_csv << "\n";
  }
  vegas_csv.close();

}
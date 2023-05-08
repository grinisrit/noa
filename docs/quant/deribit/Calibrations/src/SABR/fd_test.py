@nb.njit()
def test_black_scholes(strikes, implied_vols, forward, tenor, eps=.0001):
    n = len(strikes)
    delta_err = np.zeros(n, dtype=np.float64)
    vega_err = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bump_up = black_scholes_pv(
                sigma=implied_vols[i], S0=forward+eps, K=strikes[i], T=tenor)
        bump_down = black_scholes_pv(
                sigma=implied_vols[i], S0=forward-eps, K=strikes[i], T=tenor)
        delta_fd =  (bump_up - bump_down) / (2*eps)
        delta_bsm = black_scholes_delta(implied_vols[i], forward, strikes[i], tenor)
        delta_err[i] = (delta_fd - delta_bsm) / delta_bsm
        
        bump_up = black_scholes_pv(
                sigma=implied_vols[i]+eps, S0=forward, K=strikes[i], T=tenor)
        bump_down = black_scholes_pv(
                sigma=implied_vols[i]-eps, S0=forward-eps, K=strikes[i], T=tenor)
        vega_fd =  (bump_up - bump_down) / (2*eps)
        vega_bsm = black_scholes_vega(implied_vols[i], forward, strikes[i], tenor)
        vega_err[i] = (delta_fd - delta_bsm) / delta_bsm
        
        bump_up = black_scholes_pv(
                sigma=implied_vols[i]+eps, S0=forward, K=strikes[i], T=tenor)
        bump_down = black_scholes_pv(
                sigma=implied_vols[i]-eps, S0=forward, K=strikes[i], T=tenor)
        vega_fd =  (bump_up - bump_down) / (2*eps)
        vega_bsm = black_scholes_vega(implied_vols[i], forward, strikes[i], tenor)
            vega_err[i] = (vega_fd - vega_bsm) / vega_bsm
        #gamma_bsm = black_scholes_gamma(sigma, f, K, T, r=0.)
        
    return np.abs(delta_err).max(), np.abs(vega_err).max()


@nb.njit()
def finite_delta(strikes, forward, calibrated_params, tenor, backbone, eps=.0001):
    n = len(strikes)
    fin_delta = np.zeros(n, dtype=np.float64)
    up_vols,_,_ = vol_sabr(
        forward + eps,
        tenor,
        strikes,
        backbone,
        calibrated_params)
    down_vols,_,_ = vol_sabr(
        forward - eps,
        tenor,
        strikes,
        backbone,
        calibrated_params)
    for i in range(n):
        bump_up = black_scholes_pv(
                sigma=up_vols[i], S0=forward+eps, K=strikes[i], T=tenor, r=0., is_call=True)
        bump_down = black_scholes_pv(
                sigma=down_vols[i], S0=forward-eps, K=strikes[i], T=tenor, r=0., is_call=True)
        fin_delta[i] = (bump_up - bump_down) / (2*eps)
        
    return fin_delta
np.abs(delta - 
     finite_delta(test_strikes, new_forward, new_calibrated_params, tenor, backbone, eps=.1)
).max()

@nb.njit()
def finite_vega(strikes, forward, calibrated_params, tenor, backbone, eps=.0001):
    n = len(strikes)
    fin_vega = np.zeros(n, dtype=np.float64)
    up_alpha = calibrated_params.copy()
    up_alpha[0] += eps
    down_alpha = calibrated_params.copy()
    down_alpha[0] -= eps
    up_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        up_alpha)
    down_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        down_alpha)
    for i in range(n):
        bump_up = black_scholes_pv(
            sigma=up_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        bump_down = black_scholes_pv(
            sigma=down_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        fin_vega[i] = (bump_up - bump_down) / (2*eps)
        
    return fin_vega
np.abs((
    vega -
        finite_vega(test_strikes, new_forward, new_calibrated_params, tenor, backbone, eps=.1)
    ) / vega
).max()

@nb.njit()
def finite_rega(strikes, forward, calibrated_params, tenor, backbone, eps=.0001):
    n = len(strikes)
    fin_rega = np.zeros(n, dtype=np.float64)
    up_rho = calibrated_params.copy()
    up_rho[1] += eps
    down_rho = calibrated_params.copy()
    down_rho[1] -= eps
    up_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        up_rho)
    down_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        down_rho)
    for i in range(n):
        bump_up = black_scholes_pv(
            sigma=up_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        bump_down = black_scholes_pv(
            sigma=down_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        fin_rega[i] = (bump_up - bump_down) / (2*eps)
        
    return fin_rega
np.abs((
    rega -
        finite_rega(test_strikes, new_forward, new_calibrated_params, tenor, backbone)
    ) / rega
).max()

@nb.njit()
def finite_sega(strikes, forward, calibrated_params, tenor, backbone, eps=.0001):
    n = len(strikes)
    fin_sega = np.zeros(n, dtype=np.float64)
    up_v = calibrated_params.copy()
    up_v[2] += eps
    down_v = calibrated_params.copy()
    down_v[2] -= eps
    up_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        up_v)
    down_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes,
        backbone,
        down_v)
    for i in range(n):
        bump_up = black_scholes_pv(
            sigma=up_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        bump_down = black_scholes_pv(
            sigma=down_vols[i], S0=forward, K=strikes[i], T=tenor, r=0., is_call=True)
        fin_sega[i] = (bump_up - bump_down) / (2*eps)
    return fin_sega
np.abs((
    sega -
        finite_sega(test_strikes, new_forward, new_calibrated_params, tenor, backbone)
) / sega
).max()

@nb.njit()
def finite_kega(strikes, forward, calibrated_params, tenor, backbone, eps=.0001):
    up_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes+eps,
        backbone,
        calibrated_params)
    down_vols,_,_ = vol_sabr(
        forward,
        tenor,
        strikes - eps,
        backbone,
        calibrated_params)
    fin_kega = (up_vols - down_vols) / (2*eps)          
    return (up_vols - down_vols) / (2*eps)
np.abs(
    (strikes_grad - 
         finite_kega(test_strikes, new_forward, new_calibrated_params, tenor, backbone)
    ) / strikes_grad
).max()

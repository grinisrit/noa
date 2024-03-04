import pandas as pd
from pyquant.common import *
from pyquant.vol_surface import *

def get_vol_surface(test_file):
    df = pd.read_csv(test_file, header=[0,1], index_col=0)
    
    Ts = pd.DatetimeIndex(df.index).astype(int).values.squeeze()
    Ts = (Ts - Ts[0]) / YEAR_NANOS
    
    types_str, strikes_str = zip(*df.columns[2:].values)
    Ks = np.array([float(x) for x in strikes_str])
    option_types = np.array([True if x == 'call' else False for x in types_str])
    
    spot = Spot(df['swap'].values[0].item())
    df.iloc[:, 2:] *= spot.S
    
    Fs = df['futures'].values.squeeze()
    Fidx = np.nonzero(Fs)
    fwd_curve = ForwardCurve.from_forward_rates(spot, ForwardRates(Fs[Fidx]), TimesToMaturity(Ts[Fidx]))
    
    n_T = len(Ts) - 1
    buf_T = Ts[1:].repeat(len(Ks))
    buf_K = np.tile(Ks, n_T)
    buf_C = np.tile(option_types, n_T)
    buf_pv = df.values[1:,2:].flatten()
    pv_idx = np.nonzero(buf_pv)

    return VolSurfaceChainSpace(
        fwd_curve, 
        TimesToMaturity(buf_T[pv_idx]),
        Strikes(buf_K[pv_idx]),
        OptionTypes(buf_C[pv_idx]),
        Premiums(buf_pv[pv_idx])
    )
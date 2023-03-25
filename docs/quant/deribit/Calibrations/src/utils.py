import pandas as pd
from scipy import stats as sps
import numpy as np
import datetime
import numba as nb


def get_tick(df: pd.DataFrame, timestamp: int = None):
    """Function gets tick for each expiration and strike
    from closest timestamp from given. If timestamp is None, it takes last one."""
    if timestamp:
        data = df[df["timestamp"] <= timestamp].copy()
        # only not expired on curret tick
        data = data[data["expiration"] > timestamp].copy()
    else:
        data = df.copy()
        # only not expired on max available tick
        data = data[data["expiration"] > data["timestamp"].max()].copy()
    # tau is time before expiration in years
    data["tau"] = (data.expiration - data.timestamp) / 1e6 / 3600 / 24 / 365

    data_grouped = data.loc[
        data.groupby(["type", "expiration", "strike_price"])["timestamp"].idxmax()
    ]

    data_grouped = data_grouped[data_grouped["tau"] > 0.0]
    # We need Only out of the money to calibrate
    data_grouped = data_grouped[
        (
            (data_grouped["type"] == "call")
            & (data_grouped["underlying_price"] <= data_grouped["strike_price"])
        )
        | (
            (data_grouped["type"] == "put")
            & (data_grouped["underlying_price"] >= data_grouped["strike_price"])
        )
    ]
    data_grouped["mark_price_usd"] = (
        data_grouped["mark_price"] * data_grouped["underlying_price"]
    )
    data_grouped = data_grouped[data_grouped["strike_price"] <= 10_000]
    return data_grouped

# Newton-Raphsen
nb.njit
def get_implied_volatility(
    option_type: str,
    C: float,
    K: float,
    T: float,
    F: float,
    r: float = 0.0,
    error: float = 0.001,
) -> float:
    """
    Function to count implied volatility via given params of option, using Newton-Raphson method :

    Args:
        C (float): Option market price(USD).
        K (float): Strike(USD).
        T (float): Time to expiration in years.
        F (float): Underlying price.
        r (float): Risk-free rate.
        error (float): Given threshhold of error.

    Returns:
        float: Implied volatility in percent.
    """
    vol = 1.0
    dv = error + 1
    while abs(dv) > error:
        d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        D = np.exp(-r * T)
        if option_type.lower() == "call":
            price = F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2) * D
        elif option_type.lower() == "put":
            price = -F * sps.norm.cdf(-d1) + K * sps.norm.cdf(-d2) * D
        else:
            raise ValueError("Wrong option type, must be 'call' or 'put' ")
        Vega = F * np.sqrt(T / np.pi / 2) * np.exp(-0.5 * d1**2)
        PriceError = price - C
        dv = PriceError / Vega
        vol = vol - dv
    return vol

def process_data(data):
    # only options
    df = data.copy()
    df = df[(df["instrument"].str.endswith("C")) | (df["instrument"].str.endswith("P"))].sort_values("dt")
    df["type"] = np.where(df["instrument"].str.endswith("C"), "call", "put")
    
    perpetuals = data[data["instrument"].str.endswith("PERPETUAL")][["dt", "price"]].copy()
    perpetuals = perpetuals.rename(columns = {"price": "underlying_price"}).sort_values("dt")
    
    def get_strike(x):
        return int(x.split("-")[2])
    
    def get_expiration(x):
        return x.split("-")[1]
    

    df["strike_price"] = df["instrument"].apply(get_strike)
    df["expiration"] = df["instrument"].apply(get_expiration)
    
    def unix_time_millis(dt):
        epoch = datetime.datetime.utcfromtimestamp(0)
        return int((dt - epoch).total_seconds() * 1000_000)
    
    def get_normal_date(s):
        """Function to convert date to find years to maturity"""
        monthToNum = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }

        full_date = s.split("-")[1]
        try:
            day = int(full_date[:2])
            month = monthToNum[full_date[2:5]]
        except:
            day = int(full_date[:1])
            month = monthToNum[full_date[1:4]]
        
        year = int("20" + full_date[-2:])
        exp_date = datetime.datetime(year, month, day)
        return unix_time_millis(exp_date)
    
    df["dt"] = pd.to_datetime(df["dt"])
    perpetuals["dt"] = pd.to_datetime(perpetuals["dt"])
    
    df = pd.merge_asof(df, perpetuals, on="dt",
                       tolerance=pd.Timedelta('7 minutes'),
                       direction='nearest',)
    
    df["timestamp"] = df["dt"].apply(unix_time_millis)
    df["expiration"] = df["instrument"].apply(get_normal_date)
    df = df.rename(columns = {"price": "mark_price"})
    
    
    return df


bid_ask_approx = {
    100: 1.0682687141911174,
    150: 1.0543091799098152,
    200: 1.043041829448211,
    250: 1.0339585615110631,
    300: 1.0266471731161493,
    350: 1.0207732598846047,
    400: 1.0160655324444303,
    450: 1.012303904194882,
    500: 1.0093098273690733,
    550: 1.006938453054222,
    600: 1.0050722709184021,
    650: 1.0036159493661156,
    700: 1.0024921495554828,
    750: 1.0016381294718064,
    800: 1.0010029889434058,
    850: 1.0005454346292013,
    900: 1.0002319668393267,
    950: 1.0000354085726098,
    1000: 1.0,
    1050: 1.0,
    1100: 1.0,
    1150: 1.0000351723918721,
    1200: 1.0001647525653266,
    1250: 1.0003277282334841,
    1275: 1.0004277282334841,
    1300: 1.0005178140988367,
    1350: 1.0007299111473158,
    1400: 1.0009598827508623,
    1450: 1.0012043730280786,
    1500: 1.0014606594872277,
    1550: 1.0017265334811813,
    1600: 1.002000203225129,
    1650: 1.0022802151185861,
    1700: 1.0025653899169722,
    1750: 1.0028547709500737,
    1800: 1.003147582113679,
    1850: 1.003443193789814,
    1900: 1.0037410951991417,
    1950: 1.0040408719715295,
    2000: 1.0043421879499168,
    2050: 1.0046447704284949,
    2100: 1.0049483981770093,
    2150: 1.00525289172534,
    2200: 1.005558105481755,
    2250: 1.0058639213387546,
    2300: 1.0061702434857376,
    2350: 1.006476994200719,
    2400: 1.0067841104363124,
    2450: 1.0070915410500683,
    2500: 1.0073992445575557,
    2550: 1.0077071873095236,
    2600: 1.0080153420131015,
    2650: 1.0083236865321064,
    2700: 1.00863220291378,
    2750: 1.0089408765992163,
    2800: 1.0092496957828119,
    2850: 1.009558650892614,
    2900: 1.0098677341687459,
    2950: 1.0101769393214,
    3000: 1.0104862612533803,
    3050: 1.0107956958350135,
    3100: 1.0111052397215405,
    3150: 1.011414890204975,
    3200: 1.0117246450939223,
    3250: 1.0120345026160786,
    3300: 1.0123444613391364,
    3350: 1.0126545201066126,
    3400: 1.0129646779857948,
    3450: 1.0132749342255074,
    3500: 1.0135852882218488,
    3550: 1.0138957394903998,
    3600: 1.0142062876436715,
    3650: 1.0145169323728145,
    3700: 1.0148276734327777,
    3750: 1.0151385106302677,
    3800: 1.0154494438139845,
    3850: 1.0157604728666931,
    3900: 1.0160715976988008,
    3950: 1.0163828182431383,
    4000: 1.0166941344507299,
    4050: 1.017005546287362,
    4100: 1.0173170537307983,
    4150: 1.0176286567685209,
    4200: 1.017940355395898,
    4250: 1.0182521496146968,
    4300: 1.0185640394318767,
    4350: 1.0188760248586137,
    4400: 1.0191881059095027,
    4450: 1.019500282601917,
    4500: 1.0198125549554835,
    4550: 1.0201249229916602,
    4600: 1.0204373867333918,
    4650: 1.020749946204831,
    4700: 1.0210626014311126,
    4750: 1.021375352438168,
    4800: 1.0216881992525795,
    4850: 1.022001141901457,
    4900: 1.0223141804123401,
    4950: 1.0226273148131204,
    5000: 1.0229405451319749,
    5050: 1.0232538713973145,
    5100: 1.0235672936377422,
    5150: 1.0238808118820173,
    5200: 1.0241944261590292,
    5250: 1.024508136497773,
    5300: 1.024821942927332,
    5350: 1.0251358454768627,
    5400: 1.0254498441755824,
    5450: 1.02576393905276,
    5500: 1.0260781301377078,
    5550: 1.0263924174597738,
    5600: 1.0267068010483393,
    5650: 1.0270212809328116,
    5700: 1.027335857142622,
    5750: 1.0276505297072238,
    5800: 1.0279652986560879,
    5850: 1.0282801640187025,
    5900: 1.0285951258245716,
    5950: 1.0289101841032133,
    6000: 1.0292253388841592,
    6050: 1.029540590196953,
    6100: 1.0298559380711505,
    6150: 1.0301713825363183,
    6200: 1.0304869236220346,
    6250: 1.030802561357888,
    6300: 1.0311182957734768,
    6350: 1.0314341268984102,
    6400: 1.0317500547623064,
    6450: 1.0320660793947938,
    6500: 1.0323822008255106,
    6550: 1.0326984190841038,
    6600: 1.033014734200231,
    6650: 1.0333311462035577,
    6700: 1.03364765512376,
    6750: 1.0339642609905217,
    6800: 1.0342809638335382,
    6850: 1.034597763682512,
    6900: 1.0349146605671555,
    6950: 1.0352316545171907,
    7000: 1.0355487455623482,
    7050: 1.0358659337323681,
    7100: 1.0361832190569993,
    7150: 1.0365006015660005,
    7200: 1.0368180812891388,
    7250: 1.0371356582561906,
    7300: 1.0374533324969417,
    7350: 1.0377711040411872,
    7400: 1.0380889729187308,
    7450: 1.038406939159386,
    7500: 1.0387250027929744,
    7550: 1.0390431638493276,
    7600: 1.0393614223582865,
    7650: 1.039679778349701,
    7700: 1.0399982318534284,
    7750: 1.0403167828993385,
    7800: 1.0406354315173074,
    7850: 1.040954177737222,
    7900: 1.0412730215889767,
    7950: 1.0415919631024773,
    8000: 1.0419110023076368,
    8050: 1.0422301392343782,
    8100: 1.0425493739126337,
    8150: 1.0428687063723447,
    8200: 1.0431881366434614,
    8250: 1.0435076647559436,
    8300: 1.0438272907397597,
    8350: 1.0441470146248877,
    8400: 1.0444668364413152,
    8450: 1.0447867562190383,
    8500: 1.0451067739880622,
    8550: 1.045426889778402,
    8600: 1.0457471036200812,
    8650: 1.0460674155431333,
    8700: 1.0463878255776002,
    8750: 1.0467083337535337,
    8800: 1.0470289401009945,
    8850: 1.047349644650052,
    8900: 1.047670447430786,
    8950: 1.0479913484732843,
    9000: 1.0483123478076446,
    9050: 1.0486334454639736,
    9100: 1.0489546414723878,
    9150: 1.0492759358630117,
    9200: 1.04959732866598,
    9250: 1.0499188199114364,
    9300: 1.0502404096295337,
    9350: 1.0505620978504344,
    9400: 1.0508838846043091,
    9450: 1.0512057699213393,
    9500: 1.0515277538317143,
    9550: 1.0518498363656332,
    9600: 1.0521720175533045,
    9650: 1.052494297424946,
    9700: 1.0528166760107844,
    9750: 1.0531391533410555,
    9800: 1.053461729446005,
    9850: 1.0537844043558875,
    9900: 1.054107178100967,
    9950: 1.0544300507115165,
}


def get_bid_ask(strike):
    return bid_ask_approx[strike]

def round_params(params, n_signs = 3):
    return [round(x, n_signs) for x in params]
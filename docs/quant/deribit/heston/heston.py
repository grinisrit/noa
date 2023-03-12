import numba as nb
import numpy as np
from typing import Final, Tuple
import pandas as pd
from levenberg_marquardt import LevenbergMarquardt
from typing import Union
from sklearn.linear_model import LinearRegression
from scipy import stats as sps


_spec_market_params = [
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64[:]),
    ("K", nb.float64[:]),
    ("C", nb.float64[:]),
    ("types", nb.boolean[:]),
]
_spec_model_params = [
    ("a", nb.float64),
    ("b", nb.float64),
    ("c", nb.float64),
    ("rho", nb.float64),
    ("v0", nb.float64),
]
_spec_TagMn = [
    ("M1", nb.float64[:]),
    ("N1", nb.float64[:]),
    ("M2", nb.float64[:]),
    ("N2", nb.float64[:]),
]

_spec_tagMNJac = [
    ("pa1s", nb.float64[:]),
    ("pa2s", nb.float64[:]),
    ("pb1s", nb.float64[:]),
    ("pb2s", nb.float64[:]),
    ("pc1s", nb.float64[:]),
    ("pc2s", nb.float64[:]),
    ("prho1s", nb.float64[:]),
    ("prho2s", nb.float64[:]),
    ("pv01s", nb.float64[:]),
    ("pv02s", nb.float64[:]),
]


@nb.experimental.jitclass(_spec_market_params)
class MarketParameters(object):
    S: nb.float64
    r: nb.float64
    T: nb.float64[:]
    K: nb.float64[:]
    C: nb.float64[:]
    types: nb.boolean[:]

    def __init__(
        self,
        S: nb.float64,
        r: nb.float64,
        T: nb.float64[:],
        K: nb.float64[:],
        C: nb.float64[:],
        types: nb.boolean[:],
    ):
        self.S = S
        self.r = r
        self.T = T
        self.K = K
        self.C = C
        self.types = types


@nb.experimental.jitclass(_spec_model_params)
class ModelParameters(object):
    a: nb.float64
    b: nb.float64
    c: nb.float64
    rho: nb.float64
    v0: nb.float64

    def __init__(
        self,
        a: nb.float64,
        b: nb.float64,
        c: nb.float64,
        rho: nb.float64,
        v0: nb.float64,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.rho = rho
        self.v0 = v0


@nb.experimental.jitclass(_spec_TagMn)
class TagMn(object):
    M1: nb.float64[:]
    N1: nb.float64[:]
    M2: nb.float64[:]
    N2: nb.float64[:]

    def __init__(
        self, M1: nb.float64[:], N1: nb.float64[:], M2: nb.float64[:], N2: nb.float64[:]
    ):
        self.M1 = M1
        self.M2 = M2
        self.N1 = N1
        self.N2 = N2


@nb.experimental.jitclass(_spec_tagMNJac)
class tagMNJac(object):
    pa1s: nb.float64[:]
    pa2s: nb.float64[:]
    pb1s: nb.float64[:]
    pb2s: nb.float64[:]
    pc1s: nb.float64[:]
    pc2s: nb.float64[:]
    prho1s: nb.float64[:]
    prho2s: nb.float64[:]
    pv01s: nb.float64[:]
    pv02s: nb.float64[:]

    def __init__(
        self,
        pa1s: nb.float64[:],
        pa2s: nb.float64[:],
        pb1s: nb.float64[:],
        pb2s: nb.float64[:],
        pc1s: nb.float64[:],
        pc2s: nb.float64[:],
        prho1s: nb.float64[:],
        prho2s: nb.float64[:],
        pv01s: nb.float64[:],
        pv02s: nb.float64[:],
    ):
        self.pa1s = pa1s
        self.pa2s = pa2s
        self.pb1s = pb1s
        self.pb2s = pb2s
        self.pc1s = pc1s
        self.pc2s = pc2s
        self.prho1s = prho1s
        self.prho2s = prho2s
        self.pv01s = pv01s
        self.pv02s = pv02s


one, zero, two = (
    np.complex128(complex(1.0, 0.0)),
    np.complex128(complex(0.0, 0.0)),
    np.complex128(complex(2.0, 0.0)),
)
i = np.complex128(complex(0.0, 1.0))


pi: Final[np.float64] = np.pi
lb: Final[np.float64] = np.float64(0.0)
ub: Final[np.int32] = np.int32(200)
Q: Final[np.float64] = np.float64(0.5 * (ub - lb))
P: Final[np.float64] = np.float64(0.5 * (ub + lb))

# точки, в которых считаеются квадратуры
u64 = np.array(
    [
        0.0243502926634244325089558,
        0.0729931217877990394495429,
        0.1214628192961205544703765,
        0.1696444204239928180373136,
        0.2174236437400070841496487,
        0.2646871622087674163739642,
        0.3113228719902109561575127,
        0.3572201583376681159504426,
        0.4022701579639916036957668,
        0.4463660172534640879849477,
        0.4894031457070529574785263,
        0.5312794640198945456580139,
        0.5718956462026340342838781,
        0.6111553551723932502488530,
        0.6489654712546573398577612,
        0.6852363130542332425635584,
        0.7198818501716108268489402,
        0.7528199072605318966118638,
        0.7839723589433414076102205,
        0.8132653151227975597419233,
        0.8406292962525803627516915,
        0.8659993981540928197607834,
        0.8893154459951141058534040,
        0.9105221370785028057563807,
        0.9295691721319395758214902,
        0.9464113748584028160624815,
        0.9610087996520537189186141,
        0.9733268277899109637418535,
        0.9833362538846259569312993,
        0.9910133714767443207393824,
        0.9963401167719552793469245,
        0.9993050417357721394569056,
    ],
    dtype=np.float64,
)

# веса квадратуры Гаусса, от 0 до 1, т.к. интеграл нужен от 0
w64 = np.array(
    [
        0.0486909570091397203833654,
        0.0485754674415034269347991,
        0.0483447622348029571697695,
        0.0479993885964583077281262,
        0.0475401657148303086622822,
        0.0469681828162100173253263,
        0.0462847965813144172959532,
        0.0454916279274181444797710,
        0.0445905581637565630601347,
        0.0435837245293234533768279,
        0.0424735151236535890073398,
        0.0412625632426235286101563,
        0.0399537411327203413866569,
        0.0385501531786156291289625,
        0.0370551285402400460404151,
        0.0354722132568823838106931,
        0.0338051618371416093915655,
        0.0320579283548515535854675,
        0.0302346570724024788679741,
        0.0283396726142594832275113,
        0.0263774697150546586716918,
        0.0243527025687108733381776,
        0.0222701738083832541592983,
        0.0201348231535302093723403,
        0.0179517157756973430850453,
        0.0157260304760247193219660,
        0.0134630478967186425980608,
        0.0111681394601311288185905,
        0.0088467598263639477230309,
        0.0065044579689783628561174,
        0.0041470332605624676352875,
        0.0017832807216964329472961,
    ],
    dtype=np.float64,
)


_tmp_values_HesIntMN = {
    "csqr": nb.float64,
    "PQ_M": nb.float64[:],
    "PQ_N": nb.float64[:],
    "imPQ_M": nb.types.Array(nb.complex128, 1, "C"),
    "imPQ_N": nb.types.Array(nb.complex128, 1, "C"),
    "_imPQ_M": nb.types.Array(nb.complex128, 1, "C"),
    "_imPQ_N": nb.types.Array(nb.complex128, 1, "C"),
    "h_M": nb.types.Array(nb.complex128, 1, "C"),
    "h_N": nb.types.Array(nb.complex128, 1, "C"),
    "x0": nb.float64,
    "tmp": nb.float64,
    "kes_M1": nb.types.Array(nb.complex128, 1, "C"),
    "kes_N1": nb.types.Array(nb.complex128, 1, "C"),
    "kes_M2": nb.types.Array(nb.complex128, 1, "C"),
    "kes_N2": nb.types.Array(nb.complex128, 1, "C"),
    "m_M1": nb.types.Array(nb.complex128, 1, "C"),
    "m_N1": nb.types.Array(nb.complex128, 1, "C"),
    "m_M2": nb.types.Array(nb.complex128, 1, "C"),
    "m_N2": nb.types.Array(nb.complex128, 1, "C"),
    "d_M1": nb.types.Array(nb.complex128, 1, "C"),
    "d_M2": nb.types.Array(nb.complex128, 1, "C"),
    "d_N1": nb.types.Array(nb.complex128, 1, "C"),
    "d_N2": nb.types.Array(nb.complex128, 1, "C"),
    "tmp1": nb.float64,
    "g_M1": nb.types.Array(nb.complex128, 1, "C"),
    "g_N1": nb.types.Array(nb.complex128, 1, "C"),
    "g_M2": nb.types.Array(nb.complex128, 1, "C"),
    "g_N2": nb.types.Array(nb.complex128, 1, "C"),
    "alpha": nb.types.Array(nb.complex128, 1, "C"),
    "calp_M1": nb.types.Array(nb.complex128, 1, "C"),
    "salp_M1": nb.types.Array(nb.complex128, 1, "C"),
    "calp_M2": nb.types.Array(nb.complex128, 1, "C"),
    "salp_M2": nb.types.Array(nb.complex128, 1, "C"),
    "calp_N1": nb.types.Array(nb.complex128, 1, "C"),
    "salp_ n1": nb.types.Array(nb.complex128, 1, "C"),
    "calp_N2": nb.types.Array(nb.complex128, 1, "C"),
    "salp_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A2_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A2_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A2_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A2_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A1_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A1_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A1_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A1_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A_N2": nb.types.Array(nb.complex128, 1, "C"),
    "halft": nb.float64,
    "D_M1": nb.types.Array(nb.complex128, 1, "C"),
    "D_M2": nb.types.Array(nb.complex128, 1, "C"),
    "D_N1": nb.types.Array(nb.complex128, 1, "C"),
    "D_N2": nb.types.Array(nb.complex128, 1, "C"),
    "MNbas": TagMn.class_type.instance_type,
}


_tmp_values_HesIntJac = {
    "PQ_M": nb.float64[:],
    "PQ_N": nb.float64[:],
    "imPQ_M": nb.types.Array(nb.complex128, 1, "C"),
    "imPQ_N": nb.types.Array(nb.complex128, 1, "C"),
    "_imPQ_M": nb.types.Array(nb.complex128, 1, "C"),
    "_imPQ_N": nb.types.Array(nb.complex128, 1, "C"),
    "h_M": nb.types.Array(nb.complex128, 1, "C"),
    "h_N": nb.types.Array(nb.complex128, 1, "C"),
    "x0": nb.float64,
    "tmp": nb.float64,
    "kes_M1": nb.types.Array(nb.complex128, 1, "C"),
    "kes_N1": nb.types.Array(nb.complex128, 1, "C"),
    "kes_M2": nb.types.Array(nb.complex128, 1, "C"),
    "kes_N2": nb.types.Array(nb.complex128, 1, "C"),
    "_msqr": nb.types.Array(nb.complex128, 1, "C"),
    "_nsqr": nb.types.Array(nb.complex128, 1, "C"),
    "msqr": nb.types.Array(nb.complex128, 1, "C"),
    "nsqr": nb.types.Array(nb.complex128, 1, "C"),
    "m_M1": nb.types.Array(nb.complex128, 1, "C"),
    "m_N1": nb.types.Array(nb.complex128, 1, "C"),
    "m_M2": nb.types.Array(nb.complex128, 1, "C"),
    "m_N2": nb.types.Array(nb.complex128, 1, "C"),
    "csqr": nb.float64,
    "d_M1": nb.types.Array(nb.complex128, 1, "C"),
    "d_M2": nb.types.Array(nb.complex128, 1, "C"),
    "d_N1": nb.types.Array(nb.complex128, 1, "C"),
    "d_N2": nb.types.Array(nb.complex128, 1, "C"),
    "abrt": nb.float64,
    "tmp1": nb.float64,
    "tmp2": nb.float64,
    "g_M1": nb.types.Array(nb.complex128, 1, "C"),
    "g_N1": nb.types.Array(nb.complex128, 1, "C"),
    "g_M2": nb.types.Array(nb.complex128, 1, "C"),
    "g_N2": nb.types.Array(nb.complex128, 1, "C"),
    "halft": nb.float64,
    "alpha": nb.types.Array(nb.complex128, 1, "C"),
    "calp_M1": nb.types.Array(nb.complex128, 1, "C"),
    "salp_M1": nb.types.Array(nb.complex128, 1, "C"),
    "calp_N1": nb.types.Array(nb.complex128, 1, "C"),
    "salp_N1": nb.types.Array(nb.complex128, 1, "C"),
    "calp_M2": nb.types.Array(nb.complex128, 1, "C"),
    "salp_M2": nb.types.Array(nb.complex128, 1, "C"),
    "calp_N2": nb.types.Array(nb.complex128, 1, "C"),
    "salp_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A2_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A2_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A2_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A2_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A1_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A1_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A1_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A1_N2": nb.types.Array(nb.complex128, 1, "C"),
    "A_M1": nb.types.Array(nb.complex128, 1, "C"),
    "A_N1": nb.types.Array(nb.complex128, 1, "C"),
    "A_M2": nb.types.Array(nb.complex128, 1, "C"),
    "A_N2": nb.types.Array(nb.complex128, 1, "C"),
    "B_M1": nb.types.Array(nb.complex128, 1, "C"),
    "B_N1": nb.types.Array(nb.complex128, 1, "C"),
    "B_M2": nb.types.Array(nb.complex128, 1, "C"),
    "B_N2": nb.types.Array(nb.complex128, 1, "C"),
    "tmp3": nb.float64,
    "D_M1": nb.types.Array(nb.complex128, 1, "C"),
    "D_M2": nb.types.Array(nb.complex128, 1, "C"),
    "D_N1": nb.types.Array(nb.complex128, 1, "C"),
    "D_N2": nb.types.Array(nb.complex128, 1, "C"),
    "y1M1": nb.types.Array(nb.complex128, 1, "C"),
    "y1N1": nb.types.Array(nb.complex128, 1, "C"),
    "y1M2": nb.types.Array(nb.complex128, 1, "C"),
    "y1N2": nb.types.Array(nb.complex128, 1, "C"),
    "H_M1": nb.types.Array(nb.complex128, 1, "C"),
    "H_M2": nb.types.Array(nb.complex128, 1, "C"),
    "H_N1": nb.types.Array(nb.complex128, 1, "C"),
    "H_N2": nb.types.Array(nb.complex128, 1, "C"),
    "lnB_M1": nb.types.Array(nb.complex128, 1, "C"),
    "lnB_M2": nb.types.Array(nb.complex128, 1, "C"),
    "lnB_N1": nb.types.Array(nb.complex128, 1, "C"),
    "lnB_N2": nb.types.Array(nb.complex128, 1, "C"),
    "tmp4": nb.float64,
    "tmp5": nb.float64,
    "y3M1": nb.types.Array(nb.complex128, 1, "C"),
    "y3M2": nb.types.Array(nb.complex128, 1, "C"),
    "y3N1": nb.types.Array(nb.complex128, 1, "C"),
    "y3N2": nb.types.Array(nb.complex128, 1, "C"),
    "ctmp": nb.types.Array(nb.complex128, 1, "C"),
    "pd_prho_M1": nb.types.Array(nb.complex128, 1, "C"),
    "pA1_prho_M1": nb.types.Array(nb.complex128, 1, "C"),
    "pA2_prho_M1": nb.types.Array(nb.complex128, 1, "C"),
    "pA_prho_M1": nb.types.Array(nb.complex128, 1, "C"),
    "pB_prho_M1": nb.types.Array(nb.complex128, 1, "C"),
    "y4M1": nb.types.Array(nb.complex128, 1, "C"),
    "pd_prho_N1": nb.types.Array(nb.complex128, 1, "C"),
    "pA1_prho_N1": nb.types.Array(nb.complex128, 1, "C"),
    "pA2_prho_N1": nb.types.Array(nb.complex128, 1, "C"),
    "pA_prho_N1": nb.types.Array(nb.complex128, 1, "C"),
    "pB_prho_N1": nb.types.Array(nb.complex128, 1, "C"),
    "y4N1": nb.types.Array(nb.complex128, 1, "C"),
    "pd_prho_M2": nb.types.Array(nb.complex128, 1, "C"),
    "pA1_prho_M2": nb.types.Array(nb.complex128, 1, "C"),
    "pA2_prho_M2": nb.types.Array(nb.complex128, 1, "C"),
    "pA_prho_M2": nb.types.Array(nb.complex128, 1, "C"),
    "pB_prho_M2": nb.types.Array(nb.complex128, 1, "C"),
    "y4M2": nb.types.Array(nb.complex128, 1, "C"),
    "pd_prho_N2": nb.types.Array(nb.complex128, 1, "C"),
    "pA1_prho_N2": nb.types.Array(nb.complex128, 1, "C"),
    "pA2_prho_N2": nb.types.Array(nb.complex128, 1, "C"),
    "pA_prho_N2": nb.types.Array(nb.complex128, 1, "C"),
    "pB_prho_N2": nb.types.Array(nb.complex128, 1, "C"),
    "y4N2": nb.types.Array(nb.complex128, 1, "C"),
    "pB_pa": nb.types.Array(nb.complex128, 1, "C"),
    "y5M1": nb.types.Array(nb.complex128, 1, "C"),
    "y5M2": nb.types.Array(nb.complex128, 1, "C"),
    "y5N1": nb.types.Array(nb.complex128, 1, "C"),
    "y5N2": nb.types.Array(nb.complex128, 1, "C"),
    "pd_pc": nb.types.Array(nb.complex128, 1, "C"),
    "pA1_pc": nb.types.Array(nb.complex128, 1, "C"),
    "pA2_pc": nb.types.Array(nb.complex128, 1, "C"),
    "pA_pc": nb.types.Array(nb.complex128, 1, "C"),
    "y6M1": nb.types.Array(nb.complex128, 1, "C"),
    "y6M2": nb.types.Array(nb.complex128, 1, "C"),
    "y6N1": nb.types.Array(nb.complex128, 1, "C"),
    "y6N2": nb.types.Array(nb.complex128, 1, "C"),
    "hM1": nb.types.Array(nb.complex128, 1, "C"),
    "hM2": nb.types.Array(nb.complex128, 1, "C"),
    "hN1": nb.types.Array(nb.complex128, 1, "C"),
    "hN2": nb.types.Array(nb.complex128, 1, "C"),
    "pa1s": nb.types.Array(nb.float64, 1, "A"),
    "pa2s": nb.types.Array(nb.float64, 1, "A"),
    "pb1s": nb.types.Array(nb.float64, 1, "A"),
    "pb2s": nb.types.Array(nb.float64, 1, "A"),
    "pc1s": nb.types.Array(nb.float64, 1, "A"),
    "pc2s": nb.types.Array(nb.float64, 1, "A"),
    "prho1s": nb.types.Array(nb.float64, 1, "A"),
    "prho2s": nb.types.Array(nb.float64, 1, "A"),
    "pv01s": nb.types.Array(nb.float64, 1, "A"),
    "pv02s": nb.types.Array(nb.float64, 1, "A"),
    "Jacobian": tagMNJac.class_type.instance_type,
}


_signature_HesIntMN = TagMn.class_type.instance_type(
    ModelParameters.class_type.instance_type,
    MarketParameters.class_type.instance_type,
    nb.int32,
)

_signature_HesIntJac = tagMNJac.class_type.instance_type(
    ModelParameters.class_type.instance_type,
    MarketParameters.class_type.instance_type,
    nb.int32,
)


@nb.njit(_signature_HesIntMN, locals=_tmp_values_HesIntMN)
def HesIntMN(
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    market_pointer: int,
):
    """
    :param model_parameters:
    :param market_parameters:
    :param market_pointer:
    :return:
    """
    csqr = np.power(model_parameters.c, 2)

    PQ_M, PQ_N = P + Q * u64, P - Q * u64
    imPQ_M = i * PQ_M
    imPQ_N = i * PQ_N
    _imPQ_M = i * (PQ_M - i)
    _imPQ_N = i * (PQ_N - i)

    h_M = np.divide(np.power(market_parameters.K[market_pointer], -imPQ_M), imPQ_M)
    h_N = np.divide(np.power(market_parameters.K[market_pointer], -imPQ_N), imPQ_N)

    x0 = (
        np.log(market_parameters.S)
        + market_parameters.r * market_parameters.T[market_pointer]
    )
    tmp = model_parameters.c * model_parameters.rho

    kes_M1 = model_parameters.a - np.multiply(tmp, _imPQ_M)
    kes_N1 = model_parameters.a - np.multiply(tmp, _imPQ_N)
    kes_M2 = kes_M1 + tmp
    kes_N2 = kes_N1 + tmp

    m_M1 = imPQ_M + one + np.power(PQ_M - i, 2)
    m_N1 = imPQ_N + one + np.power(PQ_N - i, 2)
    m_M2 = imPQ_M + np.power(PQ_M - zero * i, 2)
    m_N2 = imPQ_N + np.power(PQ_N - zero * i, 2)

    d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
    d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
    d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
    d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

    tmp1 = (
        -model_parameters.a
        * model_parameters.b
        * model_parameters.rho
        * market_parameters.T[market_pointer]
        / model_parameters.c
    )

    # вот эта строка была пропущено и цены плохо считались 16.02.2023
    tmp = np.exp(tmp1)
    g_M2 = np.exp(tmp1 * imPQ_M)
    g_N2 = np.exp(tmp1 * imPQ_N)
    g_M1 = g_M2 * tmp
    g_N1 = g_N2 * tmp

    tmp = 0.5 * market_parameters.T[market_pointer]
    alpha = d_M1 * tmp
    calp_M1 = np.cosh(alpha)
    salp_M1 = np.sinh(alpha)

    alpha = d_N1 * tmp
    calp_N1 = np.cosh(alpha)
    salp_N1 = np.sinh(alpha)

    alpha = d_M2 * tmp
    calp_M2 = np.cosh(alpha)
    salp_M2 = np.sinh(alpha)

    alpha = d_N2 * tmp
    calp_N2 = np.cosh(alpha)
    salp_N2 = np.sinh(alpha)

    A2_M1 = np.multiply(d_M1, calp_M1) + np.multiply(kes_M1, salp_M1)
    A2_N1 = np.multiply(d_N1, calp_N1) + np.multiply(kes_N1, salp_N1)
    A2_M2 = np.multiply(d_M2, calp_M2) + np.multiply(kes_M2, salp_M2)
    A2_N2 = np.multiply(d_N2, calp_N2) + np.multiply(kes_N2, salp_N2)

    A1_M1 = np.multiply(m_M1, salp_M1)
    A1_N1 = np.multiply(m_N1, salp_N1)
    A1_M2 = np.multiply(m_M2, salp_M2)
    A1_N2 = np.multiply(m_N2, salp_N2)

    A_M1 = np.divide(A1_M1, A2_M1)
    A_N1 = np.divide(A1_N1, A2_N1)
    A_M2 = np.divide(A1_M2, A2_M2)
    A_N2 = np.divide(A1_N2, A2_N2)

    tmp = 2 * model_parameters.a * model_parameters.b / csqr
    halft = 0.5 * market_parameters.T[market_pointer]

    D_M1 = (
        np.log(d_M1)
        + (model_parameters.a - d_M1) * halft
        - np.log(
            (d_M1 + kes_M1) * 0.5
            + (d_M1 - kes_M1)
            * 0.5
            * np.exp(-d_M1 * market_parameters.T[market_pointer])
        )
    )
    D_M2 = (
        np.log(d_M2)
        + (model_parameters.a - d_M2) * halft
        - np.log(
            (d_M2 + kes_M2) * 0.5
            + (d_M1 - kes_M2)
            * 0.5
            * np.exp(-d_M2 * market_parameters.T[market_pointer])
        )
    )
    D_N1 = (
        np.log(d_N1)
        + (model_parameters.a - d_N1) * halft
        - np.log(
            (d_N1 + kes_N1) * 0.5
            + (d_N1 - kes_N1)
            * 0.5
            * np.exp(-d_N1 * market_parameters.T[market_pointer])
        )
    )
    D_N2 = (
        np.log(d_N2)
        + (model_parameters.a - d_N2) * halft
        - np.log(
            (d_N2 + kes_N2) * 0.5
            + (d_N2 - kes_N2)
            * 0.5
            * np.exp(-d_N2 * market_parameters.T[market_pointer])
        )
    )

    MNbas = TagMn(
        np.real(
            h_M * np.exp(x0 * _imPQ_M - model_parameters.v0 * A_M1 + tmp * D_M1) * g_M1
        ),
        np.real(
            h_N * np.exp(x0 * _imPQ_N - model_parameters.v0 * A_N1 + tmp * D_N1) * g_N1
        ),
        np.real(
            h_M * np.exp(x0 * imPQ_M - model_parameters.v0 * A_M2 + tmp * D_M2) * g_M2
        ),
        np.real(
            h_N * np.exp(x0 * imPQ_N - model_parameters.v0 * A_N2 + tmp * D_N2) * g_N2
        ),
    )
    return MNbas


_tmp_values_fHes = {
    # "num_grids": nb.int32,
    "disc": nb.float64,
    "tmp": nb.float64,
    "y1": nb.float64,
    "y2": nb.float64,
    "l": nb.int32,
    "j": nb.int32,
    "Qv1": nb.float64,
    "Qv2": nb.float64,
    "pv": nb.float64,
}


@nb.njit(locals=_tmp_values_fHes)
def fHes(
    model_parameters: ModelParameters, market_parameters: MarketParameters
) -> np.array:
    """
    Function to calculate price of option by Heston model.
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :return:
    """
    n = len(market_parameters.K)
    x = np.zeros(n, dtype=np.float64)
    for l in range(n):
        K = market_parameters.K[l]
        T = market_parameters.T[l]
        disc = np.exp(-market_parameters.r * T)
        tmp = 0.5 * (market_parameters.S - K * disc)
        # tmp = 0.5 * (market_parameters.S - K) * disc
        disc = disc / pi
        y1, y2 = nb.float64(0.0), nb.float64(0.0)

        MN: TagMn = HesIntMN(
            model_parameters=model_parameters,
            market_parameters=market_parameters,
            market_pointer=np.int32(l),
        )
        y1 = y1 + np.multiply(w64, (MN.M1 + MN.N1)).sum()
        y2 = y2 + np.multiply(w64, (MN.M2 + MN.N2)).sum()
        Qv1, Qv2 = np.float64(0.0), nb.float64(0.0)
        Qv1 = Q * y1
        Qv2 = Q * y2
        pv = np.float64(0.0)
        delta = 0.5 + Qv1 / pi
        # print(delta)

        if market_parameters.types[l]:
            # calls
            # p1 = market_parameters.S*(0.5 + Qv1/pi)
            # p2 = K*np.exp(-market_parameters.r * T)*(0.5 + Qv2/pi)
            # orig = p1 - p2
            pv = disc * (Qv1 - K * Qv2) + tmp
            # print(pv, orig)
        else:
            # puts
            # p1 = market_parameters.S*(- 0.5 + Qv1/pi)
            # p2 = K*np.exp(-market_parameters.r * T)*(- 0.5 + Qv2/pi)
            # orig = p1 - p2
            pv = disc * (Qv1 - K * Qv2) - tmp
        x[l] = pv
    return x


@nb.njit(_signature_HesIntJac, locals=_tmp_values_HesIntJac)
def HesIntJac(
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    market_pointer: int,
) -> tagMNJac:
    """
    Function to calculate integrands (real-valued) for Jacobian.
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :return:
    """
    PQ_M, PQ_N = P + Q * u64, P - Q * u64
    imPQ_M = i * PQ_M
    imPQ_N = i * PQ_N
    _imPQ_M = i * (PQ_M - i)
    _imPQ_N = i * (PQ_N - i)

    h_M = np.divide(np.power(market_parameters.K[market_pointer], -imPQ_M), imPQ_M)
    h_N = np.divide(np.power(market_parameters.K[market_pointer], -imPQ_N), imPQ_N)

    x0 = (
        np.log(market_parameters.S)
        + market_parameters.r * market_parameters.T[market_pointer]
    )
    tmp = model_parameters.c * model_parameters.rho
    kes_M1 = model_parameters.a - np.multiply(tmp, _imPQ_M)
    kes_N1 = model_parameters.a - np.multiply(tmp, _imPQ_N)
    kes_M2 = kes_M1 + tmp
    kes_N2 = kes_N1 + tmp

    _msqr = np.power(PQ_M - i, 2)
    _nsqr = np.power(PQ_N - i, 2)
    msqr = np.power(PQ_M - zero * i, 2)
    nsqr = np.power(PQ_N - zero * i, 2)

    m_M1 = imPQ_M + one + _msqr
    m_N1 = imPQ_N + one + _nsqr
    m_M2 = imPQ_M + msqr
    m_N2 = imPQ_N + nsqr

    csqr = np.power(model_parameters.c, 2)
    d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
    d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
    d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
    d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

    abrt = (
        model_parameters.a
        * model_parameters.b
        * model_parameters.rho
        * market_parameters.T[market_pointer]
    )
    tmp1 = -abrt / model_parameters.c
    tmp2 = np.exp(tmp1)

    g_M2 = np.exp(tmp1 * imPQ_M)
    g_N2 = np.exp(tmp1 * imPQ_N)
    g_M1 = g_M2 * tmp2
    g_N1 = g_N2 * tmp2

    halft = 0.5 * market_parameters.T[market_pointer]
    alpha = d_M1 * halft
    calp_M1 = np.cosh(alpha)
    salp_M1 = np.sinh(alpha)

    alpha = d_N1 * halft
    calp_N1 = np.cosh(alpha)
    salp_N1 = np.sinh(alpha)

    alpha = d_M2 * halft
    calp_M2 = np.cosh(alpha)
    salp_M2 = np.sinh(alpha)

    alpha = d_N2 * halft
    calp_N2 = np.cosh(alpha)
    salp_N2 = np.sinh(alpha)

    # A2 = d*calp + kes*salp;
    A2_M1 = d_M1 * calp_M1 + kes_M1 * salp_M1
    A2_N1 = d_N1 * calp_N1 + kes_N1 * salp_N1
    A2_M2 = d_M2 * calp_M2 + kes_M2 * salp_M2
    A2_N2 = d_N2 * calp_N2 + kes_N2 * salp_N2

    # A1 = m*salp;
    A1_M1 = m_M1 * salp_M1
    A1_N1 = m_N1 * salp_N1
    A1_M2 = m_M2 * salp_M2
    A1_N2 = m_N2 * salp_N2

    # A = A1/A2;
    A_M1 = A1_M1 / A2_M1
    A_N1 = A1_N1 / A2_N1
    A_M2 = A1_M2 / A2_M2
    A_N2 = A1_N2 / A2_N2

    # B = d*exp(a*T/2)/A2;
    tmp = np.exp(model_parameters.a * halft)
    # exp(a*T/2)
    B_M1 = d_M1 * tmp / A2_M1
    B_N1 = d_N1 * tmp / A2_N1
    B_M2 = d_M2 * tmp / A2_M2
    B_N2 = d_N2 * tmp / A2_N2

    # characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
    tmp3 = 2 * model_parameters.a * model_parameters.b / csqr
    D_M1 = (
        np.log(d_M1)
        + (model_parameters.a - d_M1) * halft
        - np.log(
            (d_M1 + kes_M1) * 0.5
            + (d_M1 - kes_M1)
            * 0.5
            * np.exp(-d_M1 * market_parameters.T[market_pointer])
        )
    )
    D_M2 = (
        np.log(d_M2)
        + (model_parameters.a - d_M2) * halft
        - np.log(
            (d_M2 + kes_M2) * 0.5
            + (d_M1 - kes_M2)
            * 0.5
            * np.exp(-d_M2 * market_parameters.T[market_pointer])
        )
    )
    D_N1 = (
        np.log(d_N1)
        + (model_parameters.a - d_N1) * halft
        - np.log(
            (d_N1 + kes_N1) * 0.5
            + (d_N1 - kes_N1)
            * 0.5
            * np.exp(-d_N1 * market_parameters.T[market_pointer])
        )
    )
    D_N2 = (
        np.log(d_N2)
        + (model_parameters.a - d_N2) * halft
        - np.log(
            (d_N2 + kes_N2) * 0.5
            + (d_N2 - kes_N2)
            * 0.5
            * np.exp(-d_N2 * market_parameters.T[market_pointer])
        )
    )

    y1M1 = np.exp(x0 * _imPQ_M - model_parameters.v0 * A_M1 + tmp3 * D_M1) * g_M1
    y1N1 = np.exp(x0 * _imPQ_N - model_parameters.v0 * A_N1 + tmp3 * D_N1) * g_N1
    y1M2 = np.exp(x0 * imPQ_M - model_parameters.v0 * A_M2 + tmp3 * D_M2) * g_M2
    y1N2 = np.exp(x0 * imPQ_N - model_parameters.v0 * A_N2 + tmp3 * D_N2) * g_N2

    # H = kes*calp + d*salp;
    H_M1 = kes_M1 * calp_M1 + d_M1 * salp_M1
    H_M2 = kes_M2 * calp_M2 + d_M2 * salp_M2
    H_N1 = kes_N1 * calp_N1 + d_N1 * salp_N1
    H_N2 = kes_N2 * calp_N2 + d_N2 * salp_N2

    # lnB = log(B);
    lnB_M1, lnB_M2, lnB_N1, lnB_N2 = D_M1, D_M2, D_N1, D_N2

    # partial b: y3 = y1*(2*a*lnB/pow(c,2)-a*rho*T*u1*i/c);
    tmp4 = tmp3 / model_parameters.b
    tmp5 = tmp1 / model_parameters.b

    y3M1 = tmp4 * lnB_M1 + tmp5 * _imPQ_M
    y3M2 = tmp4 * lnB_M2 + tmp5 * imPQ_M
    y3N1 = tmp4 * lnB_N1 + tmp5 * _imPQ_N
    y3N2 = tmp4 * lnB_N2 + tmp5 * imPQ_N

    # partial rho:
    tmp1 = tmp1 / model_parameters.rho  # //-a*b*T/c;

    # for M1
    ctmp = model_parameters.c * _imPQ_M / d_M1
    pd_prho_M1 = -kes_M1 * ctmp
    pA1_prho_M1 = m_M1 * calp_M1 * halft * pd_prho_M1
    pA2_prho_M1 = -ctmp * H_M1 * (one + kes_M1 * halft)
    pA_prho_M1 = (pA1_prho_M1 - A_M1 * pA2_prho_M1) / A2_M1
    ctmp = pd_prho_M1 - pA2_prho_M1 * d_M1 / A2_M1
    pB_prho_M1 = tmp / A2_M1 * ctmp
    y4M1 = -model_parameters.v0 * pA_prho_M1 + tmp3 * ctmp / d_M1 + tmp1 * _imPQ_M

    # for N1
    ctmp = model_parameters.c * _imPQ_N / d_N1
    pd_prho_N1 = -kes_N1 * ctmp
    pA1_prho_N1 = m_N1 * calp_N1 * halft * pd_prho_N1
    pA2_prho_N1 = -ctmp * H_N1 * (one + kes_N1 * halft)
    pA_prho_N1 = (pA1_prho_N1 - A_N1 * pA2_prho_N1) / A2_N1
    ctmp = pd_prho_N1 - pA2_prho_N1 * d_N1 / A2_N1
    pB_prho_N1 = tmp / A2_N1 * ctmp
    y4N1 = -model_parameters.v0 * pA_prho_N1 + tmp3 * ctmp / d_N1 + tmp1 * _imPQ_N

    # for M2
    ctmp = model_parameters.c * imPQ_M / d_M2
    pd_prho_M2 = -kes_M2 * ctmp
    pA1_prho_M2 = m_M2 * calp_M2 * halft * pd_prho_M2
    pA2_prho_M2 = -ctmp * H_M2 * (one + kes_M2 * halft) / d_M2
    pA_prho_M2 = (pA1_prho_M2 - A_M2 * pA2_prho_M2) / A2_M2
    ctmp = pd_prho_M2 - pA2_prho_M2 * d_M2 / A2_M2
    pB_prho_M2 = tmp / A2_M2 * ctmp
    y4M2 = -model_parameters.v0 * pA_prho_M2 + tmp3 * ctmp / d_M2 + tmp1 * imPQ_M

    # for N2
    ctmp = model_parameters.c * imPQ_N / d_N2
    pd_prho_N2 = -kes_N2 * ctmp
    pA1_prho_N2 = m_N2 * calp_N2 * halft * pd_prho_N2
    pA2_prho_N2 = -ctmp * H_N2 * (one + kes_N2 * halft)
    pA_prho_N2 = (pA1_prho_N2 - A_N2 * pA2_prho_N2) / A2_N2
    ctmp = pd_prho_N2 - pA2_prho_N2 * d_N2 / A2_N2
    pB_prho_N2 = tmp / A2_N2 * ctmp
    y4N2 = -model_parameters.v0 * pA_prho_N2 + tmp3 * ctmp / d_N2 + tmp1 * imPQ_N

    # partial a:
    tmp1 = (
        model_parameters.b
        * model_parameters.rho
        * market_parameters.T[market_pointer]
        / model_parameters.c
    )
    tmp2 = tmp3 / model_parameters.a  # 2*b/csqr;
    ctmp = -one / (model_parameters.c * _imPQ_M)

    pB_pa = ctmp * pB_prho_M1 + B_M1 * halft
    y5M1 = (
        -model_parameters.v0 * pA_prho_M1 * ctmp
        + tmp2 * lnB_M1
        + model_parameters.a * tmp2 * pB_pa / B_M1
        - tmp1 * _imPQ_M
    )

    ctmp = -one / (model_parameters.c * imPQ_M)
    pB_pa = ctmp * pB_prho_M2 + B_M2 * halft
    y5M2 = (
        -model_parameters.v0 * pA_prho_M2 * ctmp
        + tmp2 * lnB_M2
        + model_parameters.a * tmp2 * pB_pa / B_M2
        - tmp1 * imPQ_M
    )

    ctmp = -one / (model_parameters.c * _imPQ_N)
    pB_pa = ctmp * pB_prho_N1 + B_N1 * halft
    y5N1 = (
        -model_parameters.v0 * pA_prho_N1 * ctmp
        + tmp2 * lnB_N1
        + model_parameters.a * tmp2 * pB_pa / B_N1
        - tmp1 * _imPQ_N
    )
    # NOTE: here is a ZeroDivisionError if wrong P, Q
    ctmp = -one / (model_parameters.c * imPQ_N)
    pB_pa = ctmp * pB_prho_N2 + B_N2 * halft

    y5N2 = (
        -model_parameters.v0 * pA_prho_N2 * ctmp
        + tmp2 * lnB_N2
        + model_parameters.a * tmp2 * pB_pa / B_N2
        - tmp1 * imPQ_N
    )

    # partial c:
    tmp = model_parameters.rho / model_parameters.c
    tmp1 = 4 * model_parameters.a * model_parameters.b / np.power(model_parameters.c, 3)
    tmp2 = abrt / csqr
    # M1
    pd_pc = (tmp - one / kes_M1) * pd_prho_M1 + model_parameters.c * _msqr / d_M1
    pA1_pc = m_M1 * calp_M1 * halft * pd_pc
    pA2_pc = (
        tmp * pA2_prho_M1
        - one
        / _imPQ_M
        * (two / (market_parameters.T[market_pointer] * kes_M1) + one)
        * pA1_prho_M1
        + model_parameters.c * halft * A1_M1
    )
    pA_pc = pA1_pc / A2_M1 - A_M1 / A2_M1 * pA2_pc

    y6M1 = (
        -model_parameters.v0 * pA_pc
        - tmp1 * lnB_M1
        + tmp3 / d_M1 * (pd_pc - d_M1 / A2_M1 * pA2_pc)
        + tmp2 * _imPQ_M
    )

    # M2
    pd_pc = (tmp - one / kes_M2) * pd_prho_M2 + model_parameters.c * msqr / d_M2
    pA1_pc = m_M2 * calp_M2 * halft * pd_pc
    pA2_pc = (
        tmp * pA2_prho_M2
        - one
        / imPQ_M
        * (two / (market_parameters.T[market_pointer] * kes_M2) + one)
        * pA1_prho_M2
        + model_parameters.c * halft * A1_M2
    )
    pA_pc = pA1_pc / A2_M2 - A_M2 / A2_M2 * pA2_pc
    y6M2 = (
        -model_parameters.v0 * pA_pc
        - tmp1 * lnB_M2
        + tmp3 / d_M2 * (pd_pc - d_M2 / A2_M2 * pA2_pc)
        + tmp2 * imPQ_M
    )

    # N1
    pd_pc = (tmp - one / kes_N1) * pd_prho_N1 + model_parameters.c * _nsqr / d_N1
    pA1_pc = m_N1 * calp_N1 * halft * pd_pc
    pA2_pc = (
        tmp * pA2_prho_N1
        - one
        / (_imPQ_N)
        * (two / (market_parameters.T[market_pointer] * kes_N1) + one)
        * pA1_prho_N1
        + model_parameters.c * halft * A1_N1
    )
    pA_pc = pA1_pc / A2_N1 - A_N1 / A2_N1 * pA2_pc
    y6N1 = (
        -model_parameters.v0 * pA_pc
        - tmp1 * lnB_N1
        + tmp3 / d_N1 * (pd_pc - d_N1 / A2_N1 * pA2_pc)
        + tmp2 * _imPQ_N
    )

    # N2
    pd_pc = (tmp - one / kes_N2) * pd_prho_N2 + model_parameters.c * nsqr / d_N2
    pA1_pc = m_N2 * calp_N2 * halft * pd_pc
    pA2_pc = (
        tmp * pA2_prho_N2
        - one
        / (imPQ_N)
        * (two / (market_parameters.T[market_pointer] * kes_N2) + one)
        * pA1_prho_N2
        + model_parameters.c * halft * A1_N2
    )
    pA_pc = pA1_pc / A2_N2 - A_N2 / A2_N2 * pA2_pc
    y6N2 = (
        -model_parameters.v0 * pA_pc
        - tmp1 * lnB_N2
        + tmp3 / d_N2 * (pd_pc - d_N2 / A2_N2 * pA2_pc)
        + tmp2 * imPQ_N
    )

    hM1 = h_M * y1M1
    hM2 = h_M * y1M2
    hN1 = h_N * y1N1
    hN2 = h_N * y1N2

    Jacobian = tagMNJac(
        np.real(hM1 * y5M1 + hN1 * y5N1),
        np.real(hM2 * y5M2 + hN2 * y5N2),
        np.real(hM1 * y3M1 + hN1 * y3N1),
        np.real(hM2 * y3M2 + hN2 * y3N2),
        np.real(hM1 * y6M1 + hN1 * y6N1),
        np.real(hM2 * y6M2 + hN2 * y6N2),
        np.real(hM1 * y4M1 + hN1 * y4N1),
        np.real(hM2 * y4M2 + hN2 * y4N2),
        np.real(-hM1 * A_M1 - hN1 * A_N1),
        np.real(-hM2 * A_M2 - hN2 * A_N2),
    )

    return Jacobian


_tmp_values_JacHes = {
    "n": nb.int32,
    "r": nb.float64,
    "discpi": nb.float64,
    "da": nb.float64,
    "db": nb.float64,
    "dc": nb.float64,
    "drho": nb.float64,
    "dv0": nb.float64,
    "jacs": nb.types.Array(nb.float64, 2, "A"),
    "K": nb.float64,
    "T": nb.float64,
    "pa1": nb.float64,
    "pa2": nb.float64,
    "pb1": nb.float64,
    "pb2": nb.float64,
    "pc1": nb.float64,
    "pc2": nb.float64,
    "prho1": nb.float64,
    "prho2": nb.float64,
    "pv01": nb.float64,
    "pv02": nb.float64,
    # "jacint": tagMNJac,
    "Qv1": nb.float64,
    "Qv2": nb.float64,
}


@nb.njit(locals=_tmp_values_JacHes)
def JacHes(
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
) -> np.array:
    """
    Jacobian
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :return:
    """
    n = np.int32(len(market_parameters.K))
    r = market_parameters.r

    da, db, dc, drho, dv0 = (
        np.float64(0.0),
        np.float64(0.0),
        np.float64(0.0),
        np.float64(0.0),
        np.float64(0.0),
    )
    jacs = np.zeros((5, n), dtype=np.float64)
    for l in range(n):
        K = market_parameters.K[l]
        T = market_parameters.T[l]
        discpi = np.exp(-r * T) / pi
        pa1, pa2, pb1, pb2, pc1, pc2, prho1, prho2, pv01, pv02 = (
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
        )
        jacint: tagMNJac = HesIntJac(
            model_parameters=model_parameters,
            market_parameters=market_parameters,
            market_pointer=l,
        )
        pa1 += np.multiply(w64, jacint.pa1s).sum()
        pa2 += np.multiply(w64, jacint.pa2s).sum()

        pb1 += np.multiply(w64, jacint.pb1s).sum()
        pb2 += np.multiply(w64, jacint.pb2s).sum()

        pc1 += np.multiply(w64, jacint.pc1s).sum()
        pc2 += np.multiply(w64, jacint.pc2s).sum()

        prho1 += np.multiply(w64, jacint.prho1s).sum()
        prho2 += np.multiply(w64, jacint.prho2s).sum()

        pv01 += np.multiply(w64, jacint.pv01s).sum()
        pv02 += np.multiply(w64, jacint.pv02s).sum()

        Qv1 = Q * pa1
        Qv2 = Q * pa2
        da = discpi * (Qv1 - K * Qv2)
        jacs[0][l] = da

        Qv1 = Q * pb1
        Qv2 = Q * pb2
        db = discpi * (Qv1 - K * Qv2)
        jacs[1][l] = db

        Qv1 = Q * pc1
        Qv2 = Q * pc2
        dc = discpi * (Qv1 - K * Qv2)
        jacs[2][l] = dc

        Qv1 = Q * prho1
        Qv2 = Q * prho2
        drho = discpi * (Qv1 - K * Qv2)
        jacs[3][l] = drho

        Qv1 = Q * pv01
        Qv2 = Q * pv02
        dv0 = discpi * (Qv1 - K * Qv2)
        jacs[4][l] = dv0

    return jacs


# Newton-Raphsen
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


def get_nu0(df: pd.DataFrame):
    tick = df.copy()
    # get closrst expiration
    tick = tick[tick["expiration"] == tick["expiration"].min()]
    tick["dist"] = abs(tick["strike_price"] - tick["underlying_price"])
    # get closest call
    closest_call = tick[tick["type"] == "call"]
    closest_call = closest_call[
        closest_call["dist"] == closest_call["dist"].min()
    ].iloc[0]
    # get closest put
    closest_put = tick[tick["type"] == "put"]
    closest_put = closest_put[closest_put["dist"] == closest_put["dist"].min()].iloc[0]

    call_iv = get_implied_volatility(
        option_type=closest_call["type"],
        C=closest_call["mark_price_usd"],
        K=closest_call["strike_price"],
        T=closest_call["tau"],
        F=closest_call["underlying_price"],
        r=0.0,
        error=0.001,
    )

    put_iv = get_implied_volatility(
        option_type=closest_put["type"],
        C=closest_put["mark_price_usd"],
        K=closest_put["strike_price"],
        T=closest_put["tau"],
        F=closest_put["underlying_price"],
        r=0.0,
        error=0.001,
    )

    forward = (closest_put["underlying_price"] + closest_call["underlying_price"]) / 2
    strike_diff = abs(closest_call["strike_price"] - closest_put["strike_price"])
    iv = (
        call_iv * (closest_call["strike_price"] - forward) / strike_diff
        + put_iv * (forward - closest_put["strike_price"]) / strike_diff
    )
    nu0 = iv**2
    return nu0


def get_alpha_bar(df: pd.DataFrame, timestamp: int = None):
    if timestamp:
        data = df.query(f"timestamp<={timestamp}").copy()
    else:
        data = df.copy()
    forward = (
        data[["dt", "timestamp", "underlying_price"]]
        .drop_duplicates()
        .sort_values("timestamp")
    )
    # need daily
    forward["date"] = pd.to_datetime(forward["dt"]).dt.date
    forward = forward.loc[forward.groupby("date").dt.idxmax()]

    forward["underlying_price_prev"] = forward["underlying_price"].shift(1)
    forward["timestamp_prev"] = forward["timestamp"].shift(1)
    forward["residual"] = (
        forward["underlying_price"] - forward["underlying_price_prev"]
    ) / (
        forward["underlying_price_prev"]
        * (
            np.sqrt(
                (forward["timestamp"] - forward["timestamp_prev"])
                / 1e6
                / 3600
                / 24
                / 365
            )
        )
    )

    alpha_bar = (
        forward["residual"] - forward["residual"].sum() / (forward.shape[0] + 1)
    ).sum() ** 2 / forward.shape[0]
    return alpha_bar


def get_kappa(df, timestamp):
    if timestamp:
        data = df.query(f"timestamp<={timestamp}").copy()
    else:
        data = df.copy()

    # need it for correct regression
    alpha_bar_tmp = get_alpha_bar(df=data, timestamp=timestamp)

    daily = (
        data[["dt", "timestamp", "underlying_price"]]
        .drop_duplicates()
        .sort_values("timestamp")
        .copy()
    )
    daily["date"] = pd.to_datetime(daily["dt"]).dt.date
    daily = daily.loc[daily.groupby("date").dt.idxmax()]
    # need it weekly
    daily["rolling_variance"] = (daily["underlying_price"].rolling(window=7).std()) ** 2

    daily["rolling_variance_next"] = daily["rolling_variance"].shift(-1)
    daily["timestamp_next"] = daily["timestamp"].shift(-1)
    daily["alpha"] = (
        daily["rolling_variance_next"] - daily["rolling_variance"]
    ) / np.sqrt(daily["rolling_variance"])
    daily["k_coef"] = (
        alpha_bar_tmp
        * ((daily["timestamp_next"] - daily["timestamp"]) / 1e6 / 3600 / 24 / 365)
        - daily["rolling_variance"]
    ) / np.sqrt(daily["rolling_variance"])

    daily["sigma_coef"] = np.sqrt(
        (daily["timestamp_next"] - daily["timestamp"]) / 1e6 / 3600 / 24 / 365
    )  # *np.random.normal(0.0, 1.0, size=len(daily))
    daily = daily[~daily["alpha"].isna()]

    lr = LinearRegression()
    X = daily[["k_coef", "sigma_coef"]].values
    y = daily["alpha"].values
    lr.fit(X, y)

    kappa = lr.coef_[0]
    return kappa


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
    # print(data_grouped)
    return data_grouped


def calibrate_heston(
    df: pd.DataFrame,
    start_params: np.array,
    timestamp: int = None,
    calibration_type: str = "all",
) -> Tuple[np.ndarray, float]:
    """
    Function to calibrate Heston model.
    Attributes:
        @param df (pd.DataFrame): Dataframe with history
        [
            timestamp(ns),
            type(put or call),
            strike_price(usd),
            expiration(ns),
            mark_price(etc/btc),
            underlying_price(usd)
        ]
        @param start_params (np.array): Params to start calibration via LM from
        @param timestamp (int): On which timestamp to calibrate the model.
            Should be in range of df timestamps.
        @param calibration_type(str): Type of calibration. Should be one of: ["all", "nu0"]

    Return:
        calibrated_params (np.array): Array of optimal params on timestamp tick.
        error (float): Value of error on calibration.
    """

    available_calibration_types = ["all", "nu0", "nu0_and_nu_bar", "nu0_and_k", "kappa"]
    if calibration_type not in available_calibration_types:
        raise ValueError(
            f"calibration_type should be from {available_calibration_types}"
        )
    # get market params on this tick
    tick = get_tick(df=df, timestamp=timestamp)
    karr = tick.strike_price.to_numpy(dtype=np.float64)
    carr = tick.mark_price_usd.to_numpy(dtype=np.float64)
    tarr = tick.tau.to_numpy(dtype=np.float64)
    types = np.where(tick["type"] == "call", True, False)
    # take it zero as on deribit
    r_val = np.float64(0.0)
    # tick dataframes may have not similar timestamps -->
    # not equal value if underlying --> take mean
    S_val = np.float64(tick.underlying_price.mean())
    market = MarketParameters(K=karr, T=tarr, S=S_val, r=r_val, C=carr, types=types)

    def clip_params(heston_params: np.ndarray) -> np.ndarray:
        """
        This funciton project heston parameters into valid range
        Attributes:
            heston_params(np.ndarray): model parameters
        Returns:
            heston_params(np.ndarray): clipped parameters
        """
        eps = 1e-4

        def clip_all(params):
            heston_params = params
            for i in range(len(heston_params) // 5):
                a, b, c, rho, v0 = heston_params[i * 5 : i * 5 + 5]
                a = np.clip(a, eps, 500.0)
                b = np.clip(b, eps, 500.0)
                c = np.clip(c, eps, 150.0)
                rho = np.clip(rho, -1.0 + eps, 1.0 - eps)
                v0 = np.clip(v0, eps, 10.0)
                heston_params[i * 5 : i * 5 + 5] = a, b, c, rho, v0
            return heston_params

        if calibration_type == "all":
            heston_params = clip_all(heston_params)

        if calibration_type == "nu0":
            heston_params = np.concatenate([heston_params, np.array([nu0])])
            heston_params = clip_all(heston_params)[:-1]

        elif calibration_type == "nu0_and_nu_bar":
            heston_params = np.concatenate(
                [
                    heston_params[0:1],
                    np.array([nu_bar]),
                    heston_params[1:3],
                    np.array([nu0]),
                ]
            )
            heston_params = clip_all(heston_params)
            heston_params = np.concatenate([heston_params[0:1], heston_params[2:-1]])

        elif calibration_type == "nu0_and_k":
            heston_params = np.concatenate(
                [np.array([kappa]), heston_params, np.array([nu0])]
            )
            heston_params = clip_all(heston_params)[1:-1]

        elif calibration_type == "kappa":
            heston_params = np.concatenate([np.array([kappa]), heston_params])
            heston_params = clip_all(heston_params)[1:]

        return heston_params

    def get_residuals(heston_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function calculates residuals of market prices and calibrated ones
        and Jacobian matrix
        Args:
            heston_params(np.ndarray): model params
        Returns:
            res(np.ndarray) : vector or residuals
            J(np.ndarray)   : Jacobian
        """
        # Get the needed format for calibration
        if calibration_type == "all":
            model_parameters = ModelParameters(
                heston_params[0],
                heston_params[1],
                heston_params[2],
                heston_params[3],
                heston_params[4],
            )
            J = JacHes(model_parameters=model_parameters, market_parameters=market)

        if calibration_type == "nu0":
            model_parameters = ModelParameters(
                heston_params[0],
                heston_params[1],
                heston_params[2],
                heston_params[3],
                nu0,
            )
            J = JacHes(model_parameters=model_parameters, market_parameters=market)[:-1]

        elif calibration_type == "nu0_and_nu_bar":
            model_parameters = ModelParameters(
                heston_params[0],
                nu_bar,
                heston_params[1],
                heston_params[2],
                nu0,
            )

            J_tmp = JacHes(model_parameters=model_parameters, market_parameters=market)
            J = np.concatenate([J_tmp[0:1], J_tmp[2:-1]])

        elif calibration_type == "nu0_and_k":
            model_parameters = ModelParameters(
                kappa,
                heston_params[0],
                heston_params[1],
                heston_params[2],
                nu0,
            )
            J = JacHes(model_parameters=model_parameters, market_parameters=market)[
                1:-1
            ]

        elif calibration_type == "kappa":
            model_parameters = ModelParameters(
                kappa,
                heston_params[0],
                heston_params[1],
                heston_params[2],
                heston_params[3],
            )
            J = JacHes(model_parameters=model_parameters, market_parameters=market)[1:]

        # count prices for each option
        C = fHes(
            model_parameters=model_parameters,
            market_parameters=market,
        )
        weights = np.ones_like(market.K)
        weights = weights / np.sum(weights)
        res = C - market.C
        return res * weights, J @ np.diag(weights)

    # function supports several calibration types
    if calibration_type == "all":
        res = LevenbergMarquardt(200, get_residuals, clip_params, start_params)
        calibrated_params = np.array(res["x"], dtype=np.float64)

    elif calibration_type == "nu0":
        # finding nu0
        nu0 = get_nu0(tick)
        # all params exluding nu0 to LM
        res = LevenbergMarquardt(200, get_residuals, clip_params, start_params[:-1])
        calibrated_params = np.array(res["x"], dtype=np.float64)
        calibrated_params = np.concatenate([calibrated_params, np.array([nu0])])

    elif calibration_type == "nu0_and_nu_bar":
        # finding nu0
        nu0 = get_nu0(tick)
        # finding nu0_bar
        nu_bar = get_alpha_bar(df=df, timestamp=timestamp)
        # nu_bar = get_alpha_bar(df=df)
        res = LevenbergMarquardt(
            200,
            get_residuals,
            clip_params,
            np.concatenate([start_params[0:1], start_params[2:-1]]),
        )
        calibrated_params = np.array(res["x"], dtype=np.float64)
        calibrated_params = np.concatenate(
            [
                calibrated_params[0:1],
                np.array([nu_bar]),
                calibrated_params[1:3],
                np.array([nu0]),
            ]
        )

    elif calibration_type == "nu0_and_k":
        nu0 = get_nu0(tick)
        kappa = get_kappa(df=df, timestamp=timestamp)
        res = LevenbergMarquardt(200, get_residuals, clip_params, start_params[1:-1])
        calibrated_params = np.array(res["x"], dtype=np.float64)
        calibrated_params = np.concatenate(
            [np.array([kappa]), calibrated_params, np.array([nu0])]
        )

    elif calibration_type == "kappa":
        kappa = get_kappa(df=df, timestamp=timestamp)
        res = LevenbergMarquardt(200, get_residuals, clip_params, start_params[1:])
        calibrated_params = np.array(res["x"], dtype=np.float64)
        calibrated_params = np.concatenate([np.array([kappa]), calibrated_params])

    error = res["objective"][-1]

    # decomm if you want to see colebrated prices
    final_params = ModelParameters(
        calibrated_params[0],
        calibrated_params[1],
        calibrated_params[2],
        calibrated_params[3],
        calibrated_params[4],
    )
    final_prices = fHes(
        model_parameters=final_params,
        market_parameters=market,
    )

    tick["calibrated_mark_price_usd"] = final_prices
    market_ivs, calibrated_ivs = [], []

    for index, t in tick.iterrows():
        market_iv = get_implied_volatility(
            option_type=t["type"],
            C=t["mark_price_usd"],
            K=t["strike_price"],
            T=t["tau"],
            F=t["underlying_price"],
            r=0.0,
            error=0.001,
        )
        market_ivs.append(market_iv)

        calibrated_iv = get_implied_volatility(
            option_type=t["type"],
            C=t["calibrated_mark_price_usd"],
            K=t["strike_price"],
            T=t["tau"],
            F=t["underlying_price"],
            r=0.0,
            error=0.001,
        )
        calibrated_ivs.append(calibrated_iv)
    tick["market_iv"] = market_ivs
    tick["market_iv"] = tick["market_iv"] * 100

    tick["calibrated_iv"] = calibrated_ivs
    tick["calibrated_iv"] = tick["calibrated_iv"] * 100

    result = tick[
        [
            "type",
            "strike_price",
            "expiration",
            "underlying_price",
            "mark_price_usd",
            "calibrated_mark_price_usd",
            "market_iv",
            "calibrated_iv",
        ]
    ].copy()

    return calibrated_params, error, result

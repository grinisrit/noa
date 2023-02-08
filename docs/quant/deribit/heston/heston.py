import numba as nb
import numpy as np
from typing import Final

# global GLAW_DIM
# GLAW_DIM = 64 // 2

_spec_market_params = [
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64[:]),
    ("K", nb.float64[:]),
]
_spec_model_params = [
    ("a", nb.float64),
    ("b", nb.float64),
    ("c", nb.float64),
    ("rho", nb.float64),
    ("v0", nb.float64),
]
_spec_integral_settings = [
    ("numgrid", nb.int32),
    ("u", nb.float64[:]),
    ("w", nb.float64[:]),
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

    def __init__(
        self, S: nb.float64, r: nb.float64, T: nb.float64[:], K: nb.float64[:]
    ):
        self.S = S
        self.r = r
        self.T = T
        self.K = K


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


@nb.experimental.jitclass(_spec_integral_settings)
class GLAW(object):
    numgrid: nb.int32
    u: nb.float64[:]
    w: nb.float64[:]

    def __init__(self, numgrid: nb.int32, u: nb.float64[:], w: nb.float64[:]):
        self.numgrid = numgrid
        self.u = u
        self.w = w


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

# TODO
_tmp_values_JacHes = {}

_signature_JacHes = []

_signature_HesIntMN = TagMn.class_type.instance_type(
    GLAW.class_type.instance_type,
    ModelParameters.class_type.instance_type,
    MarketParameters.class_type.instance_type,
    nb.int32,
)

_signature_HesIntJac = tagMNJac.class_type.instance_type(
    GLAW.class_type.instance_type,
    ModelParameters.class_type.instance_type,
    MarketParameters.class_type.instance_type,
    nb.int32,
)


@nb.njit(_signature_HesIntMN, locals=_tmp_values_HesIntMN)
def HesIntMN(
    glaw: GLAW,
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    market_pointer: int,
):
    """

    :param glaw:
    :param model_parameters:
    :param market_parameters:
    :param market_pointer:
    :return:
    """
    csqr = np.power(model_parameters.c, 2)
    PQ_M, PQ_N = P + Q * glaw.u, P - Q * glaw.u
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
    "num_grids": nb.int32,
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
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    integration_settings: GLAW,
    m: int,
    n: int,
):
    """
    Function to calculate price of option by Heston model.
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :param integration_settings: GLAW class
    :param x: Observation
    :param m: dim_p
    :param n: dim_x
    :return:
    """
    x = np.zeros(n, dtype=np.float64)
    num_grids = integration_settings.numgrid // 2
    for l in range(n):
        K = market_parameters.K[l]
        T = market_parameters.T[l]
        disc = np.exp(-market_parameters.r * T)
        tmp = 0.5 * (market_parameters.S - K * disc)
        disc = disc / pi
        y1, y2 = nb.float64(0.0), nb.float64(0.0)
        MN: TagMn = HesIntMN(
            glaw=integration_settings,
            model_parameters=model_parameters,
            market_parameters=market_parameters,
            market_pointer=np.int32(l),
        )
        y1 = y1 + np.multiply(integration_settings.w, (MN.M1 + MN.N1)).sum()
        y2 = y2 + np.multiply(integration_settings.w, (MN.M2 + MN.N2)).sum()
        Qv1, Qv2 = np.float64(0.0), nb.float64(0.0)
        Qv1 = Q * y1
        Qv2 = Q * y2
        pv = np.float64(0.0)
        pv = tmp + disc * (Qv1 - K * Qv2)
        x[l] = pv
    return x


@nb.njit(_signature_HesIntJac, locals=_tmp_values_HesIntJac)
def HesIntJac(
    glaw: GLAW,
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    market_pointer: int,
) -> tagMNJac:
    """
    Function to calculate integrands (real-valued) for Jacobian.
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :param glaw: GLAW class
    :return:
    """
    PQ_M, PQ_N = P + Q * glaw.u, P - Q * glaw.u
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

# @nb.njit(_signature_JacHes, locals = _tmp_values_JacHes)
@nb.njit
def JacHes(
    glaw: GLAW,
    model_parameters: ModelParameters,
    market_parameters: MarketParameters,
    market_pointer: int,
    n: int,
) -> None:
    """
    Jacobian
    :param model_parameters: ModelParameters class
    :param market_parameters: MarketParameters class
    :param glaw: GLAW class
    :return:
    """
    r = market_parameters.r
    NumGrids = glaw.numgrid
    discpi = np.exp(-r * market_parameters.T) / pi
    jac = np.zeros(n, dtype=np.float64)
    w = glaw.w
    for l in range(n):
        K = market_parameters.K[l]
        T = market_parameters.T[l]
        discpi = np.exp(-r * T) / pi
        pa1, pa2, pb1, pb2, pc1, pc2, prho1, prho2, pv01, pv02 = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        jacint: tagMNJac = HesIntJac(
            glaw=glaw,
            model_parameters=model_parameters,
            market_parameters=market_parameters,
            market_pointer=l,
        )
        pa1 += np.multiply(w, jacint.pa1s).sum()
        pa2 += np.multiply(w, jacint.pa2s).sum()

        pb1 += np.multiply(w, jacint.pb1s).sum()
        pb2 += np.multiply(w, jacint.pb2s).sum()

        pc1 += np.multiply(w, jacint.pc1s).sum()
        pc2 += np.multiply(w, jacint.pc2s).sum()

        prho1 += np.multiply(w, jacint.prho1s).sum()
        prho2 += np.multiply(w, jacint.prho2s).sum()

        pv01 += np.multiply(w, jacint.pv01s).sum()
        pv02 += np.multiply(w, jacint.pv02s).sum()

        Qv1 = Q * pa1
        Qv2 = Q * pa2
        jac[l] = discpi * (Qv1 - K * Qv2)

        Qv1 = Q * pb1
        Qv2 = Q * pb2
        jac[l] = discpi * (Qv1 - K * Qv2)

        Qv1 = Q * pc1
        Qv2 = Q * pc2
        jac[l] = discpi * (Qv1 - K * Qv2)

        Qv1 = Q * prho1
        Qv2 = Q * prho2
        jac[l] = discpi * (Qv1 - K * Qv2)

        Qv1 = Q * pv01
        Qv2 = Q * pv02
        jac[l] = discpi * (Qv1 - K * Qv2)

    return jac


# examples of @molozey, example from original C++ implementation is in main.py

# model = ModelParameters(
#     a=np.float64(100.0),
#     b=np.float64(10.0),
#     c=np.float64(1.0),
#     rho=np.float64(10.0),
#     v0=np.float64(15.0),
# )

# market = MarketParameters(
#     S=np.float64(1.2),
#     r=np.float64(0.1),
#     T=np.array([1.0, 1.0, 1.0], dtype=np.float64),
#     K=np.array([1.0, 2.0, 3.0], dtype=np.float64),
# )

# glaw = GLAW(
#     numgrid=64,
#     u=np.array([1.0, 1.0, 2.0], dtype=np.float64),
#     w=np.array([1.0, 1.0, 2.0], dtype=np.float64),
# )

# back: TagMn = HesIntMN(
#     glaw=glaw,
#     model_parameters=model,
#     market_parameters=market,
#     market_pointer=np.int32(1),
# )
# fHes(
#     model_parameters=model,
#     market_parameters=market,
#     integration_settings=glaw,
#     x=np.array([1.0, 2.0, 3.0], dtype=np.float64),
#     m=10,
#     n=10,
# )


# Jac: tagMNJac = HesIntJac(
#     glaw=glaw,
#     model_parameters=model,
#     market_parameters=market,
#     market_pointer=np.int32(1),
# )
# print(Jac.pa1s)
# print(Jac.pa2s)
# print(Jac.pc1s)
# print(Jac.pc2s)
# print(Jac.pb1s)
# print(Jac.pb2s)
# print(Jac.prho1s)
# print(Jac.prho2s)
# print(Jac.pv01s)
# print(Jac.pv02s)

# print("Worked out")

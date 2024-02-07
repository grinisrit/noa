import numba as nb
from numba.experimental import jitclass
from typing import Final, Tuple

from .common import *


################# Helper classes and variables for HestonCalc #################
# noinspection DuplicatedCode
@nb.experimental.jitclass([
    ("M1", nb.float64[:]),
    ("N1", nb.float64[:]),
    ("M2", nb.float64[:]),
    ("N2", nb.float64[:]),
])
class _TagMn(object):
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


@nb.experimental.jitclass([
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
])
class _TagMNJac(object):
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


_ZERO = np.complex128(complex(0.0, 0.0))
_ONE = np.complex128(complex(1.0, 0.0))
_TWO = np.complex128(complex(2.0, 0.0))
_I = np.complex128(complex(0.0, 1.0))
_PI: Final[np.float64] = np.pi
_LOWER_BOUND: Final[np.float64] = np.float64(0.0)
_UPPER_BOUND: Final[np.int32] = np.int32(200)
_Q: Final[np.float64] = np.float64(0.5 * (_UPPER_BOUND - _LOWER_BOUND))
_P: Final[np.float64] = np.float64(0.5 * (_UPPER_BOUND + _LOWER_BOUND))

# points in which quadratures are computed
_U64 = np.array(
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

# Gaussian quadrature weights from 0 to 1 (because we integrate from zero)
_W64 = np.array(
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

##############################################################################


@nb.experimental.jitclass([
    ("v0", nb.float64)
])
class Variance:
    def __init__(self, v0: nb.float64):
        self.v0 = v0


@nb.experimental.jitclass([
    ("kappa", nb.float64)
])
class VarReversion:
    def __init__(self, kappa: nb.float64):
        self.kappa = kappa


@nb.experimental.jitclass([
    ("theta", nb.float64)
])
class AverageVar:
    def __init__(self, theta: nb.float64):
        self.theta = theta


@nb.experimental.jitclass([
    ("eps", nb.float64)
])
class VolOfVar:
    def __init__(self, eps: nb.float64):
        self.eps = eps


@nb.experimental.jitclass([
    ("rho", nb.float64)
])
class Correlation:
    def __init__(self, rho: nb.float64):
        self.rho = rho


@jitclass([
    ("v0", nb.float64),
    ("kappa", nb.float64),
    ("theta", nb.float64),
    ("eps", nb.float64),
    ("rho", nb.float64)
])
class HestonParams:
    def __init__(
        self,
        variance: Variance,
        var_reversion: VarReversion,
        average_var: AverageVar,
        vol_var: VolOfVar,
        correlation: Correlation
    ):
        self.v0 = variance.v0
        self.kappa = var_reversion.kappa
        self.theta = average_var.theta
        self.eps = vol_var.eps
        self.rho = correlation.rho

    def array(self) -> nb.float64[:]:
        return np.array([self.v0, self.kappa, self.theta, self.eps, self.rho])


@nb.experimental.jitclass([
    ("S", nb.float64),
    ("r", nb.float64),
    ("T", nb.float64[:]),
    ("K", nb.float64[:]),
    ("types", nb.boolean[:]),
    ("premiums", nb.float64[:]),
])
class MarketParams:
    """Describes market state at a single moment of time.
    The field `premiums` is used only for calibration.
    """
    def __init__(
        self,
        spot: Spot,
        forward_yield: ForwardYield,
        tenors: TimesToMaturity,
        strikes: Strikes,
        option_types: OptionTypes,
        premiums: Premiums
    ):
        if not tenors.data.shape == strikes.data.shape == premiums.data.shape == option_types.data.shape:
            raise ValueError('Inconsistent data shape between tenors, strikes, premiums and option types')
        self.S = spot.S
        self.r = forward_yield.r
        self.T = tenors.data
        self.K = strikes.data
        self.types = option_types.data
        self.premiums = premiums.data

    @staticmethod
    def from_forward_curve(
        forward_curve: ForwardCurve,
        tenors: TimesToMaturity,
        strikes: Strikes,
        option_types: OptionTypes,
        premiums: Premiums
    ):
        return MarketParams(
            Spot(forward_curve.S),
            ForwardYield(forward_curve.r[0]),
            tenors,
            strikes,
            option_types,
            premiums
        )


@nb.experimental.jitclass([
    ("w", nb.float64[:])
])
class CalibrationWeights:
    def __init__(self, w: nb.float64):
        if not np.all(w>=0):
            raise ValueError('Weights must be non-negative')
        if not w.sum() > 0:
            raise ValueError('At least one weight must be non-trivial')
        self.w = w


# noinspection DuplicatedCode
@jitclass([
    ("cached_params", nb.float64[:]),
    ("num_iter", nb.int64),
    ("tol", nb.float64),
    ("calibration_error", nb.float64),
    ("r_from_calibration", nb.float64),
])
class HestonCalc:
    def __init__(
        self,
        start_params: HestonParams,
        num_iter: nb.int64 = 50,
        tol: nb.float64 = 1e-8
    ):
        self.cached_params = start_params.array()
        self.num_iter = num_iter
        self.tol = tol
        self.calibration_error = 0.0
        self.r_from_calibration = 0.0

    def calibrate(
        self,
        market_params: MarketParams,
        calibration_weights: CalibrationWeights
    ) -> HestonParams:
        w = calibration_weights.w
        if not w.shape == market_params.premiums.shape:
            raise ValueError('Inconsistent data shape between `calibration_weights` and `market_params`')
        weights = w / w.sum()

        n_points = len(market_params.premiums)
        PARAMS_TO_CALIBRATE: nb.int64 = 5
        if not n_points - PARAMS_TO_CALIBRATE >= 0:
            raise ValueError('Need at least 5 points to calibrate Heston model')

        self.r_from_calibration = market_params.r

        def clip_params(params: np.ndarray) -> np.ndarray:
            small = 1e-4
            v0, kappa, theta, eps, rho = params[0], params[1], params[2], params[3], params[4]
            v0 = np_clip(v0, small, 10.0)
            kappa = np_clip(kappa, small, 500.0)
            theta = np_clip(theta, small, 500.0)
            eps = np_clip(eps, small, 150.0)
            rho = np_clip(rho, -1.0 + small, 1.0 - small)
            clipped_params = np.array([v0, kappa, theta, eps, rho])
            return clipped_params

        def get_residuals(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            heston_params = HestonParams(
                Variance(params[0]),
                VarReversion(params[1]),
                AverageVar(params[2]),
                VolOfVar(params[3]),
                Correlation(params[4])
            )
            premiums = self._heston_vanilla_premium(heston_params, market_params)
            residuals = (premiums - market_params.premiums) * weights
            jacobian = self._jac_hes(heston_params, market_params) @ np.diag(weights)
            return residuals, jacobian

        def levenberg_marquardt(f, proj, x0):
            x = x0.copy()

            mu = 1e-2
            nu1 = 2.0
            nu2 = 2.0

            res, J = f(x)
            F = res.T @ res

            result_x = x
            result_error = F / n_points

            for i in range(self.num_iter):
                if result_error < self.tol:
                    break
                multipl = J @ J.T
                I = np.diag(np.diag(multipl)) + 1e-5 * np.eye(len(x))
                dx = np.linalg.solve(mu * I + multipl, J @ res)
                x_ = proj(x - dx)
                res_, J_ = f(x_)
                F_ = res_.T @ res_
                if F_ < F:
                    x, F, res, J = x_, F_, res_, J_
                    mu /= nu1
                    result_error = F / n_points
                else:
                    i -= 1
                    mu *= nu2
                    continue
                result_x = x

            return result_x, result_error

        self.cached_params, self.calibration_error \
            = levenberg_marquardt(get_residuals, clip_params, self.cached_params)

        return HestonParams(
            Variance(self.cached_params[0]),
            VarReversion(self.cached_params[1]),
            AverageVar(self.cached_params[2]),
            VolOfVar(self.cached_params[3]),
            Correlation(self.cached_params[4])
        )

    def premium(
        self,
        spot: Spot,
        tenor: TimeToMaturity,
        strike: Strike,
        option_type: OptionType,
        forward_yield: nb.optional(ForwardYield) = None
    ) -> Premium:
        """Calculates the premium for one vanilla option.

        If `forward_yield` parameter isn't provided, use the forward yield value
        that Heston model was calibrated with.
        """

        if forward_yield is None:
            forward_yield = ForwardYield(self.r_from_calibration)

        result = self._heston_vanilla_premium(
            HestonParams(
                Variance(self.cached_params[0]),
                VarReversion(self.cached_params[1]),
                AverageVar(self.cached_params[2]),
                VolOfVar(self.cached_params[3]),
                Correlation(self.cached_params[4])
            ),
            MarketParams(
                spot,
                forward_yield,
                TimesToMaturity(np.array([tenor.T])),
                Strikes(np.array([strike.K])),
                OptionTypes(np.array([option_type.is_call])),
                Premiums(np.array([0.0]))  # not used when calculating premium
            )
        )
        return Premium(result.item())

    def premiums(
        self,
        spot: Spot,
        tenors: TimesToMaturity,
        strikes: Strikes,
        option_types: OptionTypes,
        forward_yield: nb.optional(ForwardYield) = None
    ) -> Premiums:
        """Calculates the premiums for vanilla options.

        If `forward_yield` parameter isn't provided, use the forward yield value
        that Heston model was calibrated with.
        """
        if forward_yield is None:
            forward_yield = ForwardYield(self.r_from_calibration)

        result = self._heston_vanilla_premium(
            HestonParams(
                Variance(self.cached_params[0]),
                VarReversion(self.cached_params[1]),
                AverageVar(self.cached_params[2]),
                VolOfVar(self.cached_params[3]),
                Correlation(self.cached_params[4])
            ),
            MarketParams(
                spot,
                forward_yield,
                tenors,
                strikes,
                option_types,
                Premiums(np.zeros(len(option_types.data)))  # not used
            )
        )
        return Premiums(result)

    def _hes_int_jac(
        self,
        heston_params: HestonParams,
        market_params: MarketParams,
        market_pointer: int,
    ) -> _TagMNJac:
        """Calculates real-valued integrands for Jacobian."""
        PQ_M, PQ_N = _P + _Q * _U64, _P - _Q * _U64
        imPQ_M = _I * PQ_M
        imPQ_N = _I * PQ_N
        _imPQ_M = _I * (PQ_M - _I)
        _imPQ_N = _I * (PQ_N - _I)

        h_M = np.divide(np.power(market_params.K[market_pointer], -imPQ_M), imPQ_M)
        h_N = np.divide(np.power(market_params.K[market_pointer], -imPQ_N), imPQ_N)

        x0 = (
            np.log(market_params.S)
            + market_params.r * market_params.T[market_pointer]
        )
        tmp = heston_params.eps * heston_params.rho
        kes_M1 = heston_params.kappa - np.multiply(tmp, _imPQ_M)
        kes_N1 = heston_params.kappa - np.multiply(tmp, _imPQ_N)
        kes_M2 = kes_M1 + tmp
        kes_N2 = kes_N1 + tmp

        _msqr = np.power(PQ_M - _I, 2)
        _nsqr = np.power(PQ_N - _I, 2)
        msqr = np.power(PQ_M - _ZERO * _I, 2)
        nsqr = np.power(PQ_N - _ZERO * _I, 2)

        m_M1 = imPQ_M + _ONE + _msqr
        m_N1 = imPQ_N + _ONE + _nsqr
        m_M2 = imPQ_M + msqr
        m_N2 = imPQ_N + nsqr

        csqr = np.power(heston_params.eps, 2)
        d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
        d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
        d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
        d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

        abrt = (
            heston_params.kappa
            * heston_params.theta
            * heston_params.rho
            * market_params.T[market_pointer]
        )
        tmp1 = -abrt / heston_params.eps
        tmp2 = np.exp(tmp1)

        g_M2 = np.exp(tmp1 * imPQ_M)
        g_N2 = np.exp(tmp1 * imPQ_N)
        g_M1 = g_M2 * tmp2
        g_N1 = g_N2 * tmp2

        halft = 0.5 * market_params.T[market_pointer]
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
        tmp = np.exp(heston_params.kappa * halft)
        # exp(a*T/2)
        B_M1 = d_M1 * tmp / A2_M1
        B_N1 = d_N1 * tmp / A2_N1
        B_M2 = d_M2 * tmp / A2_M2
        B_N2 = d_N2 * tmp / A2_N2

        # characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
        tmp3 = 2 * heston_params.kappa * heston_params.theta / csqr
        D_M1 = (
            np.log(d_M1)
            + (heston_params.kappa - d_M1) * halft
            - np.log(
            (d_M1 + kes_M1) * 0.5
            + (d_M1 - kes_M1)
            * 0.5
            * np.exp(-d_M1 * market_params.T[market_pointer])
        )
        )
        D_M2 = (
            np.log(d_M2)
            + (heston_params.kappa - d_M2) * halft
            - np.log(
            (d_M2 + kes_M2) * 0.5
            + (d_M1 - kes_M2)
            * 0.5
            * np.exp(-d_M2 * market_params.T[market_pointer])
        )
        )
        D_N1 = (
            np.log(d_N1)
            + (heston_params.kappa - d_N1) * halft
            - np.log(
            (d_N1 + kes_N1) * 0.5
            + (d_N1 - kes_N1)
            * 0.5
            * np.exp(-d_N1 * market_params.T[market_pointer])
        )
        )
        D_N2 = (
            np.log(d_N2)
            + (heston_params.kappa - d_N2) * halft
            - np.log(
            (d_N2 + kes_N2) * 0.5
            + (d_N2 - kes_N2)
            * 0.5
            * np.exp(-d_N2 * market_params.T[market_pointer])
        )
        )

        y1M1 = np.exp(x0 * _imPQ_M - heston_params.v0 * A_M1 + tmp3 * D_M1) * g_M1
        y1N1 = np.exp(x0 * _imPQ_N - heston_params.v0 * A_N1 + tmp3 * D_N1) * g_N1
        y1M2 = np.exp(x0 * imPQ_M - heston_params.v0 * A_M2 + tmp3 * D_M2) * g_M2
        y1N2 = np.exp(x0 * imPQ_N - heston_params.v0 * A_N2 + tmp3 * D_N2) * g_N2

        # H = kes*calp + d*salp;
        H_M1 = kes_M1 * calp_M1 + d_M1 * salp_M1
        H_M2 = kes_M2 * calp_M2 + d_M2 * salp_M2
        H_N1 = kes_N1 * calp_N1 + d_N1 * salp_N1
        H_N2 = kes_N2 * calp_N2 + d_N2 * salp_N2

        # lnB = log(B);
        lnB_M1, lnB_M2, lnB_N1, lnB_N2 = D_M1, D_M2, D_N1, D_N2

        # partial b: y3 = y1*(2*a*lnB/pow(c,2)-a*rho*T*u1*i/c);
        tmp4 = tmp3 / heston_params.theta
        tmp5 = tmp1 / heston_params.theta

        y3M1 = tmp4 * lnB_M1 + tmp5 * _imPQ_M
        y3M2 = tmp4 * lnB_M2 + tmp5 * imPQ_M
        y3N1 = tmp4 * lnB_N1 + tmp5 * _imPQ_N
        y3N2 = tmp4 * lnB_N2 + tmp5 * imPQ_N

        # partial rho:
        tmp1 = tmp1 / heston_params.rho  # //-a*b*T/c;

        # for M1
        ctmp = heston_params.eps * _imPQ_M / d_M1
        pd_prho_M1 = -kes_M1 * ctmp
        pA1_prho_M1 = m_M1 * calp_M1 * halft * pd_prho_M1
        pA2_prho_M1 = -ctmp * H_M1 * (_ONE + kes_M1 * halft)
        pA_prho_M1 = (pA1_prho_M1 - A_M1 * pA2_prho_M1) / A2_M1
        ctmp = pd_prho_M1 - pA2_prho_M1 * d_M1 / A2_M1
        pB_prho_M1 = tmp / A2_M1 * ctmp
        y4M1 = -heston_params.v0 * pA_prho_M1 + tmp3 * ctmp / d_M1 + tmp1 * _imPQ_M

        # for N1
        ctmp = heston_params.eps * _imPQ_N / d_N1
        pd_prho_N1 = -kes_N1 * ctmp
        pA1_prho_N1 = m_N1 * calp_N1 * halft * pd_prho_N1
        pA2_prho_N1 = -ctmp * H_N1 * (_ONE + kes_N1 * halft)
        pA_prho_N1 = (pA1_prho_N1 - A_N1 * pA2_prho_N1) / A2_N1
        ctmp = pd_prho_N1 - pA2_prho_N1 * d_N1 / A2_N1
        pB_prho_N1 = tmp / A2_N1 * ctmp
        y4N1 = -heston_params.v0 * pA_prho_N1 + tmp3 * ctmp / d_N1 + tmp1 * _imPQ_N

        # for M2
        ctmp = heston_params.eps * imPQ_M / d_M2
        pd_prho_M2 = -kes_M2 * ctmp
        pA1_prho_M2 = m_M2 * calp_M2 * halft * pd_prho_M2
        pA2_prho_M2 = -ctmp * H_M2 * (_ONE + kes_M2 * halft) / d_M2
        pA_prho_M2 = (pA1_prho_M2 - A_M2 * pA2_prho_M2) / A2_M2
        ctmp = pd_prho_M2 - pA2_prho_M2 * d_M2 / A2_M2
        pB_prho_M2 = tmp / A2_M2 * ctmp
        y4M2 = -heston_params.v0 * pA_prho_M2 + tmp3 * ctmp / d_M2 + tmp1 * imPQ_M

        # for N2
        ctmp = heston_params.eps * imPQ_N / d_N2
        pd_prho_N2 = -kes_N2 * ctmp
        pA1_prho_N2 = m_N2 * calp_N2 * halft * pd_prho_N2
        pA2_prho_N2 = -ctmp * H_N2 * (_ONE + kes_N2 * halft)
        pA_prho_N2 = (pA1_prho_N2 - A_N2 * pA2_prho_N2) / A2_N2
        ctmp = pd_prho_N2 - pA2_prho_N2 * d_N2 / A2_N2
        pB_prho_N2 = tmp / A2_N2 * ctmp
        y4N2 = -heston_params.v0 * pA_prho_N2 + tmp3 * ctmp / d_N2 + tmp1 * imPQ_N

        # partial a:
        tmp1 = (
            heston_params.theta
            * heston_params.rho
            * market_params.T[market_pointer]
            / heston_params.eps
        )
        tmp2 = tmp3 / heston_params.kappa  # 2*b/csqr;
        ctmp = -_ONE / (heston_params.eps * _imPQ_M)

        pB_pa = ctmp * pB_prho_M1 + B_M1 * halft
        y5M1 = (
            -heston_params.v0 * pA_prho_M1 * ctmp
            + tmp2 * lnB_M1
            + heston_params.kappa * tmp2 * pB_pa / B_M1
            - tmp1 * _imPQ_M
        )

        ctmp = -_ONE / (heston_params.eps * imPQ_M)
        pB_pa = ctmp * pB_prho_M2 + B_M2 * halft
        y5M2 = (
            -heston_params.v0 * pA_prho_M2 * ctmp
            + tmp2 * lnB_M2
            + heston_params.kappa * tmp2 * pB_pa / B_M2
            - tmp1 * imPQ_M
        )

        ctmp = -_ONE / (heston_params.eps * _imPQ_N)
        pB_pa = ctmp * pB_prho_N1 + B_N1 * halft
        y5N1 = (
            -heston_params.v0 * pA_prho_N1 * ctmp
            + tmp2 * lnB_N1
            + heston_params.kappa * tmp2 * pB_pa / B_N1
            - tmp1 * _imPQ_N
        )
        # NOTE: here is a ZeroDivisionError if wrong P, Q
        ctmp = -_ONE / (heston_params.eps * imPQ_N)
        pB_pa = ctmp * pB_prho_N2 + B_N2 * halft

        y5N2 = (
            -heston_params.v0 * pA_prho_N2 * ctmp
            + tmp2 * lnB_N2
            + heston_params.kappa * tmp2 * pB_pa / B_N2
            - tmp1 * imPQ_N
        )

        # partial c:
        tmp = heston_params.rho / heston_params.eps
        tmp1 = 4 * heston_params.kappa * heston_params.theta / np.power(heston_params.eps, 3)
        tmp2 = abrt / csqr
        # M1
        pd_pc = (tmp - _ONE / kes_M1) * pd_prho_M1 + heston_params.eps * _msqr / d_M1
        pA1_pc = m_M1 * calp_M1 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_M1
            - _ONE
            / _imPQ_M
            * (_TWO / (market_params.T[market_pointer] * kes_M1) + _ONE)
            * pA1_prho_M1
            + heston_params.eps * halft * A1_M1
        )
        pA_pc = pA1_pc / A2_M1 - A_M1 / A2_M1 * pA2_pc

        y6M1 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_M1
            + tmp3 / d_M1 * (pd_pc - d_M1 / A2_M1 * pA2_pc)
            + tmp2 * _imPQ_M
        )

        # M2
        pd_pc = (tmp - _ONE / kes_M2) * pd_prho_M2 + heston_params.eps * msqr / d_M2
        pA1_pc = m_M2 * calp_M2 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_M2
            - _ONE
            / imPQ_M
            * (_TWO / (market_params.T[market_pointer] * kes_M2) + _ONE)
            * pA1_prho_M2
            + heston_params.eps * halft * A1_M2
        )
        pA_pc = pA1_pc / A2_M2 - A_M2 / A2_M2 * pA2_pc
        y6M2 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_M2
            + tmp3 / d_M2 * (pd_pc - d_M2 / A2_M2 * pA2_pc)
            + tmp2 * imPQ_M
        )

        # N1
        pd_pc = (tmp - _ONE / kes_N1) * pd_prho_N1 + heston_params.eps * _nsqr / d_N1
        pA1_pc = m_N1 * calp_N1 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_N1
            - _ONE
            / (_imPQ_N)
            * (_TWO / (market_params.T[market_pointer] * kes_N1) + _ONE)
            * pA1_prho_N1
            + heston_params.eps * halft * A1_N1
        )
        pA_pc = pA1_pc / A2_N1 - A_N1 / A2_N1 * pA2_pc
        y6N1 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_N1
            + tmp3 / d_N1 * (pd_pc - d_N1 / A2_N1 * pA2_pc)
            + tmp2 * _imPQ_N
        )

        # N2
        pd_pc = (tmp - _ONE / kes_N2) * pd_prho_N2 + heston_params.eps * nsqr / d_N2
        pA1_pc = m_N2 * calp_N2 * halft * pd_pc
        pA2_pc = (
            tmp * pA2_prho_N2
            - _ONE
            / (imPQ_N)
            * (_TWO / (market_params.T[market_pointer] * kes_N2) + _ONE)
            * pA1_prho_N2
            + heston_params.eps * halft * A1_N2
        )
        pA_pc = pA1_pc / A2_N2 - A_N2 / A2_N2 * pA2_pc
        y6N2 = (
            -heston_params.v0 * pA_pc
            - tmp1 * lnB_N2
            + tmp3 / d_N2 * (pd_pc - d_N2 / A2_N2 * pA2_pc)
            + tmp2 * imPQ_N
        )

        hM1 = h_M * y1M1
        hM2 = h_M * y1M2
        hN1 = h_N * y1N1
        hN2 = h_N * y1N2

        jacobian = _TagMNJac(
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
        return jacobian

    def _heston_vanilla_premium(
        self,
        heston_params: HestonParams,
        market_params: MarketParams
    ) -> np.array:
        """Calculates the premium of vanilla option under the Heson model."""
        n = len(market_params.K)
        x = np.zeros(n, dtype=np.float64)
        for l in range(n):
            K = market_params.K[l]
            T = market_params.T[l]
            disc = np.exp(-market_params.r * T)
            tmp = 0.5 * (market_params.S - K * disc)
            # tmp = 0.5 * (market_parameters.S - K) * disc
            disc = disc / _PI
            y1, y2 = nb.float64(0.0), nb.float64(0.0)

            MN: _TagMn = self._hes_int_MN(
                heston_params=heston_params,
                market_params=market_params,
                market_pointer=np.int32(l),
            )
            y1 = y1 + np.multiply(_W64, (MN.M1 + MN.N1)).sum()
            y2 = y2 + np.multiply(_W64, (MN.M2 + MN.N2)).sum()
            Qv1, Qv2 = np.float64(0.0), nb.float64(0.0)
            Qv1 = _Q * y1
            Qv2 = _Q * y2
            pv = np.float64(0.0)
            delta = 0.5 + Qv1 / _PI
            # print(delta)

            if market_params.types[l]:
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

    def _hes_int_MN(
        self,
        heston_params: HestonParams,
        market_params: MarketParams,
        market_pointer: int,
    ) -> _TagMn:
        csqr = np.power(heston_params.eps, 2)

        PQ_M, PQ_N = _P + _Q * _U64, _P - _Q * _U64
        imPQ_M = _I * PQ_M
        imPQ_N = _I * PQ_N
        _imPQ_M = _I * (PQ_M - _I)
        _imPQ_N = _I * (PQ_N - _I)

        h_M = np.divide(np.power(market_params.K[market_pointer], -imPQ_M), imPQ_M)
        h_N = np.divide(np.power(market_params.K[market_pointer], -imPQ_N), imPQ_N)

        x0 = (
        np.log(market_params.S)
        + market_params.r * market_params.T[market_pointer]
        )
        tmp = heston_params.eps * heston_params.rho

        kes_M1 = heston_params.kappa - np.multiply(tmp, _imPQ_M)
        kes_N1 = heston_params.kappa - np.multiply(tmp, _imPQ_N)
        kes_M2 = kes_M1 + tmp
        kes_N2 = kes_N1 + tmp

        m_M1 = imPQ_M + _ONE + np.power(PQ_M - _I, 2)
        m_N1 = imPQ_N + _ONE + np.power(PQ_N - _I, 2)
        m_M2 = imPQ_M + np.power(PQ_M - _ZERO * _I, 2)
        m_N2 = imPQ_N + np.power(PQ_N - _ZERO * _I, 2)

        d_M1 = np.sqrt(np.power(kes_M1, 2) + m_M1 * csqr)
        d_N1 = np.sqrt(np.power(kes_N1, 2) + m_N1 * csqr)
        d_M2 = np.sqrt(np.power(kes_M2, 2) + m_M2 * csqr)
        d_N2 = np.sqrt(np.power(kes_N2, 2) + m_N2 * csqr)

        tmp1 = (
        -heston_params.kappa
        * heston_params.theta
        * heston_params.rho
        * market_params.T[market_pointer]
        / heston_params.eps
        )

        # вот эта строка была пропущено и цены плохо считались 16.02.2023
        tmp = np.exp(tmp1)
        g_M2 = np.exp(tmp1 * imPQ_M)
        g_N2 = np.exp(tmp1 * imPQ_N)
        g_M1 = g_M2 * tmp
        g_N1 = g_N2 * tmp

        tmp = 0.5 * market_params.T[market_pointer]
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

        tmp = 2 * heston_params.kappa * heston_params.theta / csqr
        halft = 0.5 * market_params.T[market_pointer]

        D_M1 = (
        np.log(d_M1)
        + (heston_params.kappa - d_M1) * halft
        - np.log(
        (d_M1 + kes_M1) * 0.5
        + (d_M1 - kes_M1)
        * 0.5
        * np.exp(-d_M1 * market_params.T[market_pointer])
        )
        )
        D_M2 = (
        np.log(d_M2)
        + (heston_params.kappa - d_M2) * halft
        - np.log(
        (d_M2 + kes_M2) * 0.5
        + (d_M1 - kes_M2)
        * 0.5
        * np.exp(-d_M2 * market_params.T[market_pointer])
        )
        )
        D_N1 = (
        np.log(d_N1)
        + (heston_params.kappa - d_N1) * halft
        - np.log(
        (d_N1 + kes_N1) * 0.5
        + (d_N1 - kes_N1)
        * 0.5
        * np.exp(-d_N1 * market_params.T[market_pointer])
        )
        )
        D_N2 = (
        np.log(d_N2)
        + (heston_params.kappa - d_N2) * halft
        - np.log(
        (d_N2 + kes_N2) * 0.5
        + (d_N2 - kes_N2)
        * 0.5
        * np.exp(-d_N2 * market_params.T[market_pointer])
        )
        )

        MNbas = _TagMn(
        np.real(
        h_M * np.exp(x0 * _imPQ_M - heston_params.v0 * A_M1 + tmp * D_M1) * g_M1
        ),
        np.real(
        h_N * np.exp(x0 * _imPQ_N - heston_params.v0 * A_N1 + tmp * D_N1) * g_N1
        ),
        np.real(
        h_M * np.exp(x0 * imPQ_M - heston_params.v0 * A_M2 + tmp * D_M2) * g_M2
        ),
        np.real(
        h_N * np.exp(x0 * imPQ_N - heston_params.v0 * A_N2 + tmp * D_N2) * g_N2
        ),
        )
        return MNbas

    def _jac_hes(
        self,
        heston_params: HestonParams,
        market_params: MarketParams,
    ) -> np.array:
        """Computes Jacobian w.r.t. HestonParams.

        **IMPORTANT**: *the order of rows is changed compared to the original
        implementation, so that they correspond to the new ordering of Heston
        parameters: v0, kappa, theta, eps, rho*

        Returns the array with shape (M, N_POINTS), where M = 5 (the number of
        Heston parameters, and N_POINTS is the number of market points).
        """
        n = np.int32(len(market_params.K))
        r = market_params.r

        da, db, dc, drho, dv0 = (
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
        )
        jacs = np.zeros((5, n), dtype=np.float64)
        for l in range(n):
            K = market_params.K[l]
            T = market_params.T[l]
            discpi = np.exp(-r * T) / _PI
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
            jacint: _TagMNJac = self._hes_int_jac(
                heston_params=heston_params,
                market_params=market_params,
                market_pointer=l,
            )
            pa1 += np.multiply(_W64, jacint.pa1s).sum()
            pa2 += np.multiply(_W64, jacint.pa2s).sum()

            pb1 += np.multiply(_W64, jacint.pb1s).sum()
            pb2 += np.multiply(_W64, jacint.pb2s).sum()

            pc1 += np.multiply(_W64, jacint.pc1s).sum()
            pc2 += np.multiply(_W64, jacint.pc2s).sum()

            prho1 += np.multiply(_W64, jacint.prho1s).sum()
            prho2 += np.multiply(_W64, jacint.prho2s).sum()

            pv01 += np.multiply(_W64, jacint.pv01s).sum()
            pv02 += np.multiply(_W64, jacint.pv02s).sum()

            # (initial) Variance (v0)
            Qv1 = _Q * pv01
            Qv2 = _Q * pv02
            dv0 = discpi * (Qv1 - K * Qv2)
            jacs[0][l] = dv0

            # VarReversion (kappa)
            Qv1 = _Q * pa1
            Qv2 = _Q * pa2
            da = discpi * (Qv1 - K * Qv2)
            jacs[1][l] = da

            # AverageVar (theta)
            Qv1 = _Q * pb1
            Qv2 = _Q * pb2
            db = discpi * (Qv1 - K * Qv2)
            jacs[2][l] = db

            # VolOfVar (eps)
            Qv1 = _Q * pc1
            Qv2 = _Q * pc2
            dc = discpi * (Qv1 - K * Qv2)
            jacs[3][l] = dc

            # Correlation (rho)
            Qv1 = _Q * prho1
            Qv2 = _Q * prho2
            drho = discpi * (Qv1 - K * Qv2)
            jacs[4][l] = drho

        return jacs

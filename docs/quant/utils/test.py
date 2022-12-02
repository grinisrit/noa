from optlib.Solver import *

stock = Underlying(price=65,
                   volatility=0.3,
                   interest=0.)

EuCall = Option(call=False,
                strike=50)

solver = Solver(underlying=stock,
                option=EuCall,
                xSteps=100,
                tSteps=80)

solver.solve_crank_nickolson()
solver.plot3D(cut=True, mod=Mode.VEGA)
# solver.plot3D(cut=True, mod=Mode.NORM)
# solver.plot3D(cut=True, mod=Mode.DIFF_BSM)

# solver.plot(mod=Mode.BSM, cut=True)
# solver.plot3D(mod=Mode.BSM, cut=True)


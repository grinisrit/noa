from optlib.Solver import *

stock = Underlying(price=65,
                   volatility=0.3,
                   interest=0.0)

EuCall = Option(call=False,
                strike=50)

solver = Solver(underlying=stock,
              option=EuCall,
              xSteps=100,
              tSteps=80)

solver.solve_brennan_schwarz()
#solver.plot3D(cut=False, mod=Mode.HEAT)
#solver.plot3D(cut=False, mod=Mode.NORM)
#solver.plot3D(cut=False, mod=Mode.DIFF_BSM)
#
solver.plot(cut=True, mod=Mode.NORM)
solver.plot3D(cut=True, mod=Mode.DIFF_BSM)

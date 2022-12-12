from docs.quant.heat_equation.HeatSolver import *

solver = HeatSolver(100, 200)

solver.CN()
solver.backup()
solver.plot3D()

solver.diff_t()
solver.plot3D(mod=Mode.DIFF_T)

solver.diff_x()
solver.plot3D(mod=Mode.DIFF_X)

solver.diff_sigma()
solver.plot3D(mod=Mode.DIFF_C)

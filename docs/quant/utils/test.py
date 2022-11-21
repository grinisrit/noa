from optlib.Solver import *

stock = Underlying(price=65,
                   volatility=0.3,
                   interest=0.01)

EuCall = Option(call=True,
                strike=50)

grid = Solver(underlying=stock,
              option=EuCall,
              xSteps=300,
              tSteps=200)

grid.plot(cut=False,
            mod=Mode.HEAT)

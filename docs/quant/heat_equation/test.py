from docs.quant.heat_equation.heat_solver import *

sigma_func = lambda x, t: x**2 + (t-0.5)**2 + 0.1

solver = HeatSolver(100, 200, sigma=1, xLeft=-1.0, xRight=1.0)
solver.set_sigma(sigma_func)


# printing sigma
surface = go.Surface(z=solver.sigma_net, x=solver.tGrid, y=solver.xGrid)
fig = go.Figure(surface)
fig.update_layout(title='sigma', autosize=False, width=1200, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.update_scenes(xaxis_title_text='t', yaxis_title_text='x')
fig.show()


solver.CN()  # solving va crank-nicolson
solver.backup()  # cashing
solver.plot3D()

solver.diff_t()
solver.plot3D(mod=Mode.DIFF_T)

solver.diff_x()
solver.plot3D(mod=Mode.DIFF_X)

solver.diff_sigma()
solver.plot3D(mod=Mode.DIFF_C)

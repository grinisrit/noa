# ODE solvers tutorial

[TOC]

## Introduction

In this part, we describe solvers of [ordinary differential equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation) having the
following form:

\f[ \frac{d \vec u}{dt} = \vec f( t, \vec u) \text{ on } (0,T), \f]
\f[  \vec u( 0 )  = \vec u_{ini}. \f]


TNL offers the following ODE solvers:

1. \ref TNL::Solvers::ODE::Euler - the Euler method with the 1-st order of accuracy.
2. \ref TNL::Solvers::ODE::Merson - the Runge-Kutta-Merson solver with the 4-th order of accuracy and adaptive choice of the time step.

Each solver has its static counterpart which can be run even in the GPU kernels which means that it can be combined with \ref TNL::Algorithms::ParallelFor for example. The static ODE solvers are the following:

1. \ref TNL::Solvers::ODE::StaticEuler - the Euler method with the 1-st order of accuracy.
2. \ref TNL::Solvers::ODE::StaticMerson - the Runge-Kutta-Merson solver with the 4-th order of accuracy and adaptive choice of the time step.

## Static ODE solvers

Static solvers are supposed to be used mainly when \f$ x \in R \f$ is scalar or \f$ \vec x \in R^n \f$ is vector where \f$ n \f$ is small. Firstly, we show example of scalar problem of the following form:

\f[ \frac{d u}{dt} = t \sin ( t )\ \rm{ on }\ (0,T), \f]

\f[ u( 0 )  = 0. \f]


This problem can be solved as follows:

\includelineno Solvers/ODE/StaticODESolver-SineExample.h

We first define the type `Real` representing the floating-point arithmetics, which is `double` in this case (line 4). In the main function, we define the type of the ODE solver ( `ODESolver`, line 8). We choose \ref TNL::Solvers::ODE::StaticEuler. We define the variable `final_t` (line 9) representing the size of the time interval \f$ (0,T)\f$, next we define the integration time step `tau` (line 10) and the variable `output_time_step` (line 11) representing checkpoints in which we will print value of the solution \f$ x(t)\f$. On the line 13, we create an instance of the `ODESolver` and set the integration time step (line 14) and the initial time of the solver (line 15). Next we create variable `u` representing the solution of the given ODE and we initiate it with the initial condition \f$ u(0) = 0\f$ (line 16). On the lines 17-25, we iterate over the interval \f$ (0,T) \f$ with step given by the frequency of the checkpoints given by the variable `output_time_steps`. On the line 19, we let the solver to iterate until the next checkpoint or the end of the interval \f$(0,T) \f$ depending on what occurs first (it is expressed by `TNL::min( solver.getTime() + output_time_step, final_t )`). On the lines 20-22, we define the lambda function `f` representing the right-hand side of the ODE being solved. The lambda function receives the following arguments:

* `t` is the current value of the time variable \f$ t \in (0,T)\f$,
* `tau` is the current integration time step,
* `u` is the current value of the solution \f$ u(t)\f$,
* `fu` is a reference on a variable into which we evaluate the right-hand side \f$ f(u,t) \f$ on the ODE.

The lambda function is supposed to compute just the value of `fu`. It is `t * sin(t)` on our case. Finally we call the ODE solver (line 23). As parameters, we pass the variable `u` representing the solution \f$ u(t)\f$ and a lambda function representing the right-hand side of the ODE. On the line 23, we print values of the solution at given checkpoints. The result looks as follows:

\include StaticODESolver-SineExample.out

Such data can by visualized using [Gnuplot](http://www.gnuplot.info/) as follows

```
plot 'StaticODESolver-SineExample.out' with lines
```

or it can be processed by the following [Python](https://www.python.org/) script which draws graph of the function \f$ u(t) \f$ using [Matplotlib](https://matplotlib.org/).

\includelineno Solvers/ODE/StaticODESolver-SineExample.py

 We first parse the input file (lines 13-20) and convert the data into [NumPy](https://numpy.org/) arrays (lines 24-25). Finaly, these arrays are drawn into a graph (lines 29-36). The graph of the solution looks as follows:

\image{inline} html StaticODESolver-SineExample.png "Solution of the scalar ODE problem"

In the next example, we demonstrate use of the static ODE solver to solve a system of ODEs, namely the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) which reads as:

\f[ \frac{dx}{dt} = \sigma( x - y),\ \rm{ on }\ (0,T) \f]
\f[ \frac{dy}{dt} = x(\rho - z ) - y,\ \rm{ on }\ (0,T)  \f]
\f[ \frac{dz}{dt} = xy - \beta z,\ \rm{ on }\ (0,T) \f]
\f[ \vec u(0) = (x(0),y(0),z(0)) = \vec u_{ini} \f]

for given constants \f$ \sigma, \rho \f$ and \f$ \beta \f$. The solution \f$ \vec u(t) = (x(t), y(t), z(t)) \in R^3 \f$ is represented by three-dimensional static vector ( \ref TNL::Containers::StaticVector). The solver looks as follows:

\includelineno Solvers/ODE/StaticODESolver-LorenzExample.h

The code is very similar to the previous example. There are the following differences:

1. We define the type of the variable `u` representing the solution \f$ \vec u(t) \f$ as \ref TNL::Containers::StaticVector< 3, Real > (line 9) which is reflected even in the definition of the ODE solver (\ref TNL::Solvers::ODE::StaticEuler< Vector > , line 10) and the variable `u` (line 21).
2. In addition to the parameters of the solver ( `final_t`, `tau` and `output_time_step`, lines 14-16) we define parameters of the Lorenz system (`sigma`, `rho` and `beta`, lines 14-16).
3. The initial condition \f$ \vec u(0) = (1,2,3) \f$ is set on the line 21.
4. In the lambda function representing the right-hand side of the Lorenz system (lines 25-32), we first define auxiliary aliases `x` ,`y` and `z` (lines 26-28) to make the code easier to read. The main right-hand side of the Lorenz system is implemented on the lines (29-31).
5. In the line 3, we print all components of the vector `u`.

The solver creates file with the solution \f$ (\sigma(i \tau), \rho( i \tau), \beta( i \tau )) \f$ for '\f$ i = 0, 1, \ldots N \f$ on separate lines. It looks as follows:

```
sigma[ 0 ] rho[ 0 ] beta[ 0 ]
sigma[ 1 ] rho[ 1 ] beta[ 1 ]
sigma[ 2 ] rho[ 2 ] beta[ 2 ]
...
```

Such file can visualized using [Gnuplot](http://www.gnuplot.info) interactively in 3D as follows

```
splot 'StaticODESolver-LorenzExample.out' with lines
```

or it can be processed by the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-LorenzExample.py

The script has very similar structure as in the previous example. The result looks as follows:

\image{inline} html StaticODESolver-LorenzExample.png "Solution of the Lorenz problem"

## Combining static ODE solvers with parallel for

The static solvers can be used inside of lambda functions for \ref TNL::Algorithms::ParallelFor for example. This can be useful when we need to solve large number of independent ODE problems, for example for parametric analysis. We demonstrate it on the two examples we have described above.

### Solving scalar problems in parallel

The first example solves ODE given by

\f[ \frac{d u}{dt} = t \sin ( c t )\ \rm{ on }\ (0,T), \f]

\f[ u( 0 )  = 0, \f]

where \f$ c \f$ is a constant. We will solve it in parallel ODEs with different values \f$ c \in \langle c_{min}, c_{max} \rangle \f$. The exact solution can be found [here](https://www.wolframalpha.com/input?i=y%27%28t%29+%3D++t+sin%28+a+t%29). The code reads as follows:

\includelineno Solvers/ODE/StaticODESolver-SineParallelExample.h

In this example we also show, how to run it on GPU. Therefore we moved the main solver to separate function `solveParallelODEs` which has one template parameter `Device` telling on what device it is supposed to run. The results of particular ODEs are stored in a memory and at the end they are copied to a file with given filename `file_name`. The variable \f$ u \f$ is scalar therefore we represent it by the type `Real` in the solver (line 12). Next we define parameters of the ODE solver (`final_t`, `tau` and `output_time_step`, lines 14-16), interval for the parameter of the ODE \f$ c \in \langle c_{min}, c_{max} \rangle \f$ ( `c_min`, `c_max`, lines 17-18) and number of values `c_vals` (line 19) distributed equidistantly in the interval with step `c_step` (line 20). We use the number of different values of the parameter `c` as a range for the parallel for on the line 43. This parallel for processes the lambda function `solve` which is defined on the lines 28-42. It receives a parameter `idx` which is index of the value of the parameter `c`. We compute its value on the line 29. Next we create the ODE solver (line 30) and setup its parameters (lines 31-32). We set initial condition of the given ODE and we define variable `time_step` which counts checkpoints which we store in the memory in vector `results` allocated in the line 23 and accessed in the lambda function via vector view `results_view` (defined on the line 24). We iterate over the interval \f$ (0, T) \f$ in the  while loop starting on the line 36. We set the stop time of the ODE solver (line 38) and we run the solver (line 39). Finally we store the result at given checkpoint into vector view `results_view`. If the solver runs on the GPU, it cannot write the checkpoints into a file. This is done in postprocessing in the lines 45-53.

Note, how we pass the value of parameter `c` to the lambda function `f`. The method `solve` of the ODE solvers(\ref TNL::Solvers::ODE::StaticEuler::solve, for example) accepts user defined parameters via [variadic templates](https://en.wikipedia.org/wiki/Variadic_template). It means that in addition to the variable `u` and the right-hand side `f` we can add any other parameters like `c` in this example (line 39). This parameter appears in the lambda function `f` (line 25). The reason for this is that the `nvcc` compiler (version 10.1) does not accept lambda function defined within another lambda function. If such a construction is accepted by a compiler, the lambda function `f` which can be defined within the lambda function `solve` and the variable `c` defined in the lambda function `solve` could be captured by `f`.

The solver generates file of the following format:

```
# c = c[ 0 ]
x[ 0 ] u( c[ 0 ], x[ 0 ] )
x[ 1 ] u( c[ 0 ], x[ 1 ] )
....

# c = c[ 1 ]
x[ 0 ] u( c[ 1 ], x[ 0 ] )
x[ 1 ] u( c[ 1 ], x[ 1 ] )
...
```

The file an visualized using [Gnuplot](http://www.gnuplot.info) as follows

```
splot 'StaticODESolver-SineParallelExample-result.out' with lines
```

or it can be processed by the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-SineParallelExample.py

The result of this example looks as follows:

\image{inline} html StaticODESolver-SineParallelExample.png ""


### Solving the Lorenz system in parallel

The second example is a parametric analysis of the Lorenz model

\f[ \frac{dx}{dt} = \sigma( x - y),\ \rm{ on }\ (0,T) \f]
\f[ \frac{dy}{dt} = x(\rho - z ) - y,\ \rm{ on }\ (0,T)  \f]
\f[ \frac{dz}{dt} = xy - \beta z,\ \rm{ on }\ (0,T) \f]
\f[ \vec u(0) = (x(0),y(0),z(0)) = \vec u_{ini} \f]

 which we solve for different values of the model parameters

 \f[ \sigma_i = \sigma_{min} + i  \Delta \sigma, \f]
 \f[ \rho_j = \rho_{min} + j  \Delta \rho, \f]
 \f[ \beta_k = \beta_{min} + k \Delta \beta, \f]

 where we set \f$ \Delta \sigma = \Delta \rho = \Delta \beta = l / (p-1) \f$ and where \f$ i,j,k = 0, 1, \ldots, p - 1 \f$. The code of the solver looks as follows:

\includelineno Solvers/ODE/StaticODESolver-LorenzParallelExample.h

It is very similar to the previous one. There are just the following changes:

1. On the line 17, we define minimal values for the parameters \f$ \sigma, \beta \f$ and \f$ \rho \f$. On the line 18, we define how many different values we will consider for each parameter. The size of equidistant steps in the parameter variations is defined on the line 19. The interval for parameters variations is set to \f$ [0,30] \f$ (line 19 as well).
2. On the line 23, we allocate vector `results` into which we will store the solution of the Lorenz problem for various parameters.
3. Next we define the lambda function `f` representing the right-hand side of the Lorenz problem (lines 25-33) and the lambda function `solve` representing the ODE solver for the Lorenz problem with particular setup of the parameters (lines 34-52). This lambda function is processed by \ref TNL::Algorithms::ParallelFor3D called on the line 53. Therefore the lambda function `solve` receives three indexes `i`, `j` and `k` which are used to compute particular values of the parameters \f$ \sigma_i, \rho_j, \beta_k \f$ which are represented by variables `sigma_i`, `rho_j` and `beta_k` (lines 35-37). These parameters must be passed to the lambda function `f` explicitly (line 48). The reason is the same as in the previous example - nvcc (version 10.1) does not accept a lambda function defined within another lambda function.
4. The initial condition for the Lorenz problem is set to vector \f$ (1,1,1) \f$ (line 42). Finally, we start the time loop (lines 45-51) and we store the state of the solution into the vector `results` using related vector view `results_view` in the time intervals given by the variable `output_time_step`.
5. When all ODEs ares solved, we copy all the solutions from the vector `results` into an output file.

The files has the following format:


```
# sigma = c[ 0 ] rho = rho[ 0 ] beta = beta[ 0 ]
x[ 0 ] u( sigma[ 0 ], rho[ 0 ], beta[ 0 ], x[ 0 ] )
x[ 1 ] u( sigma[ 0 ], rho[ 0 ], beta[ 0 ], x[ 1 ] )
....

# sigma = c[ 1 ] rho = rho[ 1 ] beta = beta[ 1 ]
x[ 0 ] u( sigma[ 1 ], rho[ 1 ], beta[ 1 ], x[ 0 ] )
x[ 1 ] u( sigma[ 1 ], rho[ 1 ], beta[ 1 ], x[ 1 ] )
...
```

The file can be processed by the following [Python](https://www.python.org) script:

\includelineno Solvers/ODE/StaticODESolver-LorenzParallelExample.py

The result looks as follows:

\image{inline} html StaticODESolver-LorenzParallelExample-1.png ""
\image{inline} html StaticODESolver-LorenzParallelExample-2.png ""

\image{inline} html StaticODESolver-LorenzParallelExample-3.png ""
\image{inline} html StaticODESolver-LorenzParallelExample-4.png ""

## Non-static ODE Solvers

In this section, we will show how to solve simple [heat equation](https://en.wikipedia.org/wiki/Heat_equation) in 1D. The heat equation is a parabolic partial differential equation which reads as follows:

\f[
\frac{\partial u(t,x)}{\partial t} - \frac{\partial^2 u(t,x)}{\partial^2 x} = 0\ \rm{on}\ (0,T) \times (0,1),
\f]

\f[
u(t,0) = 0,
\f]

\f[
u(t,0) = 1,
\f]

\f[
u(0,x) = u_{ini}(x)\ \rm{on}\ [0,1].
\f]

We discretize it by the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method) to get numerical approximation. We first define set of nodes \f$ x_i = ih \f$ for \f$i=0,\ldots n-1 \f$ where \f$h = 1 / (n-1) \f$ (we use C++ indexing for the consistency). Using the [method of lines](https://en.wikipedia.org/wiki/Method_of_lines) and approximating the second derivative by the central finite diference (\f$ \frac{\partial^2 u(t,x)}{\partial^2 x} \approx \frac{u_{i-1} - 2 u_i + u_{i+1}}{h^2} \f$), we obtain system of ODEs of the following form

\f[
\frac{\rm{d}u_i(t)}{\rm{d} t} = \frac{u_{i-1} - 2 u_i + u_{i+1}}{h^2}\ \rm{for}\ i = 1, \ldots, n-2,
\f]

where \f$ u_i(t) = u(t,ih) \f$ and \f$ h \f$ space step between two adjacent nodes of the numerical mesh. We also set

\f[
u_0(t) = u_{n-1}(t) = 0\ \rm{on}\ [0,T].
\f]


What are the main differences compared to the Lorenz model?

1. The size of the Lorenz model is fixed. It is equal to three since \f$ (\sigma, \rho, \beta) \in R^3 \f$ which is small vector of fixed size and it can be represented by the static vector \ref TNL::Containers::StaticVector< 3, Real >. On the other hand, the size of the ODE system arising in the discretization by the [method of lines](https://en.wikipedia.org/wiki/Method_of_lines) depends not on the problem we solve but on the desired accuracy - the larger \f$ n \f$ the more accurate numerical approximation we get. The number of nodes \f$ n \f$ used for the space discretisation defines the number of parameters defining the mesh function. These parameters are also referred as [degrees of freedom, DOFs](https://en.wikipedia.org/wiki/Degrees_of_freedom). Therefore the size of the system can be large and it is better to employ dynamic vector \ref TNL::Containers::Vector for the solution.
2. The size of the Lorenz model is small and so the evaluation of its right-hand side can be done sequentially by one thread. The size of the ODE system can be very large and evaluating the right-hand side asks for being performed in parallel.
3. The dynamic vector \ref TNL::Containers::Vector allocates data dynamically and therefore it cannot be created within a GPU kernel which means the ODE solvers cannot be created in the GPU kernel either. For this reason, the lambda function `f` evaluating the right-hand side of the ODE system is always executed on the host and it calls \ref TNL::Algorithms::ParallelFor to evaluate the right-hand side of the ODE system.

### Basic setup

The implementation of the solver looks as follows:

\includelineno Solvers/ODE/ODESolver-HeatEquationExample.h

The solver is implemented in separate function `solveHeatEquation` (line 11) having one template parameter `Device` (line 10) defining on what device the solver is supposed to be executed. We first define necessary types:

1. `Vector` (line 13) is alias for \ref TNL::Containers::Vector. The number of DOFs can be large and therefore they are stored in the resizable dynamically allocated vector \ref TNL::Containers::Vector rather then in static vector - \ref TNL::Containers::StaticVector.
2. `VectorView` (line 14) will be used for accessing DOFs within the lambda functions.
3. `ODESolver` (line 15) is type of the ODE solver which we will use for the computation of the time evolution in the time dependent heat equation.

Next we define parameters of the discretization:

1. `final_t` (line 20) represents the size of the time interval \f$ [0,T] \f$.
2. `output_time_step` (line 21) defines time intervals in which we will write the solution \f$ u \f$ into a file.
3. `n` (line 22) is the number of DOFs, i.e. number of nodes used for the [finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method).
4. `h` (line 23) is the space step, i.e. distance between two consecutive nodes.
5. `tau` (line 24) is the time step used for the integration by the ODE solver. Since we solve the second order parabolic problem, the time step should be proportional to \f$ h^2 \f$.
6. `h_sqr_inv` (line 25) is auxiliary constant equal to \f$ 1/h^2 \f$ which we use later in finite difference for approximation of the second derivative.

Next we set the initial condition \f$ u_{ini} \f$ (lines ) which is given as:

\f[
u_{ini}(x) = \left\{
   \begin{array}{rl}
   0 & \rm{for}\ x < 0.4, \\
   1 & \rm{for}\ 0.4 \leq x \leq 0.6, \\
   0 & \rm{for}\ x > 0. \\
   \end{array}
\right.
\f]

Next we write the initial condition to a file (lines 37-39) using the function `write`  which we will describe later. On the lines (44-46) we create instance of the ODE solver `solver`, we set the integration time step `tau` of the solver (\ref TNL::Solvers::ODE::ExplicitSolver::setTau ) and we set the initial time to zero (\ref TNL::Solvers::ODE::ExplicitSolver::setTime).

Finally, we proceed to the time loop (lines 52-64) but before we prepare counter of the states to be written into files (`output_idx`). The time loop uses the time variable within the ODE solver (\ref TNL::Solvers::ODE::ExplicitSolver::getTime ) and it iterates until we reach the end of the time interval \f$ [0, T] \f$ given by the variable `final_t`. On the line 54, we set the stop time of the ODE solver (\ref TNL::Solvers::ODE::ExplicitSolver::setStopTime ) to the next checkpoint for storing the state of the heat equation or the end of the time interval depending on what comes first. Next we define the lambda function `f` expressing the discretization of the second derivative of \f$ u \f$ by the central finite difference and the boundary conditions. The function receives the following parameters:

1. `i` is the index of the node and the related ODE arising from the method of lines. In fact, we have to evaluate the update of \f$ u_i^k \f$ to get to the next time level \f$ u_i^{k+1} \f$.
2. `u` is vector view representing the state \f$ u_i^k \f$ of the heat equation on the \f$ k- \f$ time level.
3. `fu` is vector of updates or time derivatives in the method of lines which will bring \f$ u \f$ to the next time level.

As we mentioned above, since `nvcc` does not accept lambda functions defined within another lambda function, we have to define `f` separately and pass the parameters `u` and `fu` explicitly (see the line 62).

Now look at the code of the lambda function `f`. Since the solution \f$ u \f$ does not change on the boundaries, we return zero on the boundary nodes (lines 56-57) and we evaluate the central difference for approximation of the second derivative on the interior nodes (line 59).

Next we define the lambda function `time_stepping` (lines ) which is responsible for computing of the updates for all nodes \f$ i = 0, \ldots n-1 \f$. It is done by means of \ref TNL::Algorithms::ParallelFor which iterates over all the nodes and calling the function `f` on each of them. It passes the vector views `u` and `fu` explicitly to `f` for the reasons we have mentioned above.

Finally, we run the ODE solver (\ref TNL::Solvers::ODE::Euler::solve ) (line 63) and we pass `u` as the current state of the heat equation and `f` the lambda function controlling the time evolution to the method `solve`. On the line 64, we store the current state to a file.

The function `write` which we use for writing the solution of the heat equation reads as follows:

\includelineno Solvers/ODE/write.h

The function accepts the following parameters:

1. `file` is the file into which we want to store the solution.
2. `u` is a vector or vector view representing solution of the heat equation at given time.
3. `n` is number of nodes the we use for the approximation of the solution.
4. `h` is space step, i.e. distance between two consecutive nodes.
5. `time` is the current time of the evolution being computed.

The solver writes the results in the following format:

```
# time = t[ 0 ]
x[ 0 ] u( t[ 0 ], x[ 0 ] )
x[ 1 ] u( t[ 0 ], x[ 1 ] )
x[ 2 ] u( t[ 0 ], x[ 2 ] )
...

# time = t[ 1 ]
x[ 0 ] u( t[ 1 ], x[ 0 ] )
x[ 1 ] u( t[ 1 ], x[ 1 ] )
x[ 2 ] u( t[ 1 ], x[ 2 ] )
...
```

The solution can be visualised with [Gnuplot](http://www.gnuplot.info/) as follows:

```
plot 'ODESolver-HeatEquationExample-result.out' with lines
```
or it can be easily parsed in [Python](https://www.python.org/) and processes by [Matplotlib](https://matplotlib.org/) using the following script:

\includelineno Solvers/ODE/ODESolver-HeatEquationExample.py

The result looks as follows:

\image{inline} html ODESolver-HeatEquationExample.png "Heat equation"

### Setup with a solver monitor

In this section we will show how to connect ODE solver with the solver monitor. Look at the following example:

\includelineno Solvers/ODE/ODESolver-HeatEquationWithMonitorExample.h

There are the following differences compared to the previous example:

1. We have to include a header file with the iterative solver monitor (line 5).
2. We have to setup the solver monitor (lines 45-50). First, we define the monitor type (line 45) and we create an instance of the monitor (line 46). Next we create a separate thread for the monitor (line 47), set the refresh rate to 10 milliseconds (line 48), turn on the verbose mode (line 49) and set the solver stage name (line 50). On the line 58, we connect the monitor with the solver using method \ref TNL::Solvers::IterativeSolver::setSolverMonitor. We stop the monitor after the ODE solver finishes by calling \ref TNL::Solvers::IterativeSolverMonitor::stopMainLoop in the line 78.

The result looks as follows:

\include /ODESolver-HeatEquationWithMonitorExample.out

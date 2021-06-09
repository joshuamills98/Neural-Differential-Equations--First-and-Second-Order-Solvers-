# Neural-Differential-Equations--First-and-Second-Order-Solvers-
In this project I experiment with Physics-Informed Neural Networks (PINNs) to solve first and second order ODEs, this code was developed to assist in my Master's thesis on physics-informed neural networks implemented engineering development platforms.

# First-order method
For the first order method the goal is to solve the equation


![equation](https://latex.codecogs.com/gif.latex?u%27%20%3D%20f%28u%2Ct%29)


Where the initial condition is given by:


![equation](https://latex.codecogs.com/gif.latex?u%280%29%20%3D%20u_%7B0%7D)


The physics-informed neural network is implemented to reduce the loss term given by:


![equation](https://latex.codecogs.com/gif.latex?L%28%5Clambda%29%20%3D%20%5Csum_%7Bi%7D%5CBig%28%5Cfrac%7Bdg%28t_%7Bi%7D%29%7D%7Bdt%7D-f%28g%28t_%7Bi%7D%29%2Ct_%7Bi%7D%29%5CBig%29%5E%7B2%7D)


Where:


![equation](https://latex.codecogs.com/gif.latex?g%28t%29%20%3D%20u_%7B0%7D%20&plus;%20tNN%28t%29)


g(t) is a universal approximator to u(t).


# Second-order method
For the second order method, the PINN is used to solve an equation of the form

![equation](https://latex.codecogs.com/gif.latex?u%27%27%28y%29%20%3D%20f%28y%29)

With


![equation](https://latex.codecogs.com/gif.latex?u%27%280%29%20%3D%20u%27_%7B0%7D)

and


![equation](https://latex.codecogs.com/gif.latex?u%280%29%20%3D%20u_%7B0%7D)


The PINN learns the form of u(t) without needing any knowledge of the output except for the initial condition.

Please feel free to use the code.

Attached shows the solution determined by the first-order ODE solver to the equation:

![equation](https://latex.codecogs.com/gif.latex?u%27%28t%29%20%3D%20-%5Csin%7B%28t%7D%29&plus;t)

![image](/FirstOrderPINNExample.png)


I was heavily inspired by https://www.youtube.com/watch?v=hKHl68Fdpq4&ab_channel=ParallelComputingandScientificMachineLearning which presents a great demonstration of the theory behind PINNs!

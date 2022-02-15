# NeODEs
## Deep Neural Network for systems of ODEs

Given two coupled system of differential equations with initial conditions, the code will perform: an experiment on the choice of numbers of neurons for better performance, experimentally compares the effectiveness of one versus two hidden layer for the given problem, solve deep neural network solution and compare with the analytical solution if closed form is provided. Lastly, it compares the deep neural network with the RK4 method.

### functions.py
The functions.py file contains collections of funcitons for excecuting differenti experiments and solving systems of ordinary differential equations. 

### Experiment1.py

Experiment for comparing the performance of the network as a fucntion of numbers of neuron.
The code is for system of differential equations with two unknowns. 

### Experiment2.py
Experiment for comparing one hidden layer vs two hidden layers. The code is for system of differential equations with two uknowns.

### Experiment3.py
It is the code for implementation of the ANN.
A system of differential equations with two unkown. Compared with analytical solution.

### Experiment4.py
Compares the Deep neural network with Runge-Kutta order 4


@article{dufera2021deep,
  title={Deep neural network for system of ordinary differential equations: vectorized algorithm and simulation},
  author={Dufera, Tamirat Temesgen},
  journal={Machine Learning with Applications},
  volume={5},
  pages={100058},
  year={2021},
  publisher={Elsevier}
}


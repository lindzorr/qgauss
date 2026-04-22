<h1>QGauss: Calculations for quantum Gaussian systems</h1>

The qgauss module allows for the efficient simulation of hybrid quantum systems composed of a mix of continuous-variables systems and finite-level systems, where the state of the continuous variable systems are said to be Gaussian. The Numpy and Scipy packages are necessary for this module to function. The syntax for functions and class methods in this project are chosen to match those of QuTiP as much as possible. The ease of use and flexibility of QuTiP was also the inspiration for this project.

<h2>Use</h2>

To use this module, simply download and import:

`import qgauss`

Currently, documentation is located along with the classes and functions in their .py files. The writing of proper documentation is a future goal. To avoid having to look through the code itself, two Jupyter notebooks are included to give examples of the syntax, and the current capabilities of the classes and functions.

<h2>Implementation</h2>

This project was motivated by the need to simulate the measurement of qubits by amplifiers comprised of sufficiently many open quantum harmonic oscillators, where the number basis representation becomes impractical to handle the open system dynamics both accurately and quickly. To this end, this module contains three classes to handle these systems: one for states, one for operators, and one for superoperators. The restriction to Gaussian systems allows for states to be represented by their moments. Operators and superoperators are at most bilinear functions of the harmonic oscillator quadrature operators (or, creation and annihilation operators), and so may be represented by their coefficients. The finite-level system components are still represented using the standard matrix representation.

This mixed representation results in less required memory and faster computing times, at the expense of a slightly messier backend. The defined classes support the expected arithmetic operations with scalars, as well as operations to combine objects of the same class. In order to better understand the logic of the backend, it is recommended to review the basics of continuous variable quantum systems and their connection to the Wigner phase-space representation. 

<h2>Functionality</h2>

Motivated by the need to model qubit measurement, the module currently includes functions that allow for the calculation of state of the hybrid system, along with quantities related to quantum nondemolition measurements. These functions allow for the calculation of steady-state backaction and measurement rates, along with time-domain functions for the backaction. Efficient implementation of the signal-to-noise ratio for time-dependent systems is ongoing. Other ideas to extend functionality may be found in the TODO file. Current functionality is directed, but is not limited to, systems of qubits. 

Given that this is a personal project, some bugs are to be expected. The code is therefore offered as is under the MIT license, as specified in the LICENSE file.

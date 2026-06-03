<h1>QGauss: Calculations for quantum Gaussian systems</h1>

The qgauss module allows for the efficient simulation of hybrid quantum systems composed of a mix of continuous-variable systems and finite-level systems, where the state of the continuous variable systems are said to be Gaussian. The Numpy and Scipy packages are necessary for this module to function. The syntax for functions and class methods in this project are chosen to match those of QuTiP as much as possible. The ease of use and flexibility of QuTiP was also the inspiration for this project.

<h2>Use</h2>

To use this module, simply download and import:

`import qgauss`

Currently, documentation is located along with the classes and functions in their .py files. The writing of proper documentation is a future goal. To avoid having to look through the code itself, example Jupyter notebooks are included to give an introduction to the syntax, and the current capabilities of the module.

<h2>Implementation</h2>

This project was motivated by the need to simulate the measurement of qubits by amplifiers comprised of sufficiently many open quantum harmonic oscillators, where the number basis representation becomes impractical to handle the open system dynamics both accurately and quickly. To this end, this module contains three classes to handle these systems: one for states, one for operators, and one for superoperators. The restriction to Gaussian states and transformations which preserve this property allows for far more efficient computations. In this module, Gaussian states are represented purely by their moments. Operators and superoperators are at most bilinear functions of the harmonic oscillator quadrature operators (or, creation and annihilation operators), and so may be represented by their coefficients. The finite-level system components are still represented using the standard matrix representation.

This mixed representation results in less required memory and faster computations times, at the expense of the slightly messier backend required to represent the data. The defined classes support the expected arithmetic operations with scalars, as well as operations to combine objects of the same class. In order to better understand the logic of this module, it is recommended to review the basics of continuous variable quantum systems and their connection to the Wigner phase-space representation. 

<h2>Functionality</h2>

Although motivated by the need to model qubit measurement, the module is not limited to just two-level systems coupled to a network of harmonic modes. The current functions can handle any finite-level system, including qudits or qubits including their leakage states. Given that the modelling of measurements is the goal, the included functions allow for calculation of the backaction of the measurement along with the measurement rate. Currently, these functions can calculate the steady-state solutions and the time-evolution of the system modes, along with the associated backaction. Only the steady-state measurement rate can currently be handled, with development of the time-domain signal-to-noise ratio function ongoing and of foremost priority. 

Given the ubiquity of the Heisenberg-Langevin equations when modelling measurement, a specific class is included to handle these equations in terms of the three base classes. While this class works, its implementation is not yet finalized. The development of an additional class to better handle time-dependent is also underway. Other ideas to extend functionality may be found in the TODO file.

Given that this is a personal project, some bugs are to be expected. Additionally, the implementation of certain functions or objects may be changed in a manner which breaks backwards compatibility. The code is therefore offered as is under the MIT license, as specified in the LICENSE file.

## Introduction

![faradlogo](faradlogo.png)

Farad (Forward and Reverse Automatic Differentiation) is a PyPi-distributed Python library for performing forward and reverse automatic differentiation, empowering the user with the ability to efficiently generate function derivatives with machine precision.

## Background

**Automatic differentiation** (AD) is a method of computing function derivatives in a programmatic fashion. It can be contrasted with **symbolic differentiation**, which involves manual calculation and encoding of the derivative by a user, and **numerical differentiation**, which entails the use of finite difference methods to approximate derivatives. Both of these methods are suboptimal.

Symbolic differentiation requires either user input or a near-infinitely sized lookup table, neither of which are practically feasible when working with large numbers of functions or especially complex functions. Numerical differentiation does not typically require explicit user input, but approximations often become poor due to the magnification of floating point errors, severely limiting the accuracy of approximations.

AD, on the other hand, resolves these issues. Only elementary functions are encoded, and complex functions are decomposed into a series of operations with elementary functions using a computational graph structure. This is done via the chain rule, which allows derivatives to be fragmented into simpler ones that become trivial to solve. In addition, the computational complexity of automatic differentiation functions are proportional to the underlying code complexity in most cases, meaning the method can be implemented with only a minor impact on runtime.

## How to Use Farad

As an encapsulated library, a user will easily interact with it by simply importing the library into the notebook.

The user can install the Farad package via the preferred installer program (pip):
```bash
>>> pip install Farad
```
The following example demonstrates how a user will interface with Farad:
```python
>>> import Farad as ad
>>>  def my_pow(x，r) :
...     y =  x ** r
...     Return y
>>> d_power = ad (my_power)
>>> d_power(3, 4)
108
```

## Software Organization

A proposed layout for the directory tree is outlined below. The project will have separate directories for performing tests (actionable by the *pytest* library). The copyright license used will be the permissive MIT License 2.0. Documentation will be created using the Sphinx package and hosted externally. A folder for applications will exist to demonstrate several example use cases of the library in action. Source files for the forward and reverse mode functions will be placed under the *autodiff* folder. A requirements.txt file and setup.py file will be present to allow the package to be distributed via PyPi and for easy user setup of the package.

```
Farad
├── LICENSE
├── README.md     
├── docs
│   └── sphinx_docs
│   └── design documents
├── requirements.txt
├── travis.yml
├── setup.py
├── autodiff
│   └── ...
├── tests
│   └── ...
├── applications
│   └── ...
```

#### Modules
-	Numpy
-	Pandas
-	Sympy (math operations)
-	Matplot (plotting)

#### Continuous Integration
-	TravisCI
-	CodeCov

#### Distribution
-	PyPi

#### Packaging
-	Sphinx?

## Implementation
Our plan on implementing the forward mode of automatic differentiation is as below:

#### Core data structures:
-	Dual number or dual vectors including the value and derivatives

#### Classes to use:
-	DataClass which defines the instance as the dual number structure; FunctionClass which reads user-specified functions as input

#### Methods and name attributes:
- Methods include all the mathematical operations like plus, multiplication, division, sin, power, log, exp, tanh, etc.
-	Name attributes include value and derivative.

#### External dependencies:
-	Numpy

#### How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?
-	Creating (overloading) separate functions as class methods for each of the elementary functions


## Advanced Feature(s)
The advanced features for this project include:
-	The addition of reverse mode automatic differentiation
-	The use of Farad in several use cases involving differential equations

##  RESOURCES TO REFERENCE FOR IDEAS:

https://github.com/autodiff/autodiff<br/>
https://github.com/HIPS/autograd

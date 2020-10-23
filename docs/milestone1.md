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

A proposed layout for the directory tree is outlined below. The project will have separate directories for performing tests (actionable by the pytest library). The copyright license used will be the permissive MIT License 2.0. Documentation will be created using Sphinx and hosted externally on Read The Docs. A folder for applications will exist to demonstrate several example use cases of the library in action. Source files for the forward and reverse mode functions will be placed under the autodiff folder. A requirements.txt file and setup.py file will be present to allow the package to be distributed via PyPi and for easy user setup of the package.

```
master
├── LICENSE
├── README.md     
├── docs
│   ├── sphinx_docs
│   └── design documents
├── requirements.txt
├── travis.yml
├── .readthedocs.yml
├── .gitignore
├── setup.py
├── farad
│   ├── __init__.py
│   └── ...
├── tests
│   ├── __init__.py
│   └── ...
└── applications
    └── ...
```

The main module for this library will be farad, which contains all of the callable submodules used for automatic differentiation. Three other modules (directories) will also be created, (1) docs, containing documentation and milestone information for the project, (2) applications, containing several use case examples of the farad library, and (3) tests, containing unit tests for farad submodules.

The package will be augmented by continuous integration, implemented via TravisCI. Travis performs automated testing of submodules upon changes to source code. In addition, CodeCov will be used for performing coverage assessment of packaged code.

The package will be distributed via PyPi in the format outlined in the above-mentioned directory tree. The only anticipated package dependency will be numpy.

## Implementation

Core data structures: Dual number or dual vectors including the value and derivatives

We will use DataClass and FunctionClass. DataClass defines the instance as the dual number structure and FunctionClass reads user-specified functions as input

Our methods include all the mathematical operations like: plusaddition, multiplication, division, sintrigonometric (sin, cos, tan), power, logarithmic, exponential, hyperbolic (sinh, cosh, tanh,),  etc.as well as multiple complex operators (e.g., arcsin, arctanh, tetration). Methods will be implemented via operator overloading. And the name attributes include function value and derivative.

The only external dependency would be Numpy

For elementary functions like sin, sqrt, log, and exp (and all the others), we will create (overload) separate functions as class methods for each of the elementary functions. Compatibility will also be made for numpy functions (and potentially scipy) for added functionality and cross-compatibility.


## Advanced Feature(s)
-	The addition of reverse mode automatic differentiation
-	The use of Farad in several use cases involving differential equations

##  RESOURCES TO REFERENCE FOR IDEAS:

https://github.com/autodiff/autodiff<br/>
https://github.com/HIPS/autograd <br/>
https://github.com/we-the-diff/cs207-FinalProject

# UCLCHEM v1.5
UCLCHEM is a gas-grain chemical code written in Fortran 95. It propagates the abundances of chemical species through a network of user-defined reactions according to the physical conditions of the gas. Included in the repository is MakeRates, a python script to combine a species list, UMIST reaction file and user-defined reaction file into a consistent network with all files required by UCLCHEM.

**************************************************************
Usage Instructions
**************************************************************

Full documentation is available from the website: uclchem.github.io

To build UCLCHEM, edit the Makefile in uclchem/src to use a compiler available on your machine. Then use "make" to create the executable.
- The Makefile also contains the choice of physics module.
- Building requires odes.f90 and network.f90 which are outputs of Makerates.
- uclchem/Makerates/ contains the Makerates python script to produce a network from the files in uclchem/Makerates/inputFiles

To run UCLCHEM, create an input file with the desired parameters. Any variable in defaultparameters.f90 can be set, any that are not will take the value given in defaultparameters.f90.
A full explanation of each parameter is given in defaultparameters.f90 and an example input file is given in example.inp
Call uclchem with the filename as an argument: "./uclchem example.inp"

**************************************************************
Python
**************************************************************
"Make python" builds a python module from the source code and wrap.f90. This can be imported into a python script and any subroutine in wrap.f90 is a function in the module. Currently, this is just uclchem.general() which takes a dictionary of any parameters in defaultparameters.f90 and runs the code.
An example script, grid.py in the scripts folder runs a grid of models by repeatedly calling that function. This demonstrates the basic use of the wrapper and how to use python Pool objects to parallize runing a grid.

Currently, the wrapper must be recompiled for different physics modules.

**************************************************************
Change Log
**************************************************************
**UCLCHEM output**
We've made a large change to the UCLCHEM outputs. The full output file is now columnated after a 2 line header that describes model features which do not change. Python scripts that are packaged with UCLCHEM have been updated to account for this.

**New Network**
Our default network now uses the diffusion and desorption mechanisms introduced by Quenard et al. 2018 for the grain network. A pseudo-grain network that simply hydrogenates species is still included for those who wish to use it.

**Rate Calculation**
Vectorized rate calculations to simplify code.

**Shock update**
C and J type shocks now allow the user to set a minimum temperature at which point the post shock gas stops cooling.

*************************************************************
Contributing
*************************************************************
This is an open source science code for the community and we are happy to accept pull requests. We are also happy to work with you to produce a physics module if none of the ones available in the repository suit the modelling work you wish to do. If you are contributing ,please try to work with our current code style. We have the following general guidelines:

- camelCase variable and subroutines names that are self-explanatory where possible 

- CAPITALIZED fortran built in functions to make code structure apparent.
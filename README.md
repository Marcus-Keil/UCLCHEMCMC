Required Packages

'pandas', 'numpy', 'corner', 'matplotlib', 'emcee', 'billiard', 'bokeh', 'flask', 'celery'


Installation Instructions:

Compile UCLCHEM:
In directory 

    /UCLCHEMCMC/src/UCLCHEM/src/ 

call 

    make
    make python

Once this completes, take the "uclchem.so" file from 

    /UCLCHEMCMC/src/UCLCHEM/

to 

    /UCLCHEMCMC/src/

Compile Spectral Radex:
In directory 

    /UCLCHEMCMC/src/SpectralRadex/radex/src

call 

    make
    make python

To Run UCLCHEMCMC:
In directory 

    /UCLCHEMCMC/

call

    bash runUCLCHEMCMC.sh

Upon finishing that, open a browser and go to following address 

    http://localhost:5000/

From here, the user interface should guide the user on how to proceed.

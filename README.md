Welcome to UCLCHEMCMC, the MCMC inference tool using chemical and radiative transfer full forward modeling. 

![Inference Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/InferencePage.png)
![Chemicals Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/ChemicalPage.png)
![Results Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/ResultsPage.png)

**************************************************************
Installation Instructions:
**************************************************************

Required Packages:
    pandas numpy corner matplotlib emcee billiard bokeh flask celery

Compile UCLCHEM:

In directory 

    /UCLCHEMCMC/src/UCLCHEM/src/ 

call 

    make
    make python

Once this completes, take the "uclchem.so" file from 

    /UCLCHEMCMC/src/UCLCHEM/

Compile Spectral Radex:

In directory 

    /UCLCHEMCMC/src/SpectralRadex

call 

    python3 setpu.py

To Run UCLCHEMCMC:

In directory 

    /UCLCHEMCMC/

call

    bash runUCLCHEMCMC.sh
    
OR

in three different terminals, or terminal windows, call the following:
    
    bash ./run-redis.sh
    python GUI.py
    celery worker -A GUI.celery --loglevel=info

Upon finishing that, open a browser and go to following address 

    http://localhost:5000/

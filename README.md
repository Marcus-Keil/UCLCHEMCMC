We present the publicly available, open source code UCLCHEMCMC, designed to estimate physical parameters of an observed cloud of gas by combining Monte Carlo Markov Chain (MCMC) sampling with chemical and radiative transfer modeling. When given the observed values of different emission lines, UCLCHEMCMC runs a Bayesian parameter inference, using a MCMC algorithm to sample the likelihood and produce an estimate of the posterior probability distribution of the parameters. UCLCHEMCMC takes a full forward modeling approach, generating model observables from the physical parameters via chemical and radiative transfer modeling. While running UCLCHEMCMC, the created chemical models and radiative transfer code results are stored in an SQL database, preventing redundant model calculations in future inferences.

For more details, we will add a link to the ApJ (Accepted) and astro-ph publication as soon as they go live.

**************************************************************
# Installation Instructions:
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

**************************************************************
# Usage Instruction
**************************************************************

Once the browser has been opened to localhost, and the Inference tab on the left has been selected. You should see the ability to input the name of a previous session or the option to choose a Coarse or Fine Grid. Inputing a session name and hitting load will take you to the Results page, while selecting a Grid should take you to the following page:

![Inference Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/ReadmeImages/InferencePage.png)

Upon submitting and verifying the Parameter ranges, you will be greeted by the page to choose species and input observed emission line information as seen below.

![Chemicals Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/ReadmeImages/ChemicalPage.png)

This is followed by the options page, where a session name can be given, number of walkers can be choosen and the amount of steps the inference should take prior to saving the inference can be input. This then takes you to the final page, the results where you have the option to start the inference as well as see the current Corner Plots of the inference.

![Results Page](https://github.com/Marcus-Keil/UCLCHEMCMC/blob/main/ReadmeImages/ResultsPage.png)


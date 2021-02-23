Required Packages

'pandas', 'numpy', 'corner', 'matplotlib', 'emcee', 'billiard', 'bokeh', 'flask', 'celery'


Installation Instructions:

    In directory "/UCLCHEMCMC/src/UCLCHEM/src/" call "make" followed by "make python" 
    move the resulting "uclchem.so" file from "/UCLCHEMCMC/src/UCLCHEM/" to 
    "/UCLCHEMCMC/src/".

    In directory "/UCLCHEMCMC/src/SpectralRadex/" call "pip install ."

    In directory "/UCLCHEMCMC/" call "pip install ." if you want to make changes to
    UCLCHEMCMC we recommend the command "pip install -e ." instead

To Run UCLCHEMCMC:

    In directory "/UCLCHEMCMC/" call "bash runUCLCHEMCMC.sh" making sure to type in
    the sudo password for the initial terminal. Upon finishing that, proceed to 
    opening a browser and going to address "http://localhost:5000/" and follow
    the instructions there.

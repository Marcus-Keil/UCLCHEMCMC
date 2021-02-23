import time
import numpy as np
import billiard as Bil
import billiard.pool as BilPool
import SpectralRadex.radex as radex
import utils
import MCFunctions as mcf
import os


#outSpecies = "C+ C p-C3H2 o-C3H2 CH-nohfs CH-H CH-H2 CH3CN CN CN-hfs C2H o-CH2 p-CH2 e-CH3OH a-CH3OH CO CS HC3N-H2 HC3N-H2-hfs HCl HCN HCO+ HCS+ HNC HNCO p-H3O+ o-H3O+ N+ p-NH3 o-NH3 N2H+ NO O2 O OCS OH OH+ o-H2CO-H2 p-H2CO-H2 o-H2CS p-H2CS o-H2O p-H2O o-H2S p-H2S o-SiC2 SiS SO"

#uclchemdict = {"phase": 1, "switch": 1, "collapse": 1, "readAbunds": 0, "writeStep": 1, "points": 1, "desorb": 1,
#               "finalOnly": "True", "fr": 1.0, "outSpecies": 7, "finalDens": 5500000.0, "initialTemp": 18.0,
#               "radfield": 0.4, "rout": 0.0085, "zeta": 0.06}
#
#
#ManagerPoolSize = 1
#ManagerPool = BilPool.Pool(ManagerPoolSize, utils.worker_main)
#FortranPoolSize = 2
#FortranPool = BilPool.Pool(FortranPoolSize, utils.worker_Fortran)
#physDF, chemDF = utils.UCLChemDataFrames(UCLChemDict=uclchemdict)
#print("Finished UCLCHEM")
#UCLParamOut = utils.RadexForGrid(UCLChemDict=uclchemdict, UCLParamDF=physDF, UCLChemDF=chemDF)
#print(UCLParamOut)
#for i in range(FortranPoolSize):
#    mcf.FortranQueue.put(("Stop", []))
#    time.sleep(1)
#FortranPool.close()
#FortranPool.join()
#mcf.SQLQueue.put(("Stop", []))
#ManagerPool.close()
#ManagerPool.join()

RadexDict = {'molfile': 'sis.dat', 'tkin': 20.0, 'tbg': 2.73, 'cdmol': 67423.99232643691, 'h2': 900874.1746561746,
             'h': 4.819763654011721e-07, 'e-': 8.907623524461051e-10, 'p-h2': 0.0, 'o-h2': 0.0,
             'h+': 8.907623524461051e-10, 'linewidth': 1.0, 'fmin': 0.0, 'fmax': 1000.0}

dataFrame = radex.run(parameters=RadexDict)

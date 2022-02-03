import os
import SpectralRadex.src.spectralradex.radex as radex
import numpy as np
import pandas as pd
import sqlite3 as sql
import time
import json
import MCFunctions as mcf
import GreLVG.GreLVG as glvg
from uclchem import wrap as uclchem

os.environ["OMP_NUM_THREADS"] = "1"

sleepTime = 0.1
timeOutForSQL = 10000

# Reading in of Columns that should be saved into the database
with open("../data/UCLCHEM_SavedModelsColumns.json") as jFile:
    savedModelsColumns = json.load(jFile)

# Declaration of which Database should be used
DBLocation = "../data/DefaultDatabase.db"

# Connection to SQL database, located in memory, to store models that were created during the use of UCLCHEMCMC
Store_con = sql.connect(":memory:", check_same_thread=False)
Store_cur = Store_con.cursor()

# Connection to SQL database, located in memory, for searching the models created in the past
Search_con = sql.connect(":memory:", check_same_thread=False)
Search_cur = Search_con.cursor()

# Reading in of file, and conversion to array of species and their emission lines that
# UCLCHEMCMC should list as available
ChemFile = "../data/Chemicals.csv"
Chem = pd.read_csv(ChemFile, delimiter=",/", engine='python')
Lines = Chem.set_index('Chemical')["Lines"].to_dict()
for C in Lines.keys():
    Lines[C] = np.asarray(Lines[C].split(", ")).astype(str)

# Dictionary keys of which parameters can be varied for the Radiative transfer code in the User interface
RadiativeCode_Keys = {'linewidth'}


def InitDatabases(KnownParameters, UnknownParameters, Ranges, PDLinesJson):
    """
    Initialise the Databases that are stored in Memory

    Args:
        KnownParameters: Dictionary of parameters and fixed values, set either by default, or by the user
        UnknownParameters: List of Parameters that are being changed by the MCMC
        Ranges: Dictionary of upper and lower bounds of each parameter
        PDLinesJson: Json file containing the lines of interest that were supplied by the user
    """
    Search_cur.execute('SELECT name FROM main.sqlite_master WHERE type="table" AND name="savedModels";')
    fetched = Search_cur.fetchall()
    if len(fetched) > 0:
        return None
    DiskDatabase = sql.connect(DBLocation)
    DiskDatabase_cur = DiskDatabase.cursor()
    LinesOfInterest = pd.read_json(PDLinesJson)
    chemicals = np.asarray([chemDat(x) for x in LinesOfInterest['Chemical'].unique()])
    ColumsToSearch = np.array([], dtype=str)
    for index, row in LinesOfInterest.iterrows():
        ColumsToSearch = np.append(ColumsToSearch,
                                   (chemDat(row['Chemical'])[:-4] + '_' + row['Line'] + ' GHz)_' + row['Units']))
    DiskDatabase_cur.execute('SELECT name FROM main.sqlite_master WHERE type="table";')
    Tables_Temp = DiskDatabase_cur.fetchall()
    TablesToCopy = [Tables_Temp[x][0] for x in range(len(Tables_Temp))]
    # Initiate database to be used for the new models that will be created
    Store_cur.execute("ATTACH DATABASE '" + DBLocation + "' AS 'DiskDatabase';")
    Search_cur.execute("ATTACH DATABASE '" + DBLocation + "' AS 'DiskDatabase';")
    Store_con.commit()
    Search_con.commit()
    Conditions = ''
    for k in KnownParameters.keys():
        if k != 'outSpecies':
            if Conditions != '':
                Conditions += ' AND '
            else:
                Conditions += 'DiskDatabase."savedModels"."' + k + '" = ' + str(KnownParameters[k])
    for k in UnknownParameters:
        if Conditions != '':
            Conditions += ' AND '
        Conditions += 'DiskDatabase."savedModels"."' + k + '" >= ' + str(Ranges[k + '_low']) + \
                      ' AND DiskDatabase."savedModels"."' + k + '" <= ' + str(Ranges[k + '_up'])
    SearchIDS = Search_cur.execute(
        'SELECT ModelID FROM DiskDatabase."savedModels" WHERE (' + Conditions + ');').fetchall()
    for i in TablesToCopy:
        if i != 'sqlite_sequence':
            DiskDatabase_cur.execute('PRAGMA table_info("' + i + '");')
            Col_Fetched = np.asarray(DiskDatabase_cur.fetchall())
            StoreCol_Store = '('
            SearchCol_Store = ''
            for j in Col_Fetched:
                if StoreCol_Store != '(':
                    StoreCol_Store += ', '
                Extras = ''
                if j[3] == 1 or j[1] == 'ModelID':
                    Extras += ' NOT NULL'
                if j[5] == 1:
                    Extras += ' PRIMARY KEY'
                StoreCol_Store += '"' + j[1] + '" ' + j[2] + Extras
                if j[1] in UnknownParameters or (i + '.dat' in chemicals and j[1] in ColumsToSearch):
                    if SearchCol_Store != '':
                        SearchCol_Store += ', '
                    SearchCol_Store += 'DiskDatabase."' + i + '"."' + j[1] + '"'
            StoreCol_Store += ');'
            Store_cur.execute('CREATE TABLE "' + i + '"' + StoreCol_Store)
            if SearchCol_Store != '':
                Search_cur.execute('CREATE TABLE "' + i + '" AS SELECT ModelID, ' + SearchCol_Store +
                                   ' FROM "' + i + '";')
                if i == 'savedModels':
                    Search_cur.execute(
                        'DELETE FROM main."' + i + '" WHERE main.ModelID NOT IN (' + str(SearchIDS)[1:-1] + ');')
                    Search_con.commit()
                else:
                    Search_cur.execute(
                        'DELETE FROM main."' + i + '" WHERE main.ModelID NOT IN (' + str(SearchIDS)[1:-1] + ');')
                    Search_con.commit()
    Store_cur.execute("DETACH DATABASE 'DiskDatabase';")
    Search_cur.execute("DETACH DATABASE 'DiskDatabase';")
    DiskDatabase.close()
    Store_con.commit()
    Search_con.commit()
    return None


def UpdateSearchDatabase(table, SID):
    """
    Update the Database in Memory that is intended to be searched by the MCMC

    Args:
         table: Pandas Dataframe of models that are not in the Searched database, but are in the Stored database
         SID: Highest ID value of the database on the Disk to be added to the ID of models to be stored
    """
    Search_cur.execute('PRAGMA table_info("' + table + '");')
    Col_Fetched = np.asarray(Search_cur.fetchall())
    Search = 'SELECT "' + '", "'.join(Col_Fetched[:, 1]) + '" FROM main."' + table + '";'
    TempDF = pd.read_sql(Search, Store_con)
    TempDF["ModelID"] = TempDF["ModelID"] + SID
    TempDF.to_sql(table, Search_con, if_exists='append', index=False)
    Search_con.commit()
    return None


"""
dict_factory is a function that allows for a switcher like function to be created

Functions chemForUCLCHEM, chemDat, and ReverseChemForUCLCHEM rely on dict_factory, and each 
take a key like in a dictionary and return the value of that dictionary. 
"""
# =============================================================================================================
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def chemForUCLCHEM(i):
    switcher = {
        'C+': 'C+',
        'C': 'C',
        'p-C3H2': 'C3H2',
        'o-C3H2': 'C3H2',
        'CH-nohfs': 'CH',
        'CH-H': 'CH',
        'CH-H2': 'CH',
        'CH3CN': 'CH3CN',
        'CN': 'CN',
        'CN-hfs': 'CN',
        'C2H': 'C2H',
        'o-CH2': 'CH2',
        'p-CH2': 'CH2',
        'e-CH3OH': 'CH3OH',
        'a-CH3OH': 'CH3OH',
        'CO': 'CO',
        'CS': 'CS',
        '13C16O': 'CO',
        'C17O': 'CO',
        'C18O': 'CO',
        'HC3N-H2': 'HC3N',
        'HC3N-H2-hfs': 'HC3N',
        'HCl': 'HCL',
        'HCN': 'HCN',
        'HCO+': 'HCO+',
        'H13CO+': 'HCO+',
        'HC17O+': 'HCO+',
        'HC18O+': 'HCO+',
        'HCS+': 'HCS+',
        'HNC': 'HCN',
        'HNCO': 'HNCO',
        'p-H3O+': 'H3O+',
        'o-H3O+': 'H3O+',
        'N+': 'N+',
        'p-NH3': 'NH3',
        'o-NH3': 'NH3',
        'N2H+': 'N2H+',
        'NO': 'NO',
        'O2': 'O2',
        'O': 'O',
        'OCS': 'OCS',
        'OH': 'OH',
        'OH+': 'OH+',
        'o-H2CO-H2': 'H2CO',
        'p-H2CO-H2': 'H2CO',
        'o-H2CS': 'H2CS',
        'p-H2CS': 'H2CS',
        'o-H2O': 'H2O',
        'p-H2O': 'H2O',
        'o-H2S': 'H2S',
        'p-H2S': 'H2S',
        'o-SiC2': 'SIC2',
        'SiS': 'SIS',
        'SO': 'SO'}
    return switcher.get(i, i + " is an invalid chemical given to chemForUCLCHEM")


def ReverseChemForUCLCHEM(i):
    switcher = {
        'C+': 'C+',
        'C': 'C',
        'C3H2': 'p-C3H2, o-C3H2',
        'CH': 'CH-nohfs, CH-H, CH-H2',
        'CH3CN': 'CH3CN',
        'CN': 'CN, CN-hfs',
        'C2H': 'C2H',
        'CH2': 'o-CH2, p-CH2',
        'CH3OH': 'e-CH3OH, a-CH3OH',
        'CO': 'CO, 13C16O, C17O, C18O',
        'CS': 'CS',
        'HC3N': 'HC3N-H2, HC3N-H2-hfs',
        'HCL': 'HCl',
        'HCN': 'HCN',
        'HCO+': 'HCO+, H13CO+, HC17O+, HC18O+',
        'HCS+': 'HCS+',
        'HNC': 'HCN',
        'HNCO': 'HNCO',
        'H3O+': 'p-H3O+, o-H3O+',
        'N+': 'N+',
        'NH3': 'p-NH3, o-NH3',
        'N2H+': 'N2H+',
        'NO': 'NO',
        'O2': 'O2',
        'O': 'O',
        'OCS': 'OCS',
        'OH': 'OH',
        'OH+': 'OH+',
        'H2CO': 'o-H2CO-H2, p-H2CO-H2',
        'H2CS': 'o-H2CS, p-H2CS',
        'H2O': 'o-H2O, p-H2O',
        'H2S': 'o-H2S, p-H2S',
        'SIC2': 'o-SiC2',
        'SIS': 'SiS',
        'SO': 'SO',
        'SO2': 'SO2'
    }
    return switcher.get(i, i + " is an invalid chemical given to ReverseChemForUCLCHEM")


def chemDat(i):
    switcher = {
        'C+': 'c+@uv.dat',
        'C': 'catom.dat',
        'p-C3H2': 'p-c3h2.dat',
        'o-C3H2': 'o-c3h2.dat',
        'CH-nohfs': 'ch-nohfs.dat',
        'CH-H': 'ch-h.dat',
        'CH-H2': 'ch-h2.dat',
        'CH3CN': 'ch3cn.dat',
        'CN': 'cn.dat',
        'CN-hfs': 'cn-hfs.dat',
        'C2H': 'c2h_h2_e.dat',
        'o-CH2': 'ch2_h2_ortho.dat',
        'p-CH2': 'ch2_h2_para.dat',
        'e-CH3OH': 'e-ch3oh.dat',
        'a-CH3OH': 'a-ch3oh.dat',
        'CO': 'co.dat',
        'CS': 'cs@lique.dat',
        '13C16O': '13co.dat',
        'C17O': 'c17o.dat',
        'C18O': 'c18o.dat',
        'HC3N-H2': 'hc3n-h2.dat',
        'HC3N-H2-hfs': 'hc3n-h2-hfs.dat',
        'HCl': 'hcl.dat',
        'HCN': 'hcn.dat',
        'HCO+': 'hco+@xpol.dat',
        'H13CO+': 'h13co+@xpol.dat',
        'HC17O+': 'hc17o+@xpol.dat',
        'HC18O+': 'hc18o+@xpol.dat',
        'HCS+': 'hcs+@xpol.dat',
        'HNC': 'hnc.dat',
        'HNCO': 'hnco.dat',
        'p-H3O+': 'p-h3o+.dat',
        'o-H3O+': 'o-h3o+.dat',
        'N+': 'n+.dat',
        'p-NH3': 'p-nh3.dat',
        'o-NH3': 'o-nh3.dat',
        'N2H+': 'n2h+@xpol.dat',
        'NO': 'no.dat',
        'O2': 'o2.dat',
        'O': 'oatom.dat',
        'OCS': 'ocs@xpol.dat',
        'OH': 'oh.dat',
        'OH+': 'oh+.dat',
        'o-H2CO-H2': 'oh2co-h2.dat',
        'p-H2CO-H2': 'ph2co-h2.dat',
        'o-H2CS': 'oh2cs.dat',
        'p-H2CS': 'ph2cs.dat',
        'o-H2O': 'oh2o@daniel.dat',
        'p-H2O': 'ph2o@daniel.dat',
        'o-H2S': 'oh2s.dat',
        'p-H2S': 'ph2s.dat',
        'o-SiC2': 'o-sic2.dat',
        'SiS': 'sis.dat',
        'SO': 'so@lique.dat',
        'SO2': 'so2-highT.dat'
    }
    return switcher.get(i, i + " is an invalid chemical given to chemDat")
# =============================================================================================================


outSpeciesList = list(Lines.keys())
outSpeciesString = " ".join(list(Lines.keys()))
UniqueOutSpecies = np.unique(np.asarray([chemForUCLCHEM(x) for x in outSpeciesList])).tolist()
UniqueOutSpeciesString = 'H H+ H2 ' + ' '.join(UniqueOutSpecies)
UniqueOutSpecieslist = ['H', 'H+', 'H2'] + UniqueOutSpecies


def DistancePercent(BaseFlux):
    """
    Function to return lower and upper bounds for finding models in the database with similar
    observational values

    Args:
        BaseFlux: The value of the observations
    """
    Lower = float(BaseFlux) * 0.8
    Upper = float(BaseFlux) * 1.2
    return Lower, Upper


def UCLChemDataFrames(UCLChemDict, Test=False, Queue=True):
    """
    Creates pandas dataframe of Physical parameters, and Chemical parameters using
    UCLCHEM.
    Args:
        UCLChemDict: Dictionary containing all physical parameters UCLCHEM needs to run
        Test: Boolean that determines if
        Queue: Boolean used if UCLChemDataFrames is called outside of the UCLCHEMCMC code
    """
    stepCount = np.zeros(shape=1, dtype=np.int32, order='F')
    UCLChemDict["outSpecies"] = len(UniqueOutSpecieslist)
    if Queue:
        CurrentPID = os.getpid()
        # Changes for GreLVG run
        mcf.FortranQueue.put(("RunUCLCHEM", [UCLChemDict, UCLChemDict['points'], CurrentPID, stepCount]))
        DictOfArrays = RetrieveFortranQueueResults(CurrentPID)
        parameterArray = DictOfArrays[0]
        chemicalAbunArray = DictOfArrays[1]
        del DictOfArrays
    else:
        parameterArray = np.zeros(shape=(10000, UCLChemDict['points'], 12), dtype=np.float64, order='F')
        chemicalAbunArray = np.zeros(shape=(10000, UCLChemDict['points'], 215), dtype=np.float64, order='F')
        allChemicalAbunArray = np.zeros(shape=(10000, UCLChemDict['points'], 215), dtype=np.float64, order='F')
        uclchem.to_df(dictionary=UCLChemDict, outspeciesin=UniqueOutSpeciesString, numberpoints=UCLChemDict['points'],
                      parameterarray=parameterArray, chemicalabunarray=chemicalAbunArray,
                      allchemicalabunarray=allChemicalAbunArray, stepcount=stepCount)
    if UCLChemDict["finalOnly"]:
        stepCount[0] = 1
    parameterArray = parameterArray[:stepCount[0], :, :]
    chemicalAbunArray = chemicalAbunArray[:stepCount[0], :, :UCLChemDict['outSpecies']]
    if Test:
        UCLChemDict['outputFile'] = '../results/out.dat'
        UCLChemDict['abundFile'] = '../results/abun.dat'
        UCLChemDict['columnFile'] = '../results/col.dat'
        uclchem.general(dictionary=UCLChemDict, outspeciesin=UniqueOutSpeciesString)
    if UCLChemDict["points"] == 1:
        physDF = pd.DataFrame(data=parameterArray[:, 0, :], columns=['ageOfCloudOut', 'hdensOut', 'cloudTempOut',
                                                                     'avOut', 'radiationOut', 'zetaOut',
                                                                     'h2FormRateOut', 'fcOut',
                                                                     'foOut', 'fmgOut', 'fheOut', 'deptOut'])
        physDF['H'] = chemicalAbunArray[:stepCount[0], 0, 0]
        physDF['H+'] = chemicalAbunArray[:stepCount[0], 0, 1]
        physDF['H2'] = chemicalAbunArray[:stepCount[0], 0, 2]
        chemDF = pd.DataFrame(data=chemicalAbunArray[:stepCount[0], 0, 3:], columns=UniqueOutSpecieslist[3:])
    else:
        exit("Currently unable to run with multiple Points")
    return physDF, chemDF


def FortranUCLCHEM(dictionary, numberpoints, CurrentPID, stepCount):
    """
    Run the Fortran wrapper for UCLCHEM so that UCLCHEMCMC adheres to only using as many threads for Fortran as were
    assigned to it.
    Args:
        dictionary: Dictionary containing all physical parameters UCLCHEM needs to run
        numberpoints: Integer of how many points UCLCHEM should use
        CurrentPID: The Process ID of the worker that submitted this job to the queue
        stepCount: the number of steps that should be taken in UCLCHEM
    """
    parameterArray = np.zeros(shape=(10000, dictionary['points'], 12), dtype=np.float64, order='F')
    chemicalAbunArray = np.zeros(shape=(10000, dictionary['points'], 215), dtype=np.float64, order='F')
    allChemicalAbunArray = np.zeros(shape=(10000, dictionary['points'], 215), dtype=np.float64, order='F')
    with suppress_stdout_stderr():
        uclchem.to_df(dictionary=dictionary, outspeciesin=UniqueOutSpeciesString, numberpoints=numberpoints,
                      parameterarray=parameterArray, chemicalabunarray=chemicalAbunArray,
                      allchemicalabunarray=allChemicalAbunArray, stepcount=stepCount)
    time.sleep(sleepTime)
    mcf.FortranResultDict[str(CurrentPID)] = [parameterArray, chemicalAbunArray, allChemicalAbunArray]
    return None


def RadexForGrid(ChemDict, ParamDF, ChemDF, Queue=True, RotDia=False):
    """
    Run radex using the library spectralradex, making sure to take into account the parameter grid.

    Args:
        ChemDict: Dictionary containing all physical parameters UCLCHEM and RADEX need to run
        ParamDF: Dataframe output of physical parameters from UCLCHEM
        ChemDF: Dataframe output of chemical abundances from UCLCHEM
        Queue: Boolean used if UCLChemDataFrames is called outside of the UCLCHEMCMC code
        RotDia: Boolean used if RadexForGrid is being used to create plots which are similar to Rotation Diagrams
    """
    for k in range(np.shape(ParamDF)[0]):
        radexDic = radex.get_default_parameters()
        radexDic['tkin'] = ParamDF.loc[k, 'cloudTempOut']
        radexDic['h2'] = ParamDF.loc[k, 'hdensOut'] * ParamDF.loc[k, 'H2']
        radexDic['h'] = ParamDF.loc[k, 'hdensOut']
        radexDic['h+'] = ParamDF.loc[k, 'hdensOut'] * ParamDF.loc[k, 'H+']
        radexDic['e-'] = ParamDF.loc[k, 'hdensOut'] * ParamDF.loc[k, 'H+']
        radexDic['fmin'] = 35.0
        radexDic['fmax'] = 950.0
        for key in RadiativeCode_Keys:
            if key in ChemDict.keys():
                radexDic[key] = ChemDict[key]
        for i in ChemDF.columns:
            radexDic['cdmol'] = ChemDF.loc[k, i] * 1.6e21 * ParamDF.loc[k, 'avOut']
            if 1e5 <= radexDic['cdmol'] <= 1e25:
                for j in ReverseChemForUCLCHEM(i).split(", "):
                    radexDic['molfile'] = chemDat(j)
                    # try:
                    if RotDia:
                        RadexDF = runRadex(RadexParamDict=radexDic, Queue=Queue, Upper=True)
                    else:
                        RadexDF = runRadex(RadexParamDict=radexDic, Queue=Queue)
                    if RadexDF is np.nan:
                        continue
                    if RotDia:
                        ParamDF = RadexToDF(ChemDict, ParamDF, RadexDF, k, RotDia=True)
                    else:
                        ParamDF = RadexToDF(ChemDict, ParamDF, RadexDF, k)
                    # except:
                    #    pass
            else:
                RadexDF = np.nan
                continue
    return ParamDF


def RadexToDF(ChemDict, ParamDF, RadexArray, indexOfModel, RotDia=False):
    """
    Convert the Array output from runRadex into pandas dataframe
    Args:
        ChemDict: Dictionary containing all physical parameters UCLCHEM and RADEX need to run
        ParamDF: Dataframe output of physical parameters from UCLCHEM
        RadexArray: Array from running RADEX
        indexOfModel:
        RotDia: Boolean used if RadexForGrid is being used to create plots which are similar to Rotation Diagrams
    """
    UCLChemDictKeys = [k for k in ChemDict]
    UCLChemDictValues = [v for v in ChemDict.values()]
    for l in range(len(UCLChemDictKeys)):
        ParamDF[UCLChemDictKeys[l]] = UCLChemDictValues[l]
    for j in range(np.shape(RadexArray)[0]):
        T_rColumnName = RadexArray[j, 0] + '_T_r'
        intColumnName = RadexArray[j, 0] + '_Intensity'
        fluColumnName = RadexArray[j, 0] + '_Flux'
        ParamDF[T_rColumnName] = RadexArray[j, 1]
        ParamDF[intColumnName] = RadexArray[j, 2]
        ParamDF[fluColumnName] = RadexArray[j, 3]
        ParamDF = ParamDF.copy()
        if RotDia:
            EuColumnName = RadexArray[j, 0] + '_Eu'
            if EuColumnName not in ParamDF.columns:
                ParamDF[EuColumnName] = np.nan
            ParamDF[RadexArray[j, 0] + '_Eu'].iloc[indexOfModel] = RadexArray[j, 4]
    return ParamDF


def runRadex(RadexParamDict, Queue=True, Upper=False):
    """
    Take in the RADEX parameter dictionary, and queue the model into the Fortran queue

    Args:
        RadexParamDict: Dictionary containing all of the parameters RADEX needs to run a model
        Queue: Boolean used if UCLChemDataFrames is called outside of the UCLCHEMCMC code
        Upper: Boolean used to return upper state Energy as well, when wanting to create
            plots similar to Rotation Diagrams
    """
    CurrentPID = os.getpid()
    if Queue:
        Checks = ['e-', 'h+']
        if (RadexParamDict['p-h2'] == 0):
            RadexParamDict['p-h2'] = 0.25 * RadexParamDict['h2']
        if (RadexParamDict['o-h2'] == 0):
            RadexParamDict['o-h2'] = 0.75 * RadexParamDict['h2']
        for col in Checks:
            if col in RadexParamDict.keys():
                if RadexParamDict[col] < 1.1e-3:
                    RadexParamDict[col] = 1.1e-3
                elif RadexParamDict[col] > 0.9e13:
                    RadexParamDict[col] = 0.9e13
        mcf.FortranQueue.put(("RunRADEX", [RadexParamDict, CurrentPID]))
        dataFrame = RetrieveFortranQueueResults(CurrentPID)
        if type(dataFrame) is type(None):
            return np.nan
    else:
        dataFrame = radex.run(parameters=RadexParamDict)
    if dataFrame.empty:
        return np.nan
    OutputList = []
    if "_" in dataFrame["Qup"].iloc[0] or " " in dataFrame["Qup"].iloc[0]:
        for i in range(np.shape(dataFrame)[0]):
            TransitionName = RadexParamDict["molfile"][:-4] + "_(" + \
                             dataFrame["Qup"].iloc[i].replace("_", ",").replace(" ", ",").replace(",,", ",") + \
                             ")-(" + \
                             dataFrame["Qlow"].iloc[i].replace("_", ",").replace(" ", ",").replace(",,", ",") + \
                             ")(" + \
                             str(dataFrame["freq"].iloc[i]) + \
                             " GHz)"
            # Name, T_r, Intensity, Flux
            if Upper:
                OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i], dataFrame["FLUX (K*km/s)"].iloc[i],
                                dataFrame["FLUX (erg/cm2/s)"].iloc[i], dataFrame["E_UP (K)"].iloc[i]]]
            else:
                OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i],
                                dataFrame["FLUX (K*km/s)"].iloc[i], dataFrame["FLUX (erg/cm2/s)"].iloc[i]]]
    else:
        for i in range(np.shape(dataFrame)[0]):
            TransitionName = RadexParamDict["molfile"][:-4] + "_" + \
                             dataFrame["Qup"].iloc[i].replace("_", ",") + "-" + \
                             dataFrame["Qlow"].iloc[i].replace("_", ",") + "(" + \
                             str(dataFrame["freq"].iloc[i]) + \
                             " GHz)"
            # Name, T_r, Intensity, Flux
            if Upper:
                OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i], dataFrame["FLUX (K*km/s)"].iloc[i],
                                dataFrame["FLUX (erg/cm2/s)"].iloc[i], dataFrame["E_UP (K)"].iloc[i]]]
            else:
                OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i],
                                dataFrame["FLUX (K*km/s)"].iloc[i], dataFrame["FLUX (erg/cm2/s)"].iloc[i]]]
    OutputArray = np.asarray(OutputList)
    return OutputArray


def FortranRADEX(RadexParamDict, CurrentPID):
    """
    Function created for the Fortran workers to run any models which are in the queue for RADEX
    Args:
        RadexParamDict: Parameter dictionary that RADEX needs in order to run
        CurrentPID: Process ID of the worker that submitted this model to the queue
    """
    with suppress_stdout_stderr():
        dataFrame = radex.run(parameters=RadexParamDict)
    mcf.FortranResultDict[str(CurrentPID)] = dataFrame
    return None


def saveModel(UCLParamDF, ParamDict, CurrentPID):
    """
    Save the models that were created but are not in the database already to the "Store" database in memory
    Args:
        UCLParamDF: Output information given by UCLCHEM and RADEX
        ParamDict: Dictionary that was used to run UCLCHEM and RADEX, not the outputs
        CurrentPID: Process ID of the worker that submitted the call
    """
    UCLParamDF.loc[0, "outSpecies"] = outSpeciesString
    for k in ParamDict.keys():
        if k not in UCLParamDF.columns:
            UCLParamDF[k] = ParamDict[k]
    nonSaveColumns = []
    for i in UCLParamDF.columns.to_list():
        if i not in savedModelsColumns:
            nonSaveColumns += [i]

    UCLCHEMTransitionDF = UCLParamDF[nonSaveColumns]
    UCLParamDF = UCLParamDF.drop(nonSaveColumns, axis=1)
    RedundantCheckSearch = ''
    for i in range(len(UCLParamDF.columns)):
        if i == 0:
            if type(UCLParamDF[UCLParamDF.columns[i]].iloc[0]) == str:
                RedundantCheckSearch = 'SELECT "ModelID" FROM main."savedModels" WHERE "' + \
                                       UCLParamDF.columns[i] + '" = "' + \
                                       UCLParamDF[UCLParamDF.columns[i]].iloc[0] + '"'
            else:
                RedundantCheckSearch = 'SELECT "ModelID" FROM main."savedModels" WHERE "' + \
                                       UCLParamDF.columns[i] + '" = ' + \
                                       str(UCLParamDF[UCLParamDF.columns[i]].iloc[0])
        else:
            if type(UCLParamDF[UCLParamDF.columns[i]].iloc[0]) == str:
                RedundantCheckSearch += ' AND "' + UCLParamDF.columns[i] + '" = "' + \
                                        UCLParamDF[UCLParamDF.columns[i]].iloc[0] + '"'
            else:
                RedundantCheckSearch += ' AND "' + UCLParamDF.columns[i] + '" = ' + \
                                        str(UCLParamDF[UCLParamDF.columns[i]].iloc[0])
    RedundantCheckSearch += ';'
    Store_cur.execute(RedundantCheckSearch)
    RedundantResults = Store_cur.fetchone()
    if type(RedundantResults) == list:
        return None
    Search_cur.execute(RedundantCheckSearch)
    RedundantResults = Search_cur.fetchone()
    if type(RedundantResults) == list:
        return None
    # This needs to have a clause to allow for RAM Saved Model
    Store_cur.execute('SELECT * FROM main."savedModels";')
    UCLParamDF.to_sql(name='savedModels', if_exists='append', con=Store_con, index=False)
    Store_cur.execute('SELECT * FROM main."savedModels";')
    ThisEntryID = (Store_cur.fetchall()[-1][0])
    Store_cur.execute("SELECT tbl_name FROM main.sqlite_master WHERE type='table';")
    tables = Store_cur.fetchall()
    TablesList = []
    for i in tables:
        TablesList += [i[0]]
    columList = UCLCHEMTransitionDF.columns.to_list()
    for i in outSpeciesList:
        datFile = chemDat(i)[:-4]
        columnsOfInterest = []
        for j in range(len(columList)):
            if datFile == columList[j][:len(datFile)] and columList[j][len(datFile)] == '_':
                columnsOfInterest += [columList[j]]

        if len(columnsOfInterest) <= 1000:
            UCLCHEMTransitionDFTemp = UCLCHEMTransitionDF[columnsOfInterest]
            if datFile not in TablesList:
                command = 'CREATE TABLE "' + datFile + \
                          '" (ModelID INTEGER PRIMARY KEY AUTOINCREMENT, FOREIGN KEY(ModelID) REFERENCES savedModels (ModelID));'
                Store_cur.execute(command)
                Store_con.commit()
            Store_cur.execute('SELECT * FROM main."' + datFile + '";')
            columnsInSQL = [description[0] for description in Store_cur.description]
            for Column in list(set(columnsOfInterest) - set(columnsInSQL)):
                command = 'ALTER TABLE "' + datFile + '" ADD "' + Column + '" REAL;'
                Store_cur.execute(command)
                Store_con.commit()
            TempValueArray = ', '.join([str(x) for x in UCLCHEMTransitionDFTemp.to_numpy()[0].tolist()])
            TempValueArray = TempValueArray.replace("nan", "0")
            TempColumnsList = '", "'.join(UCLCHEMTransitionDFTemp.columns.to_list())
            if TempColumnsList == "":
                continue
            command = 'INSERT INTO "' + datFile + '"("' + TempColumnsList + '", ModelID) VALUES (' + \
                      str(TempValueArray) + ', ' + str(ThisEntryID) + ');'
            Store_cur.execute(command)
            Store_con.commit()
        elif len(columnsOfInterest) > 1000:
            columnsLeft = columnsOfInterest
            itterations = int(len(columnsOfInterest) / 1000)
            if len(columnsOfInterest) > ((itterations + 1) * 1000):
                itterations += 1
            for k in range(itterations):
                currentDat = datFile + '_' + str(k)
                columnsThisLoop = columnsLeft[:1000]
                columnsLeft = columnsLeft[1000:]
                if currentDat not in TablesList:
                    command = 'CREATE TABLE "' + currentDat + \
                              '" (ModelID INTEGER PRIMARY KEY AUTOINCREMENT, FOREIGN KEY(ModelID) REFERENCES savedModels (ModelID));'
                    Store_cur.execute(command)
                    Store_con.commit()
                Store_cur.execute('SELECT * FROM main."' + currentDat + '";')
                columnsInSQL = [description[0] for description in Store_cur.description]
                columnsThisLoop = set(columnsThisLoop) - (set(columnsThisLoop) - set(columnsInSQL))
                UCLCHEMTransitionDFTemp = UCLCHEMTransitionDF[columnsThisLoop]
                TempValueArray = ', '.join([str(x) for x in UCLCHEMTransitionDFTemp.to_numpy()[0].tolist()])
                TempValueArray = TempValueArray.replace("nan", "0")
                TempColumnsList = '", "'.join(UCLCHEMTransitionDFTemp.columns.to_list())
                if TempColumnsList == "":
                    continue
                command = 'INSERT INTO "' + currentDat + '"("' + TempColumnsList + '", ModelID) VALUES (' + \
                          str(TempValueArray) + ', ' + str(ThisEntryID) + ');'
                Store_cur.execute(command)
                Store_con.commit()
    # End of clause, for saving.
    mcf.SQLResultDict[str(CurrentPID)] = "Complete"
    return None


def retrieveClosestMatches(LinesArray, DistanceMethod):
    """
    Retrieve any models that fall within the DistanceMethods range as compared to the given emission line values
    Args:
        LinesArray: Array of the emission lines to use for finding models in the database
        DistanceMethod: Function name for the distance method to use for deciding the range of acceptable line values
    """
    for i in range(len(LinesArray[0])):
        baseLineFlux = LinesArray[1, i]
        Lower, Upper = DistanceMethod(baseLineFlux)
        if i == 0:
            sqlSearch = 'SELECT ModelID FROM savedModels WHERE "' + LinesArray[0, i] + '" BETWEEN ' + str(
                Lower) + ' AND ' + str(Upper)
        else:
            sqlSearch = sqlSearch + ' AND "' + LinesArray[0, i] + '" BETWEEN ' + str(Lower) + ' and ' + str(Upper)
    sqlSearch = sqlSearch + ';'
    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)
    return SearchResults.values.T[0]


def retrieveLinesOfEntry(ModelID, LinesOfInterest, ChemsAndLinesDic):
    """
    retrieve the models of a specific entry, and retrieve specific lines
    Args:
        ModelID: ID of the model to Return
        LinesOfInterest: Lines which a user wishes to have returned
        ChemsAndLinesDic: Dictionary of chemical parameters and chemical lines that are desired to be returned
    """
    sqlSearch = None
    FromString = ""
    PreviousDat = ""
    WhereString = ""
    for dats in ChemsAndLinesDic.keys():
        for i in range(len(ChemsAndLinesDic[dats])):
            if sqlSearch == None:
                sqlSearch = 'SELECT "' + dats + '"."' + ChemsAndLinesDic[dats][i] + '"'
                WhereString = ' WHERE "' + dats + '".ModelID = ' + str(ModelID[0]) + ';'
            else:
                sqlSearch = sqlSearch + ', "' + dats + '"."' + ChemsAndLinesDic[dats][i] + '"'
        if FromString == "":
            FromString = ' FROM main."' + dats + '"'
            PreviousDat = dats
        else:
            FromString += ' LEFT JOIN main."' + dats + '" ON main."' + PreviousDat + '".ModelID = main."' + dats + '".ModelID'
            PreviousDat = dats
    if sqlSearch != None:
        sqlSearch = sqlSearch + FromString + WhereString
    elif sqlSearch == None:
        Intensities = np.zeros(len(LinesOfInterest))
        for i in range(len(Intensities)):
            Intensities[i] = np.nan
        return Intensities
    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)
    if SearchResults.empty:
        mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID, True]))
        SearchResults = RetrieveQueueResults(CurrentPID)
        if SearchResults.empty:
            Intensities = np.nan

    if not SearchResults.empty:
        Intensities = SearchResults.iloc[0].values
        for i in range(len(Intensities)):
            if type(Intensities[i]) == str:
                Intensities[i] = np.nan
    return Intensities


def retrieveIntensitiesUI(ParameterDict, ChangingParams, LinesOfInterest, Chemicals, LinesGiven, Test=False):
    """
    Retrieves specific models, specifically the parameters that are used to run the models in UCLCHEM and RADEX or
    the outputs of those models depending on ModelParameters

    Args:
        ParameterDict: Dictionary or Parameters to search that are fixed
        ChangingParams: Dictionary of Parameters to search which are varied by the MCMC
        LinesOfInterest: List of lines which should be retrieved
        Chemicals: List of chemicals that were used
        LinesGiven: List of lines which were given by the user
        Test: Boolean to inform the function that it is currently being tested so that it can give outputs
            in a way that a user can understand where issues may be occuring
    """
    ChangingParamsKeys = list(ChangingParams.keys())
    ChangingParamsValues = list(ChangingParams.values())
    MatchedID = []
    while len(MatchedID) < 1:
        if Test:
            MatchedID = checkAndRetrieveModel(ParameterDict=ParameterDict, ChangingParamsKeys=ChangingParamsKeys,
                                              ChangingParamsValues=ChangingParamsValues, Test=True)
        else:
            MatchedID = checkAndRetrieveModel(ParameterDict=ParameterDict, ChangingParamsKeys=ChangingParamsKeys,
                                              ChangingParamsValues=ChangingParamsValues)
    ChemsAndLines = {}
    UniqueChems = np.unique(Chemicals)
    for i in range(len(UniqueChems)):
        for j in range(len(LinesOfInterest)):
            if len(Lines[UniqueChems[i]]) <= 1000:
                if chemDat(UniqueChems[i])[:-4] in ChemsAndLines.keys() and chemDat(UniqueChems[i])[:-4] in \
                        LinesOfInterest[j]:
                    ChemsAndLines[chemDat(UniqueChems[i])[:-4]] += [LinesOfInterest[j]]
                elif chemDat(UniqueChems[i])[:-4] in LinesOfInterest[j]:
                    ChemsAndLines[chemDat(UniqueChems[i])[:-4]] = [LinesOfInterest[j]]
            else:
                TableNumber = int(np.where(Lines[UniqueChems[i]] == LinesGiven[j] + ' GHz)')[0][0] / 1000)
                ChemTable = chemDat(UniqueChems[i])[:-4] + '_' + str(TableNumber)
                if ChemTable in ChemsAndLines.keys():
                    ChemsAndLines[ChemTable] += [LinesOfInterest[j]]
                else:
                    ChemsAndLines[ChemTable] = [LinesOfInterest[j]]
    Intensities = retrieveLinesOfEntry(MatchedID, LinesOfInterest, ChemsAndLines)
    return Intensities


def checkAndRetrieveModel(ParameterDict, ChangingParamsKeys, ChangingParamsValues, Test=False):
    """
    This checks if a model already exists in the Database, if it does, it returns the Intensities, if it doesn't
    it will run the models to create the outputs and store the results

    Args:
        ParameterDict: Dictionary or Parameters to search that are fixed
        ChangingParamsKeys: Keys of the Dictionary of Parameters to search which are varied by the MCMC
        ChangingParamsValues: Values of the parameters which are being varied by the MCMC
        Test: Boolean to inform the function that it is currently being tested so that it can give outputs
            in a way that a user can understand where issues may be occuring
    """
    for i in range(len(ChangingParamsKeys)):
        if i == 0:
            sqlSearch = 'SELECT ModelID FROM main."savedModels" WHERE "' + ChangingParamsKeys[i] + '" = ' + str(
                ChangingParamsValues[i])
        else:
            sqlSearch = sqlSearch + ' AND "' + ChangingParamsKeys[i] + '" = ' + str(ChangingParamsValues[i])
    if "midDens" in ParameterDict.keys():
        sqlSearch = sqlSearch + ' AND "phase" = 4'
    sqlSearch = sqlSearch + ';'
    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)
    mcf.AttemptSteps.value += 1
    if len(SearchResults) < 1 or SearchResults.values[0, 0] == None:
        if "finalOnly" in ParameterDict:
            ParameterDict["finalOnly"] = str(bool(ParameterDict["finalOnly"][0]))
        if len(SearchResults) < 1:
            ParamDF, ChemDF = UCLChemDataFrames(UCLChemDict={k: ParameterDict[k] for k in ParameterDict.keys() -
                                                             RadiativeCode_Keys})
            # Make changes for GreLVG
            UCLParamOut = RadexForGrid(ChemDict=ParameterDict, ParamDF=ParamDF,
                                       ChemDF=ChemDF)
            if Test:
                return UCLParamOut
            if "midDens" in ParamDF.columns:
                UCLParamOut["phase"] = UCLParamOut["phase"].apply(lambda x: 4)
            mcf.SQLQueue.put(("Save", [UCLParamOut, ParameterDict, CurrentPID]))
            CompleteCheck = RetrieveQueueResults(CurrentPID)
            del CompleteCheck
            mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID, True]))
            SearchResults = RetrieveQueueResults(CurrentPID)
    else:
        mcf.Counts.value += 1
    return SearchResults.values.T[0]


def PDSearchFunction(Search, PID, PostSave=False):
    """
    Function for the SQL managing worker to search the SQL database

    Args:
        Search: String of SQL query that will be used to query the "Search" database in memory
        PID: Process ID of the worker that put this search into the queue
        PostSave: Boolean to tell the code it should search the "Store" database instead of the "Search" database
            as the model may be in the "Store" database, and not yet in the "Search" database
    """
    if PostSave:
        Results = pd.read_sql(sql=Search, con=Store_con)
    else:
        Results = pd.read_sql(sql=Search, con=Search_con)
    if Results.empty:
        for col in Results.columns:
            if col != "ModelID":
                Results[col].values[:] = np.nan
    mcf.SQLResultDict[str(PID)] = Results
    return None


def FlushStore(PID):
    """
    Store all of the models in the "Store" databse, into the database on the hard drive of the user

    Args:
        PID: Process ID of the worker that put this job into the queue
    """
    DiskDatabase = sql.connect(DBLocation)
    DiskDatabase_cur = DiskDatabase.cursor()
    StartingID = DiskDatabase_cur.execute('SELECT Max(ModelID) FROM main."savedModels";').fetchone()[0]
    if StartingID == None:
        StartingID = 0
    DiskDatabase_cur.execute('SELECT name FROM main.sqlite_master WHERE type="table";')
    Tables_Temp = DiskDatabase_cur.fetchall()
    Search_cur.execute('SELECT name FROM main.sqlite_master WHERE type="table";')
    SearchTables_Temp = Search_cur.fetchall()
    SearchTables = [SearchTables_Temp[x][0] for x in range(len(SearchTables_Temp))]
    Store_cur.execute('ATTACH DATABASE "' + DBLocation + '" AS DiskDatabase;')
    Tables = [Tables_Temp[x][0] for x in range(len(Tables_Temp))]
    for table in Tables:
        if table != 'sqlite_sequence':
            Search = 'SELECT * FROM main."' + table + '";'
            TempDF = pd.read_sql(Search, Store_con)
            TempDF["ModelID"] = TempDF["ModelID"] + StartingID
            TempDF.to_sql(table, DiskDatabase, if_exists='append', index=False)
            DiskDatabase.commit()
            if table in SearchTables:
                UpdateSearchDatabase(table, StartingID)
            del TempDF, Search
            Store_cur.execute('DELETE FROM main."' + table + '" WHERE ModelID >= 0;')
            Store_con.commit()
    Store_cur.execute('DETACH DATABASE DiskDatabase;')
    DiskDatabase.commit()
    DiskDatabase.close()
    mcf.SQLResultDict[str(PID)] = "Complete"
    return None


def createStartingPoints(LinesArray, ChangingParameterKeys, DistanceMethod, Walkers, SpreadPercent=0.9):
    """
    This creates informed starting points for the MCMC walker according to a Distance method using a top hat function

    Args:
        LinesArray: Array containing the lines of interest that the user gave for the inference
        ChangingParameterKeys: Keys of the dictionary containing the parameters that will change during the inference
        DistanceMethod: Name of the function to use for determining which models to include in the Informed startin
            positions
        Walkers: number of walkers that will be used
        SpreadPercent: The percentage value to use for determining what values of emission lines would be acceptable
            to use
    """
    ID = retrieveClosestMatches(LinesArray, DistanceMethod)
    sqlSearch = 'SELECT ' + (
        str(ChangingParameterKeys)[1:-1].replace('\'', '"')) + ' FROM savedModels WHERE ModelID in (' + str(
        ID.tolist())[1:-1] + ');'
    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)
    SearchResults = SearchResults.values
    MeanValues = SearchResults.mean(axis=0)
    StartingPoints = np.zeros((Walkers, np.shape(MeanValues)[0]))
    for i in range(Walkers):
        for j in range(len(ChangingParameterKeys)):
            if 1 - SpreadPercent <= 0:
                low = 0.001
            else:
                low = 1 - SpreadPercent
            StartingPoints[i, j] = MeanValues[j] * np.random.uniform(low=low, high=1 + SpreadPercent)
    return StartingPoints


def createGausStartingPointsGrid(LinesArray, ChangingParameterKeys, Walkers, GridDictionary,
                                 SpreadPercent=0.9, DistanceMethod=DistancePercent):
    """
    This creates informed starting points for the MCMC walker according to a Distance method using a top hat function

    Args:
        LinesArray: Array containing the lines of interest that the user gave for the inference
        ChangingParameterKeys: Keys of the dictionary containing the parameters that will change during the inference
        DistanceMethod: Name of the function to use for determining which models to include in the Informed startin
            positions
        Walkers: number of walkers that will be used
        SpreadPercent: The percentage value to use for determining what values of emission lines would be acceptable
            to use
    """
    ID = retrieveClosestMatches(LinesArray, DistanceMethod)
    sqlSearch = 'SELECT ' + (
        str(ChangingParameterKeys)[1:-1].replace('\'', '"')) + ' FROM savedModels WHERE ModelID in (' + str(
        ID.tolist())[1:-1] + ');'

    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)

    SearchResults = SearchResults.values
    MeanValues = SearchResults.mean(axis=0)
    N = SearchResults.shape[0]
    StartingPoints = np.zeros((Walkers, np.shape(MeanValues)[0]))
    for j in range(len(ChangingParameterKeys)):
        sigma = np.sqrt((1 / N) * sum((SearchResults[:, j] - MeanValues[j]) ** 2))
        for i in range(Walkers):
            StartingPoints[i, j] = np.random.normal(loc=MeanValues[j], scale=sigma * 3)
            if StartingPoints[i, j] <= 0.000000000000000000000000000000000001:
                StartingPoints[i, j] = MeanValues[j]
            StartingPoints[i, j] = GridDictionary[ChangingParameterKeys[j]][
                (np.abs(GridDictionary[ChangingParameterKeys[j]] - StartingPoints[i, j])).argmin()]
    return StartingPoints


#############################################################################################################



#############################################################################################################
# The suppress_stdout_stderr function suppresses the output of any fortran code that is run when it is used #
# The addition of this code serves only to allow for cleaner terminal use of the code and is not necessary  #
#############################################################################################################
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


#############################################################################################################
# The following classes and function allow for No-Daemon Process parallel workers and are intended only     #
# as a workaround for the limitations posed by the multiprocessing package available in python.             #
#############################################################################################################
BoolDict = {}
BoolDict["Running"] = True


def StopManager():
    """
    Stops the managers of the different queues
    """
    BoolDict["Running"] = False


def worker_main():
    """
    Worker function that controls the SQL database functions
    """
    FuncDict = {"Search": PDSearchFunction, "Save": saveModel, "Flush": FlushStore, "Stop": StopManager}
    while BoolDict["Running"]:
        f, args = mcf.SQLQueue.get()
        FuncDict[f](*args)
    print("Worker main ended.")
    BoolDict["Running"] = True
    return None


def RetrieveQueueResults(CurrentProcess):
    """
    Worker function that asks for the results of the SQL database workers function calls
    """
    SQLResultDict = mcf.SQLResultDict.keys()
    while str(CurrentProcess) not in SQLResultDict:
        SQLResultDict = mcf.SQLResultDict.keys()
    Results = mcf.SQLResultDict[str(CurrentProcess)]
    del mcf.SQLResultDict[str(CurrentProcess)]
    return Results


def worker_Fortran():
    """
    Worker function that controls the Fortran functions
    """
    FuncDict = {"RunUCLCHEM": FortranUCLCHEM, "RunRADEX": FortranRADEX, "Stop": StopManager}
    while BoolDict["Running"]:
        f, args = mcf.FortranQueue.get()
        FuncDict[f](*args)
    print("Worker Fortran ended.")
    BoolDict["Running"] = True
    return None


def RetrieveFortranQueueResults(CurrentProcess):
    """
    Worker function that asks for the results of the Fortran function calls
    """
    FortranResultKeys = mcf.FortranResultDict.keys()
    while str(CurrentProcess) not in FortranResultKeys:
        FortranResultKeys = mcf.FortranResultDict.keys()
    Results = mcf.FortranResultDict[str(CurrentProcess)]
    del mcf.FortranResultDict[str(CurrentProcess)]
    return Results

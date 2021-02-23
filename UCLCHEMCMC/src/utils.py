import os
import SpectralRadex.radex as radex
import numpy as np
import pandas as pd
import sqlite3 as sql
import time
import MCFunctions as mcf
os.environ["OMP_NUM_THREADS"] = "1"
from uclchem import wrap as uclchem

sleepTime = 1
timeOutForSQL = 1000
savedModelsColumns = ['phase', 'switch', 'collapse', 'readAbunds', 'writeStep', 'points', 'outSpecies', 'desorb',
                      'initialDens', 'finalDens', 'initialTemp', 'maxTemp', 'zeta', 'radfield', 'rin', 'rout', 'fr',
                      'ageOfCloudOut', 'h2densOut', 'cloudTempOut', 'avOut', 'radiationOut', 'zetaOut', 'h2FormRateOut',
                      'fcOut', 'foOut', 'fmgOut', 'fheOut', 'deptOut', 'finalOnly', 'H', 'H+']

DBLocation = "../data/Database.db"
ChemFile = "../data/Chemicals.csv"
Chem = pd.read_csv(ChemFile, delimiter=",/", engine='python')
Lines = Chem.set_index('Chemical')["Lines"].to_dict()
for C in Lines.keys():
    Lines[C] = np.asarray(Lines[C].split(", ")).astype(str)


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


outSpeciesList = list(Lines.keys())
outSpeciesString = " ".join(list(Lines.keys()))
UniqueOutSpecies = np.unique(np.asarray([chemForUCLCHEM(x) for x in outSpeciesList])).tolist()
UniqueOutSpeciesString = 'H H+ ' + ' '.join(UniqueOutSpecies)
UniqueOutSpecieslist = ['H', 'H+'] + UniqueOutSpecies

def DistancePercent(BaseFlux):
    Lower = float(BaseFlux) * 0.8
    Upper = float(BaseFlux) * 1.2
    return Lower, Upper


def UCLChemDataFrames(UCLChemDict, Test=False, Queue=True):
    stepCount = np.zeros(shape=1, dtype=np.int32, order='F')
    UCLChemDict["outSpecies"] = len(UniqueOutSpecieslist)
    if Queue:
        CurrentPID = os.getpid()
        mcf.FortranQueue.put(("RunUCLCHEM", [UCLChemDict, UCLChemDict['points'], CurrentPID, stepCount]))
        DictOfArrays = RetrieveFortranQueueResults(CurrentPID)
        parameterArray = DictOfArrays[0]
        chemicalAbunArray = DictOfArrays[1]
    else:
        parameterArray = np.zeros(shape=(10000, UCLChemDict['points'], 12), dtype=np.float64, order='F')
        chemicalAbunArray = np.zeros(shape=(10000, UCLChemDict['points'], 215), dtype=np.float64, order='F')
        uclchem.to_df(dictionary=UCLChemDict, outspeciesin=UniqueOutSpeciesString, numberpoints=UCLChemDict['points'],
                      parameterarray=parameterArray, chemicalabunarray=chemicalAbunArray, stepcount=stepCount)
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
        physDF = pd.DataFrame(data=parameterArray[:, 0, :], columns=['ageOfCloudOut', 'h2densOut', 'cloudTempOut',
                                                                     'avOut', 'radiationOut', 'zetaOut',
                                                                     'h2FormRateOut', 'fcOut',
                                                                     'foOut', 'fmgOut', 'fheOut', 'deptOut'])
        physDF['H'] = chemicalAbunArray[:stepCount[0], 0, 0]
        physDF['H+'] = chemicalAbunArray[:stepCount[0], 0, 1]
        chemDF = pd.DataFrame(data=chemicalAbunArray[:, 0, 2:], columns=UniqueOutSpecieslist[2:])
    else:
        exit("Currently unable to run with multiple Points")
    return physDF, chemDF


def FortranUCLCHEM(dictionary, numberpoints, CurrentPID, stepCount):
    parameterArray = np.zeros(shape=(10000, dictionary['points'], 12), dtype=np.float64, order='F')
    chemicalAbunArray = np.zeros(shape=(10000, dictionary['points'], 215), dtype=np.float64, order='F')
    with suppress_stdout_stderr():
        uclchem.to_df(dictionary=dictionary, outspeciesin=UniqueOutSpeciesString, numberpoints=numberpoints,
                      parameterarray=parameterArray, chemicalabunarray=chemicalAbunArray, stepcount=stepCount)
    time.sleep(sleepTime)
    mcf.FortranResultDict[str(CurrentPID)] = [parameterArray, chemicalAbunArray]
    return None


def RadexForGrid(UCLChemDict, UCLParamDF, UCLChemDF):
    for k in range(np.shape(UCLParamDF)[0]):
        radexDic = radex.get_default_parameters()
        radexDic['tkin'] = UCLParamDF.loc[k, 'cloudTempOut']
        radexDic['h2'] = UCLParamDF.loc[k, 'h2densOut']
        radexDic['h'] = UCLParamDF.loc[k, 'H']
        radexDic['h+'] = UCLParamDF.loc[k, 'H+']
        radexDic['e-'] = UCLParamDF.loc[k, 'H+']
        radexDic['fmax'] = 1000.0
        for i in UCLChemDF.columns:
            radexDic['cdmol'] = UCLChemDF.loc[k, i] * 1.6e21 * UCLParamDF.loc[k, 'avOut']
            if 1e5 <= radexDic['cdmol'] <= 1e25:
                for j in ReverseChemForUCLCHEM(i).split(", "):
                    radexDic['molfile'] = chemDat(j)
                    #try:
                    RadexDF = runRadex(RadexParamDict=radexDic)
                    if RadexDF is np.nan:
                        continue
                    UCLParamDF = RadexToDF(UCLChemDict, UCLParamDF, RadexDF, k)
                    #except:
                    #    pass
            else:
                pass
    return UCLParamDF


def RadexToDF(UCLChemDict, UCLParamDF, RadexArray, indexOfModel):
    UCLChemDictKeys = [k for k in UCLChemDict]
    UCLChemDictValues = [v for v in UCLChemDict.values()]
    for l in range(len(UCLChemDictKeys)):
        UCLParamDF[UCLChemDictKeys[l]] = UCLChemDictValues[l]
    for j in range(np.shape(RadexArray)[0]):
        T_rColumnName = RadexArray[j, 0] + '_T_r'
        if T_rColumnName not in UCLParamDF.columns:
            UCLParamDF[T_rColumnName] = np.nan
        intColumnName = RadexArray[j, 0] + '_Intensity'
        if intColumnName not in UCLParamDF.columns:
            UCLParamDF[intColumnName] = np.nan
        fluColumnName = RadexArray[j, 0] + '_Flux'
        if fluColumnName not in UCLParamDF.columns:
            UCLParamDF[fluColumnName] = np.nan
        UCLParamDF[T_rColumnName].iloc[indexOfModel] = RadexArray[j, 1]
        UCLParamDF[intColumnName].iloc[indexOfModel] = RadexArray[j, 2]
        UCLParamDF[fluColumnName].iloc[indexOfModel] = RadexArray[j, 3]
    return UCLParamDF


def runRadex(RadexParamDict, Queue=True):
    CurrentPID = os.getpid()
    if Queue:
        mcf.FortranQueue.put(("RunRADEX", [RadexParamDict, CurrentPID]))
        dataFrame = RetrieveFortranQueueResults(CurrentPID)
    else:
        dataFrame = radex.run(parameters=RadexParamDict)
    if dataFrame.empty:
        return np.nan
    OutputList = []
    if "_" in dataFrame["QN Upper"].iloc[0] or " " in dataFrame["QN Upper"].iloc[0]:
        for i in range(np.shape(dataFrame)[0]):
            TransitionName = RadexParamDict["molfile"][:-4] + "_(" + \
                             dataFrame["QN Upper"].iloc[i].replace("_", ",").replace(" ", ",").replace(",,", ",") + \
                             ")-(" + \
                             dataFrame["QN Lower"].iloc[i].replace("_", ",").replace(" ", ",").replace(",,", ",") + \
                             ")(" + \
                             str(dataFrame["freq"].iloc[i]) + \
                             " GHz)"
            # Name, T_r, Intensity, Flux
            OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i],
                           dataFrame["FLUX (K*km/s)"].iloc[i], dataFrame["FLUX (erg/cm2/s)"].iloc[i]]]
    else:
        for i in range(np.shape(dataFrame)[0]):
            TransitionName = RadexParamDict["molfile"][:-4] + "_" + \
                             dataFrame["QN Upper"].iloc[i].replace("_", ",") + "-" + \
                             dataFrame["QN Lower"].iloc[i].replace("_", ",") + "(" + \
                             str(dataFrame["freq"].iloc[i]) + \
                             " GHz)"
            # Name, T_r, Intensity, Flux
            OutputList += [[TransitionName, dataFrame["T_R (K)"].iloc[i],
                            dataFrame["FLUX (K*km/s)"].iloc[i], dataFrame["FLUX (erg/cm2/s)"].iloc[i]]]
    OutputArray = np.asarray(OutputList)
    return OutputArray


def FortranRADEX(RadexParamDict, CurrentPID):
    with suppress_stdout_stderr():
        dataFrame = radex.run(parameters=RadexParamDict)
    mcf.FortranResultDict[str(CurrentPID)] = dataFrame
    return None


def saveModel(UCLParamDF, ParamDict, CurrentPID):
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
    con = sql.connect(DBLocation, timeout=timeOutForSQL)
    cur = con.cursor()
    RedundantCheckSearch = ''
    for i in range(len(UCLParamDF.columns)):
        if i == 0:
            if type(UCLParamDF[UCLParamDF.columns[i]].iloc[0]) == str:
                RedundantCheckSearch = 'SELECT ModelID FROM savedModels WHERE "' + \
                                       UCLParamDF.columns[i] + '" == "' + \
                                       UCLParamDF[UCLParamDF.columns[i]].iloc[0] + '"'
            else:
                RedundantCheckSearch = 'SELECT ModelID FROM savedModels WHERE "' + \
                                       UCLParamDF.columns[i] + '" == ' + \
                                       str(UCLParamDF[UCLParamDF.columns[i]].iloc[0])
        else:
            if type(UCLParamDF[UCLParamDF.columns[i]].iloc[0]) == str:
                RedundantCheckSearch += ' AND "' + UCLParamDF.columns[i] + '" == "' + \
                                       UCLParamDF[UCLParamDF.columns[i]].iloc[0] + '"'
            else:
                RedundantCheckSearch += ' AND "' + UCLParamDF.columns[i] + '" == ' + \
                                       str(UCLParamDF[UCLParamDF.columns[i]].iloc[0])
    RedundantCheckSearch += ';'
    cur.execute(RedundantCheckSearch)
    RedundantResults = cur.fetchone()
    if type(RedundantResults) == list:
        con.close()
        return None
    cur.execute('SELECT * FROM savedModels;')
    UCLParamDF.to_sql(name='savedModels', if_exists='append', con=con, index=False)
    cur.execute('SELECT * FROM savedModels;')
    ThisEntryID = (cur.fetchall()[-1][0])
    cur.execute("SELECT tbl_name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
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
                cur.execute(command)
                con.commit()
            cur.execute('SELECT * FROM "' + datFile + '";')
            columnsInSQL = [description[0] for description in cur.description]
            for Column in list(set(columnsOfInterest) - set(columnsInSQL)):
                command = 'ALTER TABLE "' + datFile + '" ADD "' + Column + '" REAL;'
                cur.execute(command)
                con.commit()
            TempValueArray = ', '.join([str(x) for x in UCLCHEMTransitionDFTemp.to_numpy()[0].tolist()])
            TempValueArray = TempValueArray.replace("nan", "0")
            TempColumnsList = '", "'.join(UCLCHEMTransitionDFTemp.columns.to_list())
            if TempColumnsList == "":
                continue
            command = 'INSERT INTO "' + datFile + '"("' + TempColumnsList +'", ModelID) VALUES (' + \
                      str(TempValueArray) + ', ' + str(ThisEntryID) + ');'
            cur.execute(command)
            con.commit()
        elif len(columnsOfInterest) > 1000:
            columnsLeft = columnsOfInterest
            itterations = int(len(columnsOfInterest) / 1000)
            if len(columnsOfInterest) > ((itterations+1) * 1000):
                itterations += 1
            for k in range(itterations):
                currentDat = datFile + '_' + str(k)
                columnsThisLoop = columnsLeft[:1000]
                columnsLeft = columnsLeft[1000:]
                UCLCHEMTransitionDFTemp = UCLCHEMTransitionDF[columnsThisLoop]
                if currentDat not in TablesList:
                    command = 'CREATE TABLE "' + currentDat + \
                              '" (ModelID INTEGER PRIMARY KEY AUTOINCREMENT, FOREIGN KEY(ModelID) REFERENCES savedModels (ModelID));'
                    cur.execute(command)
                    con.commit()
                cur.execute('SELECT * FROM "' + currentDat + '";')
                columnsInSQL = [description[0] for description in cur.description]
                for Column in list(set(columnsThisLoop) - set(columnsInSQL)):
                    command = 'ALTER TABLE "' + currentDat + '" ADD "' + Column + '" REAL;'
                    cur.execute(command)
                    con.commit()
                TempValueArray = ', '.join([str(x) for x in UCLCHEMTransitionDFTemp.to_numpy()[0].tolist()])
                TempValueArray = TempValueArray.replace("nan", "0")
                TempColumnsList = '", "'.join(UCLCHEMTransitionDFTemp.columns.to_list())
                if TempColumnsList == "":
                    continue
                command = 'INSERT INTO "' + currentDat + '"("' + TempColumnsList +'", ModelID) VALUES (' + \
                      str(TempValueArray) + ', ' + str(ThisEntryID) + ');'
                cur.execute(command)
                con.commit()
    con.close()
    mcf.SQLResultDict[str(CurrentPID)] = "Complete"
    return None


def retrieveClosestMatches(LinesArray, DistanceMethod):
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
    sqlSearch = None
    FromString = ""
    PreviousDat = ""
    WhereString = ""
    for dats in ChemsAndLinesDic.keys():
        for i in range(len(ChemsAndLinesDic[dats])):
            if sqlSearch == None:
                sqlSearch = 'SELECT "' + dats + '"."' + ChemsAndLinesDic[dats][i] + '"'
                WhereString = ' WHERE "' + dats + '".ModelID == ' + str(ModelID[0]) + ';'
            else:
                sqlSearch = sqlSearch + ', "' + dats + '"."' + ChemsAndLinesDic[dats][i] + '"'
        if FromString == "":
            FromString = ' FROM "' + dats + '"'
            PreviousDat = dats
        else:
            FromString += ' LEFT JOIN "' + dats + '" ON "' + PreviousDat + '".ModelID = "' + dats + '".ModelID'
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
        Intensities = np.nan
    else:
        Intensities = SearchResults.iloc[0].values
        for i in range(len(Intensities)):
            if type(Intensities[i]) == str:
                Intensities[i] = np.nan
    return Intensities


def retrieveEntry(ID, ModelParameters=False):
    if ModelParameters:
        sqlSearch = 'SELECT ModelID, phase, switch, collapse, readAbunds, writeStep, points, outSpecies, desorb, ' \
                    'initialDens, finalDens, initialTemp, maxTemp, zeta, radfield, rin, rout, fr, finalOnly' \
                    ' FROM savedModels WHERE ModelID in ' + '(' + str(ID.tolist())[1:-1] + ');'
        CurrentPID = os.getpid()
        mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
        SearchResults = RetrieveQueueResults(CurrentPID)
    else:
        sqlSearch = 'SELECT ModelID, ageOfCloudOut, h2densOut, cloudTempOut, avOut, radiationOut, zetaOut, ' \
                    '"h2FormRateOut", fcOut, foOut, fmgOut, fheOut, deptOut FROM savedModels WHERE ModelID in (' + \
                    str(ID.tolist())[1:-1] + ');'
        CurrentPID = os.getpid()
        mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
        SearchResults = RetrieveQueueResults(CurrentPID)
    return SearchResults


def retrieveIntensitiesUI(ParameterDict, ChangingParams, LinesOfInterest, Chemicals, LinesGiven, Test=False):
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
                if chemDat(UniqueChems[i])[:-4] in ChemsAndLines.keys() and chemDat(UniqueChems[i])[:-4] in LinesOfInterest[j]:
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
    for i in range(len(ChangingParamsKeys)):
        if i == 0:
            sqlSearch = 'SELECT ModelID FROM savedModels WHERE "' + ChangingParamsKeys[i] + '" == ' + str(
                ChangingParamsValues[i])
        else:
            sqlSearch = sqlSearch + ' AND "' + ChangingParamsKeys[i] + '" == ' + str(ChangingParamsValues[i])
    sqlSearch = sqlSearch + ';'
    CurrentPID = os.getpid()
    mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
    SearchResults = RetrieveQueueResults(CurrentPID)
    if len(SearchResults) < 1 or SearchResults.values[0, 0] != None:
        if "finalOnly" in ParameterDict:
            ParameterDict["finalOnly"] = str(bool(ParameterDict["finalOnly"][0]))
        if len(SearchResults) < 1:
            ParamDF, ChemDF = UCLChemDataFrames(UCLChemDict=ParameterDict)
            UCLParamOut = RadexForGrid(UCLChemDict=ParameterDict, UCLParamDF=ParamDF,
                                       UCLChemDF=ChemDF)
            if Test:
                return UCLParamOut
            mcf.SQLQueue.put(("Save", [UCLParamOut, ParameterDict, CurrentPID]))
            CompleteCheck = RetrieveQueueResults(CurrentPID)
            del CompleteCheck
            mcf.SQLQueue.put(("Search", [sqlSearch, CurrentPID]))
            SearchResults = RetrieveQueueResults(CurrentPID)
    return SearchResults.values.T[0]


def PDSearchFunction(Search, PID):
    con = sql.connect(DBLocation, timeout=timeOutForSQL)
    Results = pd.read_sql(sql=Search, con=con)
    con.close()
    if Results.empty:
        for col in Results.columns:
            if col != "ModelID":
                Results[col].values[:] = np.nan
    mcf.SQLResultDict[str(PID)] = Results
    return None


def createStartingPoints(LinesArray, ChangingParameterKeys, DistanceMethod, Walkers, SpreadPercent=0.9):
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


def createGausStartingPoints(LinesArray, ChangingParameterKeys, DistanceMethod, Walkers, SpreadPercent=0.9):
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
    return StartingPoints


def createGausStartingPointsGrid(LinesArray, ChangingParameterKeys, Walkers, GridDictionary,
                                 SpreadPercent=0.9, DistanceMethod=DistancePercent):
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
        'SO': 'SO'}
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
        'SO': 'so@lique.dat'}
    return switcher.get(i, i + " is an invalid chemical given to chemDat")


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
    BoolDict["Running"] = False


def worker_main():
    FuncDict = {"Search": PDSearchFunction, "Save": saveModel, "Stop": StopManager}
    while BoolDict["Running"]:
        f, args = mcf.SQLQueue.get()
        FuncDict[f](*args)
    print("Worker main ended.")
    BoolDict["Running"] = True
    return None


def RetrieveQueueResults(CurrentProcess):
    SQLResultDict = mcf.SQLResultDict.keys()
    while str(CurrentProcess) not in SQLResultDict:
        SQLResultDict = mcf.SQLResultDict.keys()
    Results = mcf.SQLResultDict[str(CurrentProcess)]
    del mcf.SQLResultDict[str(CurrentProcess)]
    return Results


def worker_Fortran():
    FuncDict = {"RunUCLCHEM": FortranUCLCHEM, "RunRADEX": FortranRADEX, "Stop": StopManager}
    while BoolDict["Running"]:
        f, args = mcf.FortranQueue.get()
        FuncDict[f](*args)
    print("Worker Fortran ended.")
    BoolDict["Running"] = True
    return None


def RetrieveFortranQueueResults(CurrentProcess):
    FortranResultKeys = mcf.FortranResultDict.keys()
    while str(CurrentProcess) not in FortranResultKeys:
        FortranResultKeys = mcf.FortranResultDict.keys()
    Results = mcf.FortranResultDict[str(CurrentProcess)]
    del mcf.FortranResultDict[str(CurrentProcess)]
    return Results

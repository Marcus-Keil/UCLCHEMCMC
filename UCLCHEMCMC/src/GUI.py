#!/usr/bin/env python
import os
import shelve
import numpy as np
import pandas as pd
import billiard.pool as BilPool
from celery import Celery
import sqlite3 as sql
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.embed import autoload_static
from flask import Flask, request, render_template, session, redirect, \
    url_for, jsonify
import json
import calendar
import time

os.environ["OMP_NUM_THREADS"] = "1"

import utils as utils
import MCFunctions as mcf

# Celery and Flask Initiations
# =========================================================================================================
app = Flask(__name__)
app.secret_key = "jjdd"

# Things marked with these lines are examples for Celery
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
# =========================================================================================================

# Parameter declarations, for now hard coded, will be exported to a file that can be changed
# at the end.
# =========================================================================================================

CodeName = 'UCLCHEMCMC'

ChemFile = "../data/Chemicals.csv"
Chem = pd.read_csv(ChemFile, delimiter=",/", engine='python')
Lines = Chem.set_index('Chemical')["Lines"].to_dict()
for C in Lines.keys():
    Lines[C] = np.asarray(Lines[C].split(", ")).astype(str)
Chemicals = list(Lines.keys())
Chemicals += ['']


# The physical parameters that are allowed to be changed
with open("../data/UCLCHEM_ChangeableParameterNames.json") as jFile:
    ChemicalCode_Parameters = json.load(jFile)

# The parameters currently not allowd to be changed, but are needed
with open("../data/UCLCHEM_UnchangableParameters.json") as jFile:
    ChemicalCode_UnchangableParameters = json.load(jFile)

with open("../data/RADEX_ChangeableParameterNames.json") as jFile:
    RadiativeCode_Parameters = json.load(jFile)

with open("../data/UCLCHEMOnline_UnchangableParameters.json") as jFile:
    Online_UCLCHEMParameters = json.load(jFile)

with open("../data/UCLCHEMOnline_ChangeableParameterDefaults.json") as jFile:
    Online_UCLCHEMDefaults = json.load(jFile)

Switches = ["switch", "collapse", "desorb"]
# Default parameter values if nothing is specified
ParameterDefaults = {'finalDens': 1.0e5,
                     'initialTemp': 10,
                     'zeta': 1,
                     'radfield': 1,
                     'rout': 0.05}

# MCMC Parameters as of now

#
MCMCwalkers = 1
MCMCStepsPerRun = 200
MCMCStepsPerItteration = 100
ManagerPoolSize = 1
FortranPoolSize = 1

MCMCNumberProcesses = os.cpu_count() - 6
if MCMCNumberProcesses <= 1:
    MCMCNumberProcesses = 2
BaseUIFolder = "./"
ResultsFolder = "../results/"
SaveFolder = "../saves/"

Counts = 0

def CountAdd():
    global Counts
    Counts += 1


def CountRet():
    global Counts
    return Counts

# Currently unused parameters
# 'phase', 'switch', 'collapse', 'desorb', 'initialDens', 'initialDens_up',
# 'maxTemp', 'maxTemp_up',
# =========================================================================================================

# The celery task, wip.
# =========================================================================================================
@celery.task(bind=True)
def RunMCMC(self, BaseDict, ChangingParamList, ChangingDictRanges, PDLinesJson,
            MCMCFile, GridDictionary, Informed=False, Walkers=MCMCwalkers,
            StepsPerClick=MCMCStepsPerRun, StepsPerSave=MCMCStepsPerItteration, startPoints=[]):
    """

    Args:

    """
    ParameterRanges = {}
    for P in GridDictionary.keys():
        Array = np.array(GridDictionary[P]).astype(np.float)
        ParameterRanges[P + "_low"] = min(Array)
        ParameterRanges[P + "_high"] = max(Array)
        GridDictionary[P] = Array
        del Array
    BaseDict["outSpecies"] = len(utils.UniqueOutSpecieslist)
    UserRangesDict = {}
    for ind, RangeLim in enumerate(ChangingParamList):
        low = ChangingDictRanges[RangeLim + "_low"]
        lowGrid = (np.abs(GridDictionary[RangeLim] - low)).argmin()
        if lowGrid != 0:
            lowGrid -= 1
        UserRangesDict[RangeLim + "_low"] = lowGrid
        up = ChangingDictRanges[RangeLim + "_up"]
        upGrid = (np.abs(GridDictionary[RangeLim] - up)).argmin()
        if upGrid < len(GridDictionary[RangeLim]) - 1:
            upGrid += 1
        UserRangesDict[RangeLim + "_up"] = upGrid
    toDel = []
    for P in UserRangesDict.keys():
        if "_low" in P:
            if UserRangesDict[P] == UserRangesDict[P[:-4] + "_up"]:
                toDel += [P[:-4]]
    for i in toDel:
        del UserRangesDict[i + "_low"]
        del ChangingDictRanges[i + "_low"]
        del UserRangesDict[i + "_up"]
        del ChangingDictRanges[i + "_up"]
        ChangingParamList.remove(i)
    utils.InitDatabases(BaseDict, ChangingParamList, ChangingDictRanges, PDLinesJson)
    iterations = int(StepsPerClick / StepsPerSave)
    ManagerPool = BilPool.Pool(ManagerPoolSize, utils.worker_main)
    FortranPool = BilPool.Pool(FortranPoolSize, utils.worker_Fortran)
    for i in range(iterations):
        Start = time.time()
        ndim = int(len(ChangingParamList))
        sampler = mcf.MCMCSavesGridUI(MCMCFile, MCMCNumberProcesses, ndim, Walkers,
                                      StepsPerSave, BaseDict, ChangingParamList, startPoints,
                                      GridDictionary, ParameterRanges, PDLinesJson,
                                      UserRangesDict)
        self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': iterations,
                                                  'status': "completed " + str(
                                                      (i + 1) * StepsPerSave) + " of " + str(
                                                      StepsPerClick) + " steps"})  # ,'figureJS': js, 'figureTag': tag})
        CurrentPID = os.getpid()
        mcf.SQLQueue.put(("Flush", [CurrentPID]))
        with open("Text.txt", "a") as myfile:
            myfile.write(
                str(time.time() - Start) + ", " + str(mcf.Counts.value) + ", " + str(mcf.AttemptSteps.value) + "\n")
        mcf.AttemptSteps.value = 0
        mcf.Counts.value = 0
        CompleteCheck = utils.RetrieveQueueResults(CurrentPID)
    for i in range(FortranPoolSize):
        mcf.FortranQueue.put(("Stop", []))
        time.sleep(2)
    mcf.SQLQueue.put(("Stop", []))
    FortranPool.close()
    FortranPool.join()
    ManagerPool.close()
    ManagerPool.join()
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    return {'current': iterations, 'total': iterations, 'status': 'Task completed!',
            'result': "GridWebUI-" + str(len(sampler.get_chain()))}
# =========================================================================================================


# The celery task, wip.
# =========================================================================================================
@celery.task(bind=True)
def RunUCLCHEMAlone(self, UCLCHEMDict):
    """

    Args:

    """
    UCLCHEMDict["phase"] = int(1)
    UCLCHEMDict["switch"] = int(UCLCHEMDict["switch"])
    UCLCHEMDict["collapse"] = int(UCLCHEMDict["collapse"])
    UCLCHEMDict["desorb"] = int(UCLCHEMDict["desorb"])
    UCLCHEMDict["readAbunds"] = int(0)
    UCLCHEMDict["writeStep"] = int(1)
    UCLCHEMDict["points"] = int(1)
    UCLCHEMDict["finalOnly"] = "True"
    UCLCHEMDict["outSpecies"] = len(utils.UniqueOutSpecieslist)
    ManagerPool = BilPool.Pool(ManagerPoolSize, utils.worker_main)
    FortranPool = BilPool.Pool(FortranPoolSize, utils.worker_Fortran)
    self.update_state(state='PROGRESS', meta={'current': 1, 'total': 2, 'status': "completed UCLCHEM, joining workers"})
    PhysDF, ChemDF = utils.UCLChemDataFrames(UCLCHEMDict)
    for i in range(FortranPoolSize):
        mcf.FortranQueue.put(("Stop", []))
        time.sleep(2)
    mcf.SQLQueue.put(("Stop", []))
    FortranPool.close()
    FortranPool.join()
    ManagerPool.close()
    ManagerPool.join()
    PhysDF.to_html('templates/UCLCHEMPhysResults.html')
    ChemDF.to_html('templates/UCLCHEMChemResults.html')
    self.update_state(state='COMPLETE', meta={'current': 1, 'total': 2, 'status': "Task complete!"})
    return {'current': 2, 'total': 2, 'status': 'Task completed!', 'result': 'UCLCHEMChemResults.html'}
# =========================================================================================================


# Home or Index Page
# =========================================================================================================
@app.route('/')
def Index():
    """

    Args:

    """
    session.clear()
    return render_template('index.html', name=CodeName)
# =========================================================================================================


# Initial MCMC page to give parameter ranges
# =========================================================================================================
@app.route('/MCMCInference/', methods=["POST", "GET"])
def MCMC():
    """

    Args:

    """
    if request.method == "POST":
        if 'Load_Session' in request.form and request.form['Session_name'] != '':
            SessionName = SaveFolder + request.form['Session_name']
            if os.path.isfile(SessionName + ".bak"):
                with shelve.open(SessionName) as db:
                    for key in db.keys():
                        session[key] = db[key]
                return redirect(url_for('Options'))
        if 'Add_Grid' in request.form and "Grid" not in session.keys():
            session["Grid"] = request.form["Grid"]
            GridFile = "../data/" + session["Grid"] + "_Grid.csv"
            Grid = pd.read_csv(GridFile, delimiter=",/", engine='python')
            GridDictionary = Grid.set_index('Parameter')["Grid"].to_dict()
            session["ParamRanges"] = {}
            for P in GridDictionary.keys():
                Array = np.array(GridDictionary[P].split(",")).astype(np.float)
                session["ParamRanges"][P + "_low"] = min(Array)
                session["ParamRanges"][P + "_high"] = max(Array)
                GridDictionary[P] = Array.tolist()
                del Array
            return redirect(url_for('MCMC'))
        ParameterRanges = session["ParamRanges"]
        Chem_parameter = 0
        while Chem_parameter < len(ChemicalCode_Parameters):
            if request.form[ChemicalCode_Parameters[Chem_parameter]] != '':
                lower = float(request.form[ChemicalCode_Parameters[Chem_parameter]])
                if ("_up" in ChemicalCode_Parameters[Chem_parameter] and ChemicalCode_Parameters[Chem_parameter-1] not in session[ChemicalCode_Parameters[Chem_parameter]]):
                    return render_template('MCMC.html', name=CodeName, Chemicals=Chemicals,
                                           Lines=Lines, session=session)
                if lower >= ParameterRanges[ChemicalCode_Parameters[Chem_parameter]+"_low"] and lower <= ParameterRanges[ChemicalCode_Parameters[Chem_parameter]+"_high"]:
                    session[ChemicalCode_Parameters[Chem_parameter]] = lower
                elif lower < ParameterRanges[ChemicalCode_Parameters[Chem_parameter]+"_low"]:
                    session[ChemicalCode_Parameters[Chem_parameter]] = ParameterRanges[ChemicalCode_Parameters[Chem_parameter] + "_low"]
                else:
                    session[ChemicalCode_Parameters[Chem_parameter]] = ParameterRanges[ChemicalCode_Parameters[Chem_parameter] + "_high"]
                if '_up' in ChemicalCode_Parameters[Chem_parameter+1]:
                    if request.form[ChemicalCode_Parameters[Chem_parameter+1]] != '':
                        upper = float(request.form[ChemicalCode_Parameters[Chem_parameter+1]])
                        if upper <= ParameterRanges[ChemicalCode_Parameters[Chem_parameter]+"_high"] and upper > session[ChemicalCode_Parameters[Chem_parameter]]:
                            session[ChemicalCode_Parameters[Chem_parameter + 1]] = upper
                        elif upper <= session[ChemicalCode_Parameters[Chem_parameter]]:
                            print("Upper bound was less than the lower number, keeping the parameter constant")
                        else:
                            session[ChemicalCode_Parameters[Chem_parameter + 1]] = ParameterRanges[ChemicalCode_Parameters[Chem_parameter]+"_high"]
                    elif ChemicalCode_Parameters[Chem_parameter+1] in session.keys():
                        session.pop(ChemicalCode_Parameters[Chem_parameter+1], None)
                    Chem_parameter += 1
            Chem_parameter += 1
        Rad_parameter = 0
        print("Got to RadWhile")
        while Rad_parameter < (len(RadiativeCode_Parameters)):
            if request.form[RadiativeCode_Parameters[Rad_parameter]] != '':
                lower = float(request.form[RadiativeCode_Parameters[Rad_parameter]])
                if ("_up" in RadiativeCode_Parameters[Rad_parameter] and RadiativeCode_Parameters[
                    Rad_parameter - 1] not in session[RadiativeCode_Parameters[Rad_parameter]]):
                    return render_template('MCMC.html', name=CodeName, Chemicals=Chemicals,
                                           Lines=Lines, session=session)
                if lower >= ParameterRanges[RadiativeCode_Parameters[Rad_parameter] + "_low"] and lower <= \
                        ParameterRanges[RadiativeCode_Parameters[Rad_parameter] + "_high"]:
                    session[RadiativeCode_Parameters[Rad_parameter]] = lower
                elif lower < ParameterRanges[RadiativeCode_Parameters[Rad_parameter] + "_low"]:
                    session[RadiativeCode_Parameters[Rad_parameter]] = ParameterRanges[
                        RadiativeCode_Parameters[Rad_parameter] + "_low"]
                else:
                    session[RadiativeCode_Parameters[Rad_parameter]] = ParameterRanges[
                        RadiativeCode_Parameters[Rad_parameter] + "_high"]
                if '_up' in RadiativeCode_Parameters[Rad_parameter + 1]:
                    if request.form[RadiativeCode_Parameters[Rad_parameter + 1]] != '':
                        upper = float(request.form[RadiativeCode_Parameters[Rad_parameter + 1]])
                        if upper <= ParameterRanges[RadiativeCode_Parameters[Rad_parameter] + "_high"] and upper > \
                                session[RadiativeCode_Parameters[Rad_parameter]]:
                            session[RadiativeCode_Parameters[Rad_parameter + 1]] = upper
                        elif upper <= session[RadiativeCode_Parameters[Rad_parameter]]:
                            print("Upper bound was less than the lower number, keeping the parameter constant")
                        else:
                            session[RadiativeCode_Parameters[Rad_parameter + 1]] = ParameterRanges[
                                RadiativeCode_Parameters[Rad_parameter] + "_high"]
                    elif RadiativeCode_Parameters[Rad_parameter + 1] in session.keys():
                        session.pop(RadiativeCode_Parameters[Rad_parameter + 1], None)
                    Rad_parameter += 1
            Rad_parameter += 1
        return redirect(url_for("MCMCPhys"))
    else:
        return render_template('MCMC.html', name=CodeName, Chemicals=Chemicals,
                                   Lines=Lines, session=session)
# =========================================================================================================


# Parameter range confirmation Page
# =========================================================================================================
@app.route('/MCMCInference/Phys', methods=["POST", "GET"])
def MCMCPhys():
    """

    Args:

    """
    if request.method == "POST":
        return redirect(url_for("MCMCChem"))
    else:
        return render_template('MCMC_Phys.html', name=CodeName, Chemicals=Chemicals,
                                   Lines=Lines, session=session)
# =========================================================================================================


# Chemical line adding page
# =========================================================================================================
@app.route('/MCMCInference/Chem', methods=["POST", "GET"])
def MCMCChem():
    """

    Args:

    """
    if "outSpecies" in session:
        NewChem = [Chem for Chem in Chemicals if Chem not in session["outSpecies"]]
    else:
        NewChem = Chemicals
        session["outSpecies"] = []

    if request.method == "POST":
        session["Error"] = ""
        if "Add_Mol" in request.form and request.form["AddSpecies"] != '':
            molecule = request.form["AddSpecies"]
            session["outSpecies"] += [molecule]
            NewChem = [Chem for Chem in Chemicals if Chem not in session["outSpecies"]]
            return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                   Lines=Lines, session=session)
        elif "Run_Inference" in request.form:
            for ParamDef in ParameterDefaults.keys():
                if ParamDef not in session.keys():
                    session[ParamDef] = ParameterDefaults[ParamDef]
            return redirect(url_for("Options"))
        elif "Add_Mol" not in request.form and "Run_Inference" not in request.form:
            for Mol in session["outSpecies"]:
                if Mol + "_Add_Line" in request.form and (Mol + "_Lines" in request.form):
                    session[Mol] = request.form.getlist(Mol + "_Lines")
                    return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                           Lines=Lines, session=session)
                elif Mol + "_Clear" in request.form:
                    Species = session.pop("outSpecies", None)
                    Species.remove(Mol)
                    session.pop(Mol, None)
                    topop = []
                    for key in session.keys():
                        if Mol in key:
                            topop += [key]
                    for key in topop:
                        session.pop(key, None)
                    session["outSpecies"] = Species
                    NewChem = [Chem for Chem in Chemicals if Chem not in session["outSpecies"]]
                    return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                           Lines=Lines, session=session)
                elif Mol + "_LineSubmission" in request.form:
                    for line in session[Mol]:
                        if Mol + "_" + line + "_Int" not in session or type(
                                session[Mol + "_" + line + "_Int"]) != float:
                            try:
                                session[Mol + "_" + line + "_Int"] = float(request.form[line + "_Int"])
                            except ValueError:
                                session[Mol + "_" + line + "_Int"] = request.form[line + "_Int"]
                                session["Error"] = "Please enter only floats in the Intensities (Format ex: 1.00e-20)"
                        if Mol + "_" + line + "_ER" not in session or type(session[Mol + "_" + line + "_ER"]) != float:
                            try:
                                session[Mol + "_" + line + "_ER"] = float(request.form[line + "_ER"])
                            except ValueError:
                                session[Mol + "_" + line + "_ER"] = request.form[line + "_ER"]
                                session["Error"] = "Please enter only floats in the Intensities (Format ex: 1.00e-20)"
                        if Mol + "_" + line + "_Unit" not in session:
                            session[Mol + "_" + line + "_Unit"] = request.form[line + "_Unit"]
                    return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                           Lines=Lines, session=session)

            return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                   Lines=Lines, session=session)
        else:
            return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                   Lines=Lines, session=session)
    else:
        session["Error"] = " "
        return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem, Lines=Lines, session=session)
# =========================================================================================================


# =========================================================================================================
@app.route('/UCLCHEM/', methods=["POST", "GET"])
def UCLCHEM():
    """

    Args:

    """
    UCLCHEMDict = {}
    for parameter in Online_UCLCHEMParameters:
        UCLCHEMDict[parameter] = Online_UCLCHEMDefaults[parameter]
    if request.method == "POST":
        parameter = 0
        while parameter < (len(Online_UCLCHEMParameters)):
            if request.form[Online_UCLCHEMParameters[parameter]] != '':
                if Online_UCLCHEMParameters[parameter] in Switches:
                    UCLCHEMDict[Online_UCLCHEMParameters[parameter]] = int(
                        request.form[Online_UCLCHEMParameters[parameter]])
                else:
                    UCLCHEMDict[Online_UCLCHEMParameters[parameter]] = float(
                        request.form[Online_UCLCHEMParameters[parameter]])
                session[Online_UCLCHEMParameters[parameter]] = UCLCHEMDict[Online_UCLCHEMParameters[parameter]]
            parameter += 1
    session["UCLCHEMDict"] = UCLCHEMDict
    return render_template('UCLCHEM.html', name=CodeName, session=session)
# =========================================================================================================


# =========================================================================================================
@app.route('/UCLCHEM/PhysResults', methods=["GET"])
def UCLCHEMPhysResults():
    """

    Args:

    """
    return render_template('UCLCHEMPhysResults.html', name=CodeName, session=session)


@app.route('/UCLCHEM/ChemResults', methods=["GET"])
def UCLCHEMChemResults():
    """

    Args:

    """
    return render_template('UCLCHEMChemResults.html', name=CodeName, session=session)
# =========================================================================================================


# =========================================================================================================
@app.route('/MCMCInference/Options', methods=["POST", "GET"])
def Options():
    """

    Args:

    """
    if request.method == "POST":
        session["Informed"] = request.form["Informed"]
        session["Walkers"] = request.form["Walkers"]
        session["StepsPerClick"] = request.form["StepsPerClick"]
        session["Walkers"] = int(session["Walkers"])
        session["StepsPerClick"] = int(session["StepsPerClick"])
        if session["StepsPerClick"] >= 100 and session["StepsPerClick"] % 10 == 0:
            session["StepsPerSave"] = 100
        elif session["StepsPerClick"] % 10 == 0 and session["StepsPerClick"] >= 10:
            session["StepsPerSave"] = 10
        elif session["StepsPerClick"] % 5 == 0 and session["StepsPerClick"] >= 5:
            session["StepsPerSave"] = 5
        elif session["StepsPerClick"] % 2 == 0 and session["StepsPerClick"] >= 2:
            session["StepsPerSave"] = 2
        else:
            session["StepsPerSave"] = 1
        session["SessionBackend"] = SaveFolder + request.form["SessionBackend"] + ".h5"
        if session["SessionBackend"] == ".h5":
            session["SessionBackend"] = SaveFolder + "MCMCBackend_Session_" + str(calendar.timegm(time.gmtime())) + ".h5"
        with shelve.open(session["SessionBackend"][:-3]) as db:
            for key in session.keys():
                db[key] = session[key]
        return redirect(url_for('Results'))
    return render_template('Options.html', name=CodeName, Lines=Lines, session=session)
# =========================================================================================================


# Inference loading and Results page, wip.
# =========================================================================================================
@app.route('/MCMCInference/Results', methods=["POST", "GET"])
def Results():
    """

    Args:

    """
    ChangingParamList = []
    ChangingDictRange = {}
    BaseDict = ChemicalCode_UnchangableParameters.copy()
    for key in session.keys():
        if key == "outSpecies":
            outSpeciesInRun = []
            PDLines = pd.DataFrame(columns=["Chemical", "Line", "Intensity", "Sigma", "Units"])
            indexPfDF = 0
            for species in session["outSpecies"]:
                outSpeciesInRun += [species]
                for line in session[species]:
                    baseName = species + "_" + line + "_"
                    PDLines.loc[indexPfDF] = [species, line, session[baseName + "Int"],
                                              session[baseName + "ER"], session[baseName + "Unit"]]
                    indexPfDF += 1
            BaseDict["outSpecies"] = outSpeciesInRun
        elif (key in ChemicalCode_Parameters or key in RadiativeCode_Parameters) and "_up" not in key:
            if key + "_up" in session.keys():
                ChangingParamList += [key]
                ChangingDictRange[key + "_low"] = session[key]
                ChangingDictRange[key + "_up"] = session[key + "_up"]
            else:
                BaseDict[key] = session[key]
        elif key == "Grid":
            GridFile = "../data/" + session["Grid"] + "_Grid.csv"
            Grid = pd.read_csv(GridFile, delimiter=",/", engine='python')
            GridDictionary = Grid.set_index('Parameter')["Grid"].to_dict()
            for P in GridDictionary.keys():
                GridDictionary[P] = (np.array(GridDictionary[P].split(",")).astype(np.float)).tolist()
        elif key == "Informed":
            if session["Informed"] == "Informed":
                session["Informed"] = True
            else:
                session["Informed"] = False
    for keys in GridDictionary.keys():
        GridDictionary[keys] = np.asarray(GridDictionary[keys])
    UserRangesDict = {}
    for ind, RangeLim in enumerate(ChangingParamList):
        low = ChangingDictRange[RangeLim + "_low"]
        lowGrid = (np.abs(GridDictionary[RangeLim] - low)).argmin()
        if lowGrid != 0:
            lowGrid -= 1
        UserRangesDict[RangeLim + "_low"] = lowGrid
        up = ChangingDictRange[RangeLim + "_up"]
        upGrid = (np.abs((GridDictionary)[RangeLim] - up)).argmin()
        if upGrid < len((GridDictionary)[RangeLim]) - 1:
            upGrid += 1
        UserRangesDict[RangeLim + "_up"] = upGrid
        if ind == 0:
            startPoints = np.random.randint(lowGrid, upGrid + 1, int(session["Walkers"]))
        else:
            startPoints = np.vstack([startPoints, np.random.randint(lowGrid, upGrid + 1, int(session["Walkers"]))])
    if len(ChangingParamList) > 1:
        startPoints = startPoints.T
    startPoints = startPoints.tolist()
    session['startPoints'] = startPoints
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    session["GridDictionary"] = GridDictionary
    js, tag = mcf.plotLastCornerWebUI(session["SessionBackend"], ChangingParamList, session["GridDictionary"],
                                      startPoints)
    if request.method == 'GET':
        return render_template('Results.html', name=CodeName, Lines=Lines, session=session, JS=js, Tag=tag)
    return redirect(url_for('Results'))


@app.route('/UCLCHEM/UCLCHEMTask', methods=['POST'])
def UCLCHEMTask():
    """

    Args:

    """
    UCLCHEMDict = session["UCLCHEMDict"]
    task = RunUCLCHEMAlone.apply_async(args=[UCLCHEMDict])
    return jsonify({}), 202, {'Location': url_for('UCLCHEMTaskStatus', task_id=task.id)}


@app.route('/UCLCHEM/UCLCHEMTask', methods=['POST'])
def UCLCHEMTask():
    """

    Args:

    """
    UCLCHEMDict = session["UCLCHEMDict"]
    task = RunUCLCHEMAlone.apply_async(args=[UCLCHEMDict])
    return jsonify({}), 202, {'Location': url_for('UCLCHEMTaskStatus', task_id=task.id)}


@app.route('/UCLCHEM/status/<task_id>')
def UCLCHEMTaskStatus(task_id):
    """

    Args:

    """
    task = RunUCLCHEMAlone.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 2,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 2,
            'total': 2,
            'status': str(task.info)
        }
    return jsonify(response)


@app.route('/MCMCInference/Results/longtask', methods=['POST'])
def longtask():
    """

    Args:

    """
    ChangingParamList = []
    ChangingDictRange = {}
    BaseDict = ChemicalCode_UnchangableParameters.copy()
    for key in session.keys():
        if key == "outSpecies":
            outSpeciesInRun = []
            PDLines = pd.DataFrame(columns=["Chemical", "Line", "Intensity", "Sigma", "Units"])
            indexPfDF = 0
            for species in session["outSpecies"]:
                outSpeciesInRun += [species]
                for line in session[species]:
                    baseName = species + "_" + line + "_"
                    PDLines.loc[indexPfDF] = [species, line, session[baseName + "Int"],
                                              session[baseName + "ER"], session[baseName + "Unit"]]
                    indexPfDF += 1
            BaseDict["outSpecies"] = outSpeciesInRun
        elif (key in ChemicalCode_Parameters or key in RadiativeCode_Parameters) and "_up" not in key:
            if key + "_up" in session.keys():
                ChangingParamList += [key]
                ChangingDictRange[key+"_low"] = session[key]
                ChangingDictRange[key+"_up"] = session[key+"_up"]
            else:
                BaseDict[key] = session[key]
    task = RunMCMC.apply_async(args=(BaseDict, ChangingParamList, ChangingDictRange, PDLines.to_json(),
                                     session["SessionBackend"], session["GridDictionary"],
                                     session["Informed"], int(session["Walkers"]), session["StepsPerClick"],
                                     session["StepsPerSave"], session["startPoints"]))
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route('/MCMCInference/Results/status/<task_id>')
def taskstatus(task_id):
    """

    Args:

    """
    task = RunMCMC.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...',
            'figurePath': ''
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', ''),
            'figurePath': task.info.get('figurePath', '')
        }
        if 'figureJS' in task.info:
            response['figureJS'] = task.info['figureJS']
            response['figureTag'] = task.info['figureTag']
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
            'figurePath': ''
        }
    return jsonify(response)
# =========================================================================================================


# About page, detailing funding infor and citations
# =========================================================================================================
@app.route('/About')
def About():
    """

    Args:

    """
    return render_template('about.html', name=CodeName)
# =========================================================================================================


# Page informing users how to cite this package
# =========================================================================================================
@app.route('/Citation')
def Citation():
    """

    Args:

    """
    return render_template('Citation.html', name=CodeName)
# =========================================================================================================

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

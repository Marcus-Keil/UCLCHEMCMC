#!/usr/bin/env python
import os
import shelve
import numpy as np
import pandas as pd
import billiard.pool as BilPool
from celery import Celery
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
#Lines = {"CO": ['1-0', '2-1'], "HCO+": ['(2,1)-(2,0)', '(2,2)-(2,1)']}
#Chemicals = ['CO', 'HCO+', ' ']

ChemFile = "../data/Chemicals.csv"
Chem = pd.read_csv(ChemFile, delimiter=",/", engine='python')
Lines = Chem.set_index('Chemical')["Lines"].to_dict()
for C in Lines.keys():
    Lines[C] = np.asarray(Lines[C].split(", ")).astype(str)
Chemicals = list(Lines.keys())
Chemicals += ['']


# File that contains the Ranges that are allowed
RangesFile = "../data/Range.csv"
Ranges = pd.read_csv(RangesFile, delimiter=":", engine='python')
ParameterRanges = Ranges.set_index("key")['value'].to_dict()

# The physical parameters that are allowed to be changed
PhysicalParameters = ['finalDens', 'finalDens_up', 'initialTemp', 'initialTemp_up',
                      'zeta', 'zeta_up', 'radfield', 'radfield_up',
                      'rout', 'rout_up']

# The parameters currently not allowd to be changed, but are needed
UnchangableParams = {"phase": 1, "switch": 1, "collapse": 1, "readAbunds": 0,
                     "writeStep": 1, "points": 1, "desorb": 1, "finalOnly": "True", 'fr': 1.0}

# Default parameter values if nothing is specified
ParameterDefaults = {'finalDens': 1.0e5,
                     'initialTemp': 10,
                     'zeta': 1,
                     'radfield': 1,
                     'rout': 0.05}

# MCMC Parameters as of now

# echo '.dump' | sqlite3 Database_New.db > test_new.dump
# cat test.dump | sqlite3 Database.db
#
MCMCwalkers = 10
MCMCStepsPerRun = 200
MCMCStepsPerItteration = 10
ManagerPoolSize = 1
FortranPoolSize = 2

MCMCNumberProcesses = 3
BaseUIFolder = "./"
ResultsFolder = "../results/"
DBLocation = "../data/Database.db"



# Currently unused parameters
# 'phase', 'switch', 'collapse', 'desorb', 'initialDens', 'initialDens_up',
# 'maxTemp', 'maxTemp_up',
# =========================================================================================================


# The celery task, wip.
# =========================================================================================================
@celery.task(bind=True)
def RunMCMC(self, BaseDict, ChangingParamList, ChangingDictRanges, PDLinesJson,
            MCMCFile, GridDictionary, Emulator=False, Informed=False, Walkers=MCMCwalkers,
            StepsPerClick=MCMCStepsPerRun, StepsPerSave=MCMCStepsPerItteration):
    OutSpecies = BaseDict["outSpecies"]
    for keys in GridDictionary.keys():
        GridDictionary[keys] = np.asarray(GridDictionary[keys])
    BaseDict["outSpecies"] = len(OutSpecies)
    UserRangesDict = {}
    for ind, RangeLim in enumerate(ChangingParamList):
        low = ChangingDictRanges[RangeLim+"_low"]
        lowGrid = (np.abs(GridDictionary[RangeLim]-low)).argmin()
        if lowGrid != 0:
            lowGrid -= 1
        UserRangesDict[RangeLim+"_low"] = lowGrid
        up = ChangingDictRanges[RangeLim+"_up"]
        upGrid = (np.abs(GridDictionary[RangeLim]-up)).argmin()
        if upGrid < len(GridDictionary[RangeLim])-1:
            upGrid += 1
        UserRangesDict[RangeLim+"_up"] = upGrid
        if Informed:
            startPoints = utils.createGausStartingPointsGrid(PDLinesJson, ChangingParamList, DBLocation, Walkers, GridDictionary)
            for i in range(len(startPoints)):
                if startPoints[i] < lowGrid:
                    startPoints[i] = lowGrid
                elif startPoints[i] > upGrid:
                    startPoints[i] = upGrid
        else:
            if ind == 0:
                startPoints = np.random.randint(lowGrid, upGrid+1, Walkers)
            else:
                startPoints = np.vstack([startPoints, np.random.randint(lowGrid, upGrid+1, Walkers)])
    if len(ChangingParamList) > 1:
        startPoints = startPoints.T
    iterations = int(StepsPerClick/StepsPerSave)
    if Emulator:
        MCMCFunction = mcf.MCMCSavesGridUIWithEmu
    else:
        MCMCFunction = mcf.MCMCSavesGridUI
    ManagerPool = BilPool.Pool(ManagerPoolSize, utils.worker_main)
    FortranPool = BilPool.Pool(FortranPoolSize, utils.worker_Fortran)
    for i in range(iterations):
        sampler = MCMCFunction(MCMCFile, MCMCNumberProcesses, len(ChangingParamList), Walkers,
                               StepsPerSave, BaseDict, ChangingParamList, startPoints,
                               DBLocation, GridDictionary, ParameterRanges, PDLinesJson,
                               UserRangesDict)
        js, tag = mcf.plotChainAndCornersWebUI(sampler, len(ChangingParamList), ChangingParamList, GridDictionary, UserRangesDict)
        self.update_state(state='PROGRESS', meta={'current': i+1, 'total': iterations,
                                                  'status': "completed " + str(
                                                      (i + 1) * StepsPerSave) + " of " + str(
                                                      StepsPerClick) + " steps",
                                                  'figureJS': js, 'figureTag': tag})
    for i in range(FortranPoolSize):
        mcf.FortranQueue.put(("Stop", []))
        time.sleep(2)
    mcf.Queue.put(("Stop", []))
    FortranPool.close()
    FortranPool.join()
    ManagerPool.close()
    ManagerPool.join()
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    return {'current': iterations, 'total': iterations, 'status': 'Task completed!',
            'result': "GridWebUI-" + str(len(sampler.get_chain()))}
# =========================================================================================================


# Home or Index Page
# =========================================================================================================
@app.route('/')
def Index():
    return render_template('index.html', name=CodeName)
# =========================================================================================================


# Initial MCMC page to give parameter ranges
# =========================================================================================================
@app.route('/MCMCInference/', methods=["POST", "GET"])
def MCMC():
    if request.method == "POST":
        if 'Load_Session' in request.form and request.form['Session_name'] != '':
            SessionName = request.form['Session_name']
            if os.path.isfile(SessionName + ".bak"):
                with shelve.open(SessionName) as db:
                    for key in db.keys():
                        session[key] = db[key]
                return redirect(url_for('Options'))
        parameter = 0
        while parameter < (len(PhysicalParameters)):
            if request.form[PhysicalParameters[parameter]] != '':
                lower = float(request.form[PhysicalParameters[parameter]])
                if ("_up" in PhysicalParameters[parameter] and PhysicalParameters[parameter-1] not in session[PhysicalParameters[parameter]]):
                    return render_template('MCMC.html', name=CodeName, Chemicals=Chemicals,
                                           Lines=Lines, session=session)
                if lower >= ParameterRanges[PhysicalParameters[parameter]+"_low"] and lower <= ParameterRanges[PhysicalParameters[parameter]+"_high"]:
                    session[PhysicalParameters[parameter]] = lower
                elif lower < ParameterRanges[PhysicalParameters[parameter]+"_low"]:
                    session[PhysicalParameters[parameter]] = ParameterRanges[PhysicalParameters[parameter] + "_low"]
                else:
                    session[PhysicalParameters[parameter]] = ParameterRanges[PhysicalParameters[parameter] + "_high"]
                if '_up' in PhysicalParameters[parameter+1]:
                    if request.form[PhysicalParameters[parameter+1]] != '':
                        upper = float(request.form[PhysicalParameters[parameter+1]])
                        if upper <= ParameterRanges[PhysicalParameters[parameter]+"_high"] and upper > session[PhysicalParameters[parameter]]:
                            session[PhysicalParameters[parameter + 1]] = upper
                        elif upper <= session[PhysicalParameters[parameter]]:
                            print("Upper bound was less than the lower number, keeping the parameter constant")
                        else:
                            session[PhysicalParameters[parameter + 1]] = ParameterRanges[PhysicalParameters[parameter]+"_high"]
                    elif PhysicalParameters[parameter+1] in session.keys():
                        session.pop(PhysicalParameters[parameter+1], None)
                    parameter += 1
            parameter += 1
        return redirect(url_for("MCMCPhys"))
    else:
        return render_template('MCMC.html', name=CodeName, Chemicals=Chemicals,
                                   Lines=Lines, session=session)
# =========================================================================================================


# Parameter range confirmation Page
# =========================================================================================================
@app.route('/MCMCInference/Phys', methods=["POST", "GET"])
def MCMCPhys():
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
    if "outSpecies" in session:
        NewChem = [Chem for Chem in Chemicals if Chem not in session["outSpecies"]]
    else:
        NewChem = Chemicals
        session["outSpecies"] = []

    if request.method == "POST":
        session["Error"] = " "
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
                if Mol+"_Add_Line" in request.form and (Mol+"_Lines" in request.form):
                    session[Mol] = request.form.getlist(Mol+"_Lines")
                    return render_template('MCMC_Chem.html', name=CodeName, Chemicals=NewChem,
                                           Lines=Lines, session=session)
                elif Mol+"_Clear" in request.form:
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
                elif Mol+"_LineSubmission" in request.form:
                    for line in session[Mol]:
                        if Mol+"_"+line+"_Int" not in session or type(session[Mol+"_"+line+"_Int"]) != float:
                            try:
                                session[Mol+"_"+line+"_Int"] = float(request.form[line+"_Int"])
                            except ValueError:
                                session[Mol + "_" + line + "_Int"] = request.form[line + "_Int"]
                                session["Error"] = "Please enter only floats in the Intensities (Format ex: 1.00e-20)"
                        if Mol + "_" + line + "_ER" not in session or type(session[Mol+"_"+line+"_ER"]) != float:
                            try:
                                session[Mol + "_" + line + "_ER"] = float(request.form[line + "_ER"])
                            except ValueError:
                                session[Mol+"_"+line+"_ER"] = request.form[line + "_ER"]
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
@app.route('/MCMCInference/Options', methods=["POST", "GET"])
def Options():
    if request.method == "POST":
        session["Grid"] = request.form["Grid"]
        session["Emulator"] = request.form["Emulator"]
        session["Informed"] = request.form["Informed"]
        session["Walkers"] = request.form["Walkers"]
        session["StepsPerClick"] = request.form["StepsPerClick"]
        session["Walkers"] = int(session["Walkers"])
        session["StepsPerClick"] = int(session["StepsPerClick"])
        if session["StepsPerClick"] > 100 and session["StepsPerClick"] % 10 == 0:
            session["StepsPerSave"] = 100
        elif session["StepsPerClick"] % 10 == 0 and session["StepsPerClick"] >= 10:
            session["StepsPerSave"] = 10
        elif session["StepsPerClick"] % 5 == 0 and session["StepsPerClick"] >= 5:
            session["StepsPerSave"] = 5
        elif session["StepsPerClick"] % 2 == 0 and session["StepsPerClick"] >= 2:
            session["StepsPerSave"] = 2
        else:
            session["StepsPerSave"] = 1
        session["SessionBackend"] = request.form["SessionBackend"] + ".h5"
        if session["SessionBackend"] == ".h5":
            session["SessionBackend"] = "MCMCBackend_Session_" + str(calendar.timegm(time.gmtime())) + ".h5"
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
    ChangingParamList = []
    ChangingDictRange = {}
    BaseDict = UnchangableParams.copy()
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
        elif key in PhysicalParameters and "_up" not in key:
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
        elif key == "Emulator":
            if session["Emulator"] == "Emulator":
                session["Emulator"] = True
            else:
                session["Emulator"] = False
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
            startPoints = np.random.randint(lowGrid, upGrid + 1, MCMCwalkers)
        else:
            startPoints = np.vstack([startPoints, np.random.randint(lowGrid, upGrid + 1, MCMCwalkers)])

    if len(ChangingParamList) > 1:
        startPoints = startPoints.T
    startPoints = startPoints.tolist()
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    session["GridDictionary"] = GridDictionary
    js, tag = mcf.plotLastCornerWebUI(session["SessionBackend"], ChangingParamList, session["GridDictionary"],
                                      PDLines.to_json(), UserRangesDict, BaseDict, DBLocation,
                                      ParameterRanges, startPoints)
    if request.method == 'GET':
        return render_template('Results.html', name=CodeName, Lines=Lines, session=session, JS=js, Tag=tag)
    return redirect(url_for('Results'))


@app.route('/MCMCInference/Results/longtask', methods=['POST'])
def longtask():
    ChangingParamList = []
    ChangingDictRange = {}
    BaseDict = UnchangableParams.copy()
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
        elif key in PhysicalParameters and "_up" not in key:
            if key + "_up" in session.keys():
                ChangingParamList += [key]
                ChangingDictRange[key+"_low"] = session[key]
                ChangingDictRange[key+"_up"] = session[key+"_up"]
            else:
                BaseDict[key] = session[key]
    task = RunMCMC.apply_async(args=(BaseDict, ChangingParamList, ChangingDictRange, PDLines.to_json(),
                                     session["SessionBackend"], session["GridDictionary"], session["Emulator"],
                                     session["Informed"], session["Walkers"], session["StepsPerClick"],
                                     session["StepsPerSave"]))
    return jsonify({}), 202, {'Location': url_for('taskstatus', task_id=task.id)}


@app.route('/MCMCInference/Results/status/<task_id>')
def taskstatus(task_id):
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
    return render_template('about.html', name=CodeName)
# =========================================================================================================


# Page informing users how to cite this package
# =========================================================================================================
@app.route('/Citation')
def Citation():
    return render_template('Citation.html', name=CodeName)
# =========================================================================================================

if __name__ == '__main__':
    app.run(debug=True)
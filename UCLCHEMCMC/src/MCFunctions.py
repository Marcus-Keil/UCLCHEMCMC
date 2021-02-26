import os
import time
import corner
import matplotlib
import emcee as mc
import numpy as np
import pandas as pd
from colour import Color
import matplotlib.pyplot as plt
import GUI
import billiard as Bil
import billiard.pool as BilPool
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.embed import autoload_static

import utils
os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use("Agg")

TMin = 3
TMax = 40
rhoMin = 1e3
rhoMax = 1e10
AvMin = 0.001
AvMax = 200

rangeForCorners = [(np.log10(rhoMin), np.log10(rhoMax)), (TMin, TMax),
                   ((1.6e21 * AvMin)/(1e6 * 3.086e+18), (1.6e21 * AvMax)/(1e6 * 3.086e+18))]

SQLManager = Bil.Manager()
SQLQueue = SQLManager.Queue()
SQLResultDict = SQLManager.dict()
FortranManager = Bil.Manager()
FortranQueue = FortranManager.Queue()
FortranResultDict = FortranManager.dict()


def priorWithRangesUI(Parameters, ChangingParamsKeys, GridDictionary, RangesDict, UserRangesDict, FinalDensity):
    for i in range(len(Parameters)):
        if Parameters[i] < UserRangesDict[ChangingParamsKeys[i]+"_low"] or Parameters[i] >= UserRangesDict[ChangingParamsKeys[i]+"_up"]:
            return -np.inf
    if "rout" in ChangingParamsKeys:
        AV = (FinalDensity * 3.086e+18 * GridDictionary["rout"][Parameters[ChangingParamsKeys.index("rout")]])/(1.6e21)
        if AV < RangesDict["Av_low"] or AV > RangesDict["Av_high"]:
            return -np.inf
    return 0.0


def UILikelihood(ChangingParamsValues, PDLines, BaseParameterDict, ChangingParamsKeys,
                 GridDictionary, RangesDict, UserRangesDict):
    ChangingParamsValues = ChangingParamsValues.astype(int)
    observation = PDLines["Intensity"].values
    sigmaObservation = PDLines["Sigma"].values
    Chemicals = PDLines["Chemical"].values
    if "finalDens" not in BaseParameterDict:
        if len(GridDictionary["finalDens"]) > ChangingParamsValues[ChangingParamsKeys.index("finalDens")] >= 0:
            FinalDensity = GridDictionary["finalDens"][ChangingParamsValues[ChangingParamsKeys.index("finalDens")]]
        else:
            return -np.inf
    else:
        FinalDensity = BaseParameterDict["finalDens"]

    if (priorWithRangesUI(ChangingParamsValues, ChangingParamsKeys, GridDictionary, RangesDict, UserRangesDict, FinalDensity) == 0):
        ChangingParams = {ChangingParamsKeys[i]: GridDictionary[ChangingParamsKeys[i]][ChangingParamsValues[i]] for i in range(len(ChangingParamsKeys))}
        LinesOfInterest = []
        for i in range(PDLines.shape[0]):
            LinesOfInterest += [utils.chemDat(PDLines.iloc[i]["Chemical"])[:-4] + "_" + PDLines.iloc[i]["Line"] +
                                " GHz)_" + PDLines.iloc[i]["Units"]]
        LinesOfInterest = np.asarray(LinesOfInterest)
        ParameterDict = {**BaseParameterDict, **ChangingParams}
        ModelData = utils.retrieveIntensitiesUI(ParameterDict=ParameterDict, ChangingParams=ChangingParams,
                                                LinesOfInterest=LinesOfInterest, Chemicals=Chemicals,
                                                LinesGiven=PDLines["Line"].values)
        if type(ModelData) == type(np.nan):
            if np.isnan(ModelData):
                return -np.inf
        if (None in ModelData):
            return -np.inf
        for i in range(len(ModelData)):
            if ModelData[i] == np.nan:
                ModelData[i] = -10000000000000000000000000000000
        sigma2 = sigmaObservation**2
        PofDK = -0.5*sum(((observation - ModelData)**2)/sigma2)
        print(PofDK)
        return PofDK
    else:
        return -np.inf


def MCMCSavesGridUI(MCMCSaveFile, numberProcesses, ndim, nwalkers, nSteps, knownParams, unknownParamsKeys,
                    startingPos, GridDictionary, RangesDict, PDLinesJson, UserRangesDict):
    os.environ["OMP_NUM_THREADS"] = "1"
    PDLines = pd.read_json(PDLinesJson)
    if os.path.isfile(MCMCSaveFile):
        try:
            backend = mc.backends.HDFBackend(MCMCSaveFile)
            previousChain = backend.get_chain()
            startingPos = previousChain[len(previousChain) - 1]
            nwalkers, ndim = startingPos.shape
        except:
            del backend
            os.remove(MCMCSaveFile)
            time.sleep(1)
            backend = mc.backends.HDFBackend(MCMCSaveFile)
            backend.reset(nwalkers, ndim)
    elif not os.path.isfile(MCMCSaveFile):
        backend = mc.backends.HDFBackend(MCMCSaveFile)
        backend.reset(nwalkers, ndim)
    with BilPool.Pool(numberProcesses) as pool:
        pool.lost_worker_timeout = 500
        sampler = mc.EnsembleSampler(nwalkers, ndim, UILikelihood,
                                     args=((PDLines, knownParams, unknownParamsKeys, GridDictionary, RangesDict,
                                            UserRangesDict)),
                                     pool=pool, backend=backend)
        sampler.run_mcmc(startingPos, nSteps, progress=True, store=True)
        pool.close()
        pool.join()
    return sampler


def plotChainAndCornersWebUI(sampler, ndim, changingParamsKeys, GridDictionary, UserRangesDict):
    samplesToUse = sampler.get_chain()
    samples = samplesToUse.copy()
    UserRangesArray = []
    for N in range(ndim):
        samples[:, :, N] = GridDictionary[changingParamsKeys[N]][samples.astype(int)[:, :, N]]
        if changingParamsKeys[N] == "finalDens":
            UserRangesArray += [(np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_low"]]),
                                 np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_up"]]))]
        else:
            UserRangesArray += [(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N]+"_low"]],
                                 GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N]+"_up"]])]
    flat_samples = sampler.get_chain(flat=True)
    for N in range(ndim):
        flat_samples[:, N] = GridDictionary[changingParamsKeys[N]][flat_samples.astype(int)[:, N]]
    flat_samples[:, 0] = np.log10(flat_samples[:, 0])
    grid = make_Grid(ndim, flat_samples, changingParamsKeys)
    js, tag = autoload_static(grid, CDN, "Corner Plot")
    return js, tag


def plotLastCornerWebUI(MCMCSaveFile, changingParamsKeys, GridDictionary, PDLinesJson,
                        UserRangesDict, knownParams, RangesDict, startingPos):
    PDLines = pd.read_json(PDLinesJson)
    startingPosArray = np.asarray(startingPos.copy())
    nwalkers, ndim = startingPosArray.shape
    for keys in GridDictionary.keys():
        GridDictionary[keys] = np.asarray(GridDictionary[keys])

    if os.path.isfile(MCMCSaveFile):
        try:
            backend = mc.backends.HDFBackend(MCMCSaveFile)
            previousChain = backend.get_chain()
            startingPosArray = previousChain[len(previousChain) - 1]
            nwalkers, ndim = startingPosArray.shape
            sampler = mc.EnsembleSampler(nwalkers, ndim, UILikelihood,
                                         args=((PDLines, knownParams, changingParamsKeys, GridDictionary,
                                                RangesDict, UserRangesDict)),
                                         backend=backend)
            samples = sampler.get_chain()
            UserRangesArray = []
            for N in range(ndim):
                samples[:, :, N] = GridDictionary[changingParamsKeys[N]][samples.astype(int)[:, :, N]]
                if changingParamsKeys[N] == "finalDens":
                    UserRangesArray += [
                        (
                        np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_low"]]),
                        np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_up"]]))]
                else:
                    UserRangesArray += [
                        (GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_low"]],
                         GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_up"]])]
            flat_samples = sampler.get_chain(flat=True)
            for N in range(ndim):
                flat_samples[:, N] = GridDictionary[changingParamsKeys[N]][flat_samples.astype(int)[:, N]]
            flat_samples[:, 0] = np.log10(flat_samples[:, 0])
            grid = make_Grid(ndim, flat_samples, changingParamsKeys)
            js, tag = autoload_static(grid, CDN, "Corner Plot")
            for keys in GridDictionary.keys():
                GridDictionary[keys] = (GridDictionary[keys]).tolist()
            return js, tag
        except:
            print("Unable to start from previous point")
    UserRangesArray = []
    for N in range(ndim):
        if changingParamsKeys[N] == "finalDens":
            UserRangesArray += [
                (np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_low"]]),
                 np.log10(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_up"]]))]
        else:
            UserRangesArray += [(GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_low"]],
                                 GridDictionary[changingParamsKeys[N]][UserRangesDict[changingParamsKeys[N] + "_up"]])]
    flat_samples = startingPosArray.astype(float).copy()
    for N in range(ndim):
        flat_samples[:, N] = GridDictionary[changingParamsKeys[N]][startingPosArray.astype(int)[:, N]]
    grid = make_Grid(ndim, flat_samples, changingParamsKeys)
    js, tag = autoload_static(grid, CDN, "Corner Plot")
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    return js, tag


def make_Hist(hist, edges,xAxisLabel = '', yAxisLabel = '', Log=True, width=100):
    if Log:
        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                   background_fill_color="#ffffff", x_axis_type='log', plot_width=width, plot_height=width)
    else:
        p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                   background_fill_color="#ffffff", plot_width=width, plot_height=width)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#000000", line_color="white", alpha=0.75)
    p.y_range.start = 0
    p.xaxis.axis_label = xAxisLabel
    p.yaxis.axis_label = yAxisLabel
    p.grid.grid_line_color = "white"
    return p


def make_HM(hist, xedge, yedge, xAxisLabel = '', yAxisLabel = '', width=100, xLog=True, yLog=True):
    values = np.unique(hist)
    for i in range(len(values)):
        hist = np.where(hist == values[i], i, hist)
    hist = hist.astype(int)
    Start = Color("#ffffff")
    colors = list(Start.range_to(Color("#000000"), len(values)))
    for i in range(len(colors)):
        colors[i] = colors[i].get_hex_l()
    colors = np.asarray(colors)
    if xLog and yLog:
        p = figure(background_fill_color=colors[0], plot_width=width, plot_height=width, x_axis_type='log',
                   y_axis_type='log')
    elif xLog and not yLog:
        p = figure(background_fill_color=colors[0], plot_width=width, plot_height=width, x_axis_type='log')
    elif xLog and not yLog:
        p = figure(background_fill_color=colors[0], plot_width=width, plot_height=width, y_axis_type='log')
    else:
        p = figure(background_fill_color=colors[0], plot_width=width, plot_height=width)
    for i in range(len(xedge)-1):
        p.quad(top=yedge[1:], bottom=yedge[:-1], left=xedge[i], right=xedge[i+1], color=colors[hist[i, :]])
    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    p.xaxis.axis_label = xAxisLabel
    p.yaxis.axis_label = yAxisLabel
    p.grid.grid_line_color="white"
    return p


def make_Grid(ndim, flat_samples, ParameterNames):
    defaultWidth = int(800 / ndim)
    allDimensions = []
    for i in range(1, ndim+1):
        ThisLine = []
        for j in range(i):
            if j == 0 and (j+1) == i:
                yAxisLabel = "P("+ParameterNames[j]+")"
                yTickSize = '12pt'
            elif j == 0:
                yAxisLabel = ParameterNames[i-1]
                yTickSize = '12pt'
            else:
                yAxisLabel = ''
                yTickSize = '0pt'
            if i == ndim:
                xAxisLabel = ParameterNames[j]
                xTickSize = '12pt'
            else:
                xAxisLabel = ''
                xTickSize = '0pt'

            if (j+1) == i:
                hist, edges = np.histogram(flat_samples[:, j], bins=np.shape(np.unique(flat_samples[:, j]))[0])
                if ParameterNames[j] == "":  # "rout" or ParameterNames[j] == "finalDens":
                    plot = make_Hist(hist, edges, xAxisLabel=xAxisLabel, yAxisLabel=yAxisLabel,
                                 Log=True, width=defaultWidth)
                else:
                    plot = make_Hist(hist, edges, xAxisLabel=xAxisLabel, yAxisLabel=yAxisLabel,
                                     Log=False, width=defaultWidth)
                plot.xaxis.major_label_text_font_size = xTickSize
                plot.yaxis.major_label_text_font_size = yTickSize
                ThisLine += [plot]
            else:
                if np.shape(flat_samples)[0] == 0:
                    hist = [[0, 0], [0, 0]]
                    xedge = [0, 1]
                    yedge = [0, 1]
                else:
                    OutFromHist= plt.hist2d(flat_samples[:, j], flat_samples[:, i-1],
                                            bins=(np.shape(np.unique(flat_samples[:, j]))[0],
                                                  np.shape(np.unique(flat_samples[:, i-1]))[0]))
                    hist = OutFromHist[0]
                    xedge = OutFromHist[1]
                    yedge = OutFromHist[2]
                if ParameterNames[j] == "":  # "rout" or ParameterNames[j] == "finalDens":
                    HM_X_log = True
                else:
                    HM_X_log = False
                if ParameterNames[i-1] == "":  # "rout" or ParameterNames[i-1] == "finalDens":
                    HM_Y_log = True
                else:
                    HM_Y_log = False
                plot = make_HM(hist, xedge, yedge, xAxisLabel=xAxisLabel, yAxisLabel=yAxisLabel,
                               width=defaultWidth, xLog=HM_X_log, yLog=HM_Y_log)
                plot.xaxis.major_label_text_font_size = xTickSize
                plot.yaxis.major_label_text_font_size = yTickSize
                ThisLine += [plot]
        allDimensions += [ThisLine]
    grid = gridplot(allDimensions)
    return grid


def CornerPlots(SessionName, Parameters, ParameterRanges=np.array([[-1]]),
                GridType='Coarse', BurnIn=-1, PlotChains = False,
                CornerPackagePlot=False, PlotGivenValues=False, GivenValues=[]):
    FileName = SessionName[:-3]
    if PlotGivenValues and (GivenValues == [] or len(Parameters) != len(GivenValues)):
        print("You must set GivenValues to have the same number of entries as Parameters, if you set PlotGivenValues to True")
        return
    GridFile = GridType + "_Grid.csv"
    Grid = pd.read_csv(GridFile, delimiter=",/", engine='python')
    GridDictionary = Grid.set_index('Parameter')["Grid"].to_dict()
    for P in GridDictionary.keys():
        GridDictionary[P] = (np.array(GridDictionary[P].split(",")).astype(np.float))

    if BurnIn < 0:
        BurnIn = 0
    backend = mc.backends.HDFBackend(SessionName)
    samplesToUse = backend.get_chain(discard=BurnIn)
    samples = samplesToUse.copy()
    ndim = samples.shape[2]
    for dim in range(ndim):
        samples[:, :, dim] = GridDictionary[Parameters[dim]][samples.astype(int)[:, :, dim]]
    flat_samples = backend.get_chain(discard=BurnIn, flat=True)
    for dim in range(ndim):
        flat_samples[:, dim] = GridDictionary[Parameters[dim]][flat_samples.astype(int)[:, dim]]

    if PlotChains:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(Parameters[i])
            if Parameters[i] == 'finalDens' or Parameters[i] == 'rout':
                ax.set_yscale('log')
            else:
                ax.set_yscale('linear')
            if PlotGivenValues:
                ax.axhline(GivenValues[i], color='b', linestyle='solid', linewidth=3, alpha=0.2)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(FileName + "_Chain.png")
        plt.close(fig)
    if CornerPackagePlot:
        if ParameterRanges[0][0] == -1:
            fig = corner.corner(flat_samples, labels=Parameters, range=rangeForCorners)
        else:
            fig = corner.corner(flat_samples, labels=Parameters)
            plt.show()
        fig.savefig(FileName + "_Corner.png")
        plt.close(fig)
    else:
        fig, axs = plt.subplots(ndim, ndim, figsize=(10, 10))
        if PlotGivenValues:
            for i in range(len(Parameters)):
                if Parameters[i] == "rout" or Parameters[i] == "finalDens":
                    GivenValues[i] = np.log10(GivenValues[i])
        for i in range(0, ndim):
            if i == 0:
                weights = np.ones_like(flat_samples[:, 0]) / len(flat_samples[:, 0])
                if Parameters[0] == "rout" or Parameters[0] == "finalDens":
                    hist, edges = np.histogram(np.log10(flat_samples[:, 0]), density=True,
                                               bins=np.unique(np.log10(flat_samples[:, 0])))
                    axs[0, 0].hist(np.log10(flat_samples[:, 0]), weights=weights, bins=np.unique(
                        np.log10(flat_samples[:, 0])))  # Make histogram of 1d potential here in Log space
                    if PlotGivenValues:
                        axs[0, 0].axvline(GivenValues[0], color='k', linestyle='dashed', linewidth=1)
                else:
                    hist, edges = np.histogram(flat_samples[:, 0], density=True, bins=np.unique(flat_samples[:, 0]))
                    axs[0, 0].hist(flat_samples[:, 0], weights=weights, bins=np.unique(flat_samples[:, 0]))
                    if PlotGivenValues:
                        axs[0, 0].axvline(GivenValues[0], color='k', linestyle='dashed', linewidth=1)
                yAxisLabel = "P(" + Parameters[0] + ")"
                axs[0, 0].set_ylabel(yAxisLabel)
                axs[0, 0].set_yticks([], [])
                axs[0, 0].set_xticks([], [])
                axs[0, 0].set_xlim((edges.min(), edges.max()))
            else:
                for j in range(i + 1):
                    if j == 0 and j == i:
                        yAxisLabel = "P(" + Parameters[j] + ")"
                    elif j == 0:
                        yAxisLabel = Parameters[i]
                    else:
                        yAxisLabel = ''
                    if i + 1 == ndim:
                        xAxisLabel = Parameters[j]
                    else:
                        xAxisLabel = ''
                    if Parameters[j] == "rout" or Parameters[j] == "finalDens":
                        xValues = np.log10(flat_samples[:, j])
                    else:
                        xValues = flat_samples[:, j]
                    if Parameters[i] == "rout" or Parameters[i] == "finalDens":
                        yValues = np.log10(flat_samples[:, i])
                    else:
                        yValues = flat_samples[:, i]
                    if j == i:
                        weights = np.ones_like(xValues) / len(xValues)
                        if Parameters[j] == "rout" or Parameters[j] == "finalDens":
                            hist, edges = np.histogram(xValues, density=True,
                                                       bins=np.shape(np.unique(xValues))[0])
                            axs[i, j].hist(xValues, weights=weights, bins=np.unique(xValues))
                        else:
                            hist, edges = np.histogram(xValues, density=True,
                                                       bins=np.shape(np.unique(xValues))[0])
                            axs[i, j].hist(xValues, weights=weights, bins=np.unique(xValues))
                        if PlotGivenValues:
                            axs[i, j].axvline(GivenValues[j], color='k', linestyle='dashed', linewidth=1)
                        axs[i, j].set_xlim(edges.min(), edges.max())
                        axs[i, j].set_yticks([], [])
                    else:
                        weights = np.ones_like(xValues) / len(xValues)
                        if i == 1:
                            h, x, y, image = axs[i, j].hist2d(xValues, yValues,
                                                              bins=(np.shape(np.unique(xValues))[0],
                                                                    np.shape(np.unique(yValues))[0]), weights=weights)
                        else:
                            axs[i, j].hist2d(xValues, yValues,
                                             bins=(np.shape(np.unique(xValues))[0],
                                                   np.shape(np.unique(yValues))[0]), weights=weights)
                        if PlotGivenValues:
                            axs[i, j].axvline(GivenValues[j], color='w', linestyle='solid', linewidth=1)
                            axs[i, j].axhline(GivenValues[i], color='w', linestyle='solid', linewidth=1)
                    axs[i, j].set_ylabel(yAxisLabel)
                    axs[i, j].set_xlabel(xAxisLabel)
                    if i + 1 != ndim:
                        axs[i, j].set_xticks([], [])
                    if j != 0 and j != i:
                        axs[i, j].set_yticks([], [])
        for i in range(ndim):
            for j in range(ndim):
                if j > i:
                    axs[i, j].remove()

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(image, cax=cbar_ax)
        plt.savefig(FileName+".png")
        plt.close(fig)
    return


def MakeSyntheticData(ChangingParameters, LinesOfInterestDict, StandardDeviationSpread=0.1, WithError=True):
    BaseDictionary = {"phase": 1, "switch": 1, "collapse": 1, "readAbunds": 0, "writeStep": 1, "points": 1, "desorb": 1,
                      "finalOnly": "True", "fr": 1.0}
    UCLCHEMDict = {**BaseDictionary, **ChangingParameters}
    for key in GUI.ParameterDefaults.keys():
        if key not in UCLCHEMDict:
            UCLCHEMDict[key] = GUI.ParameterDefaults[key]
    ParamDF, ChemDF = utils.UCLChemDataFrames(UCLChemDict=UCLCHEMDict, Queue=False)
    UCLParamOut = utils.RadexForGrid(UCLChemDict=UCLCHEMDict, UCLParamDF=ParamDF, UCLChemDF=ChemDF, Queue=False)
    ReturnLines = []
    for chems in LinesOfInterestDict.keys():
        chemical = utils.chemDat(chems)[:-4]
        for line in LinesOfInterestDict[chems]:
            ReturnLines += [chemical + '_' + line]
    ReturnLines = np.asarray(ReturnLines)
    SyntheticData_True = np.asarray(UCLParamOut[ReturnLines].values[0]).astype(float)
    if WithError:
        SyntheticData_Gaus = np.random.normal(SyntheticData_True, SyntheticData_True*StandardDeviationSpread)
        SyntheticData_Error = SyntheticData_True*StandardDeviationSpread
        SyntheticDataPD = pd.DataFrame(np.array([ReturnLines, SyntheticData_Gaus, SyntheticData_Error]).T,
                                       columns=['Line_Unit', 'Measurement', 'Error'])
        SyntheticDataPD = SyntheticDataPD.astype({'Measurement': 'float', 'Error': 'float'})
        return SyntheticDataPD
    else:
        return UCLParamOut[ReturnLines]

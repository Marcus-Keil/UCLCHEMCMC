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
import shelve
import billiard as Bil
import billiard.pool as BilPool
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.embed import autoload_static
from bokeh.models import FuncTickFormatter

import utils
os.environ["OMP_NUM_THREADS"] = "1"
matplotlib.use("Agg")

#Initialisation of the Managers for the SQL and Fortran workers.
SQLManager = Bil.Manager()
SQLQueue = SQLManager.Queue()
SQLResultDict = SQLManager.dict()
FortranManager = Bil.Manager()
FortranQueue = FortranManager.Queue()
FortranResultDict = FortranManager.dict()

# Negative infinity stand in for the prior, in case a different value is desired
NegInfStandIn = -np.inf
# Dictionary to define how the parameters should look on the final plot, rather than the code based naming convention
OutputParameterDic = {'finalDens': 'Density', 'initialTemp': 'Temperature', 'rout': 'R_out', 'radfield': 'UV factro',
                      'zeta': 'CR factor'}


def priorWithRangesUI(Parameters, ChangingParamsKeys, GridDictionary, RangesDict, UserRangesDict, FinalDensity):
    """
    Prior function that checks if all parameters that the MCMC changes are within the values the user provided

    Args:
        Parameters: Value of the changing parameters
        ChangingParamsKeys: Keys of the dictionary of the parameters that are allowed to change
        GridDictionary: Dictionary containing the arrays of the grid for each parameter
        RangesDict: Dictionary containing the upper and lower ends of the allowed ranges in grid space
        UserRangesDict: Dictionary containing the upper and lower ends of the allowed ranges in parameter space
        FinalDensity: The final density that the current model has, to be used with rout to make sure that AV
            stays within acceptable ranges based on inputs from the user
    """
    for i in range(len(Parameters)):
        if Parameters[i] < UserRangesDict[ChangingParamsKeys[i]+"_low"] or Parameters[i] >= \
                UserRangesDict[ChangingParamsKeys[i]+"_up"] or np.isnan(Parameters[i]):
            return -np.inf
    if "rout" in ChangingParamsKeys:
        AV = (FinalDensity * 3.086e+18 * GridDictionary["rout"][Parameters[ChangingParamsKeys.index("rout")]])/(1.6e21)
        if AV < RangesDict["Av_low"] or AV > RangesDict["Av_high"]:
            return -np.inf
    return 0.0


def UILikelihood(ChangingParamsValues, PDLines, BaseParameterDict, ChangingParamsKeys,
                 GridDictionary, RangesDict, UserRangesDict):
    """
    Likelihood function for the MCMC inference which will also call the requierd functions to create or retrieve models

    Args:
        ChangingParamsValues: List of the values for the current MCMC walkers step
        PDLines: Pandas dataframe of the lines given by the user
        BaseParameterDict: The basic UCLCHEM and RADEX parameters, that are not changing for the inference
        ChangingParamsKeys: Keys of the dictionary of the parameters that are allowed to change
        GridDictionary: Dictionary containing the arrays of the grid for each parameter
        RangesDict: Dictionary containing the upper and lower ends of the allowed ranges in grid space
        UserRangesDict: Dictionary containing the upper and lower ends of the allowed ranges in parameter space
    """
    ChangingParamsValues = ChangingParamsValues.astype(int)
    observation = PDLines["Intensity"].values
    sigmaObservation = PDLines["Sigma"].values
    Chemicals = PDLines["Chemical"].values
    if "finalDens" not in BaseParameterDict:
        if len(GridDictionary["finalDens"]) > ChangingParamsValues[ChangingParamsKeys.index("finalDens")] >= 0:
            FinalDensity = GridDictionary["finalDens"][ChangingParamsValues[ChangingParamsKeys.index("finalDens")]]
        else:
            return NegInfStandIn
    else:
        FinalDensity = BaseParameterDict["finalDens"]
    if (priorWithRangesUI(ChangingParamsValues, ChangingParamsKeys, GridDictionary, RangesDict, UserRangesDict,
                          FinalDensity) == 0):
        ChangingParams = {ChangingParamsKeys[i]: GridDictionary[ChangingParamsKeys[i]][ChangingParamsValues[i]] for i in
                          range(len(ChangingParamsKeys))}
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
                return NegInfStandIn
        if (None in ModelData):
            return NegInfStandIn
        for i in ModelData:
            if type(i) == type(np.nan):
                if np.isnan(i):
                    return NegInfStandIn
            elif i == None:
                return NegInfStandIn

        sigma2 = sigmaObservation ** 2
        PofDK = -0.5 * np.sum(((observation - ModelData) ** 2) / sigma2)
        return PofDK
    else:
        return NegInfStandIn


def MCMCSavesGridUI(MCMCSaveFile, numberProcesses, ndim, nwalkers, nSteps, knownParams, unknownParamsKeys,
                    startingPos, GridDictionary, RangesDict, PDLinesJson, UserRangesDict):
    """
    This function calls the MCMC after checking if the inference has previously been started and loading the
    walker file if it has. This also creates the worker pool the walkers have access to.

    Args:
        MCMCSaveFile: name of the current session for sake of storing the MCMC model and user input
        numberProcesses: number of processor cores that the walkers are allowed to use, not including the workers
            for the Fortran and SQL management
        ndim: number of dimension that the infernce has
        nwalkers: number of walkers the MCMC inference is using
        nSteps: number of steps to take prior to saving the current state
        knownParams: list of parameter names that will not be changing during the inference
        unknownParamsKeys: Keys of the dictionary of the parameters that are allowed to change
        startingPos: Values of the starting position of the Inference
        GridDictionary: Dictionary containing the arrays of the grid for each parameter
        RangesDict: Dictionary containing the upper and lower ends of the allowed ranges in grid space
        PDLinesJson: Pandas dataframe of the lines given by the user as a JSON object
        UserRangesDict: Dictionary containing the upper and lower ends of the allowed ranges in parameter space
    """
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
                                     moves=mc.moves.DEMove(1e-02),  # (mc.moves.DESnookerMove(), 0.3),],
                                     pool=pool, backend=backend, live_dangerously=True)
        sampler.run_mcmc(startingPos, nSteps, progress=True, store=True, skip_initial_state_check=True)
        pool.close()
        pool.join()
    return sampler


def plotLastCornerWebUI(MCMCSaveFile, changingParamsKeys, GridDictionary, startingPos):
    """
    Plots the most recent corner plot that is available from the save file, or initial conditions of the inference

    Args:
        MCMCSaveFile: name of the current session for sake of storing the MCMC model and user input
        changingParamsKeys: Keys of the dictionary of the parameters that are allowed to change
        GridDictionary: Dictionary containing the arrays of the grid for each parameter
        startingPos: Values of the starting position of the Inference
    """
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
            flat_samples = backend.get_chain(flat=True).astype(int)
            grid = make_Grid(ndim, flat_samples, changingParamsKeys, GridDictionary)
            js, tag = autoload_static(grid, CDN, "Corner Plot")
            for keys in GridDictionary.keys():
                GridDictionary[keys] = (GridDictionary[keys]).tolist()
            return js, tag
        except:
            print("Unable to start from previous point")
    flat_samples = startingPosArray.astype(float).copy()
    grid = make_Grid(ndim, flat_samples, changingParamsKeys, GridDictionary)
    js, tag = autoload_static(grid, CDN, "Corner Plot")
    for keys in GridDictionary.keys():
        GridDictionary[keys] = (GridDictionary[keys]).tolist()
    return js, tag


def make_Hist(hist, edges, CurrentGrid, xAxisLabel = '', yAxisLabel = '', width=100):
    """
    Makes a histogram of the posteriors using just one parameter, used in plotting corner
    plots when not using the "corner" package

    Args:
        hist: The values of the np.hist function
        edges: The edges of each bin for the histogram
        CurrentGrid: Grid dictionary of the current parameters grid space
        xAxisLabel: Label to be given to the X axis
        yAxisLabel: Label to be given to the Y axis
        width: the width that the plot should have
    """
    if xAxisLabel != '':
        height = width + 37
    else:
        height = width
    if yAxisLabel != '':
        width = width + 77
    p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
               background_fill_color="#ffffff", plot_width=width, plot_height=height)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#000000", line_color="white", alpha=0.75)
    p.y_range.start = 0
    p.xaxis.axis_label = xAxisLabel
    if xAxisLabel != '':
        xlab_dict = {}
        if xAxisLabel == 'finalDens' or xAxisLabel == 'rout':
            p.xaxis.ticker.desired_num_ticks = 2
            for i, s in enumerate(CurrentGrid):
                xlab_dict[i] = "{:.1e}".format(s)
        else:
            p.xaxis.ticker.desired_num_ticks = 3
            for i, s in enumerate(CurrentGrid):
                xlab_dict[i] = s

        p.xaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % xlab_dict)
    p.yaxis.axis_label = yAxisLabel
    p.grid.grid_line_color = "white"
    return p


def make_HM(hist, xedge, yedge, XGrid, YGrid, xAxisLabel = '', yAxisLabel = '', width=100):
    """
    Creates the 2d distribution of two parameters, used when the "corner" package is not being used to create
    the corner plots
    Args:
        hist: The values of the np.hist function
        xedge: The edges of each bin for the histogram of the X axis
        yedge: The edges of each bin for the histogram of the Y axis
        XGrid: Grid dictionary of the current parameters grid space for the X axis
        YGrid: Grid dictionary of the current parameters grid space for the Y axis
        xAxisLabel: Label to be given to the X axis
        yAxisLabel: Label to be given to the Y axis
        width: the width that the plot should have
    """
    values = np.unique(hist)
    for i in range(len(values)):
        hist = np.where(hist == values[i], i, hist)
    hist = hist.astype(int)
    Start = Color("#ffffff")
    colors = list(Start.range_to(Color("#000000"), len(values)))
    for i in range(len(colors)):
        colors[i] = colors[i].get_hex_l()
    colors = np.asarray(colors)
    if xAxisLabel != '':
        height = width+37
    else:
        height = width
    if yAxisLabel != '':
        width = width+77
    p = figure(background_fill_color=colors[0], plot_width=width, plot_height=height)
    for i in range(len(xedge)-1):
        p.quad(top=yedge[1:], bottom=yedge[:-1], left=xedge[i], right=xedge[i+1], color=colors[hist[i, :]])
    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    p.xaxis.axis_label = xAxisLabel
    p.yaxis.axis_label = yAxisLabel
    if xAxisLabel != '':
        xlab_dict = {}
        if xAxisLabel == 'finalDens' or xAxisLabel == 'rout':
            p.xaxis.ticker.desired_num_ticks = 2
            for i, s in enumerate(XGrid):
                xlab_dict[i] = "{:.1e}".format(s)
        else:
            p.xaxis.ticker.desired_num_ticks = 3
            for i, s in enumerate(XGrid):
                xlab_dict[i] = s
        p.xaxis.formatter = FuncTickFormatter(code="""
            var labels = %s;
            return labels[tick];
        """ % xlab_dict)
    if yAxisLabel != '':
        ylab_dict = {}
        if yAxisLabel == 'finalDens' or yAxisLabel == 'rout':
            for i, s in enumerate(YGrid):
                ylab_dict[i] = "{:.1e}".format(s)
        else:
            for i, s in enumerate(YGrid):
                ylab_dict[i] = s
        p.yaxis.formatter = FuncTickFormatter(code="""
                    var labels = %s;
                    return labels[tick];
                """ % ylab_dict)
    p.grid.grid_line_color="white"
    return p


def make_Grid(ndim, flat_samples, ParameterNames, GridDictionary):
    """
    Corner plot function that uses make_HM and make_hist to create corner plots for the User interface

    Args:
        ndim: number of dimensions that the inference has
        flat_samples: the flattened chain samples from the inference
        ParameterNames: name of the parameters being inferred
        GridDictionary: Dictionary containing the arrays of the grid for each parameter
    """
    defaultWidth = int(800 / ndim)
    allDimensions = []
    for i in range(1, ndim+1):
        ThisLine = []
        for j in range(i):
            if j == 0 and (j+1) == i:
                yAxisLabel = "P("+ParameterNames[j]+")"
                yTickSize = '0pt'
                currentXBins = np.linspace(flat_samples[:, j].min() - 0.5, flat_samples[:, j].max() + 0.5,
                                           int(flat_samples[:, j].max() - flat_samples[:, j].min() + 2))
            elif j == 0:
                yAxisLabel = ParameterNames[i-1]
                yTickSize = '10pt'
                currentYBins = np.linspace(flat_samples[:, i-1].min() - 0.5, flat_samples[:, i-1].max() + 0.5,
                                           int(flat_samples[:, i-1].max() - flat_samples[:, i-1].min() + 2))
            else:
                yAxisLabel = ''
                yTickSize = '0pt'
                currentYBins = np.linspace(flat_samples[:, i-1].min() - 0.5, flat_samples[:, i-1].max() + 0.5,
                                           int(flat_samples[:, i-1].max() - flat_samples[:, i-1].min() + 2))
            if i == ndim:
                xAxisLabel = ParameterNames[j]
                xTickSize = '10pt'
                currentXBins = np.linspace(flat_samples[:, j].min() - 0.5, flat_samples[:, j].max() + 0.5,
                                           int(flat_samples[:, j].max() - flat_samples[:, j].min() + 2))
            else:
                xAxisLabel = ''
                xTickSize = '0pt'
                currentXBins = np.linspace(flat_samples[:, j].min() - 0.5, flat_samples[:, j].max() + 0.5,
                                           int(flat_samples[:, j].max() - flat_samples[:, j].min() + 2))
            if (j+1) == i:
                hist, edges = np.histogram(flat_samples[:, j], bins=currentXBins)
                plot = make_Hist(hist, edges, xAxisLabel=xAxisLabel, yAxisLabel=yAxisLabel,
                                 width=defaultWidth, CurrentGrid=GridDictionary[ParameterNames[j]])
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
                                            bins=(currentXBins, currentYBins))
                    hist = OutFromHist[0]
                    xedge = OutFromHist[1]
                    yedge = OutFromHist[2]
                plot = make_HM(hist, xedge, yedge, xAxisLabel=xAxisLabel, yAxisLabel=yAxisLabel,
                               width=defaultWidth, XGrid=GridDictionary[ParameterNames[j]],
                               YGrid=GridDictionary[ParameterNames[i-1]])
                plot.xaxis.major_label_text_font_size = xTickSize
                plot.yaxis.major_label_text_font_size = yTickSize
                ThisLine += [plot]
        allDimensions += [ThisLine]
    grid = gridplot(allDimensions)
    return grid


def CornerPlots(SessionName, Parameters, GridType='Coarse', BurnIn=-1, PlotChains = False,
                CornerPackagePlot=False, PlotGivenValues=False, GivenValues=[],ReturnMostProbable=False):
    """
    Creates corner plots with or without the "corner" package, based on the preferences of a user, in order to
    be used for publications

    Args:
        SessionName: name of the current session for sake of storing the MCMC model and user input
        Parameters: list of the parameters over which the inference is being performed
        GridType: name of the grid that is being used
        BurnIn: number of steps to remove from the beginning of the chain to remove the steps the walkers took
            from the starting positions towards the distribution they settled into
        PlotChains: boolean to determine if the chains for each parameter should be plotted
        CornerPackagePlot: boolean to determine if the "corner" package should be used or not
        PlotGivenValues: boolean to determine if given values should be plotted, if this is true GivenValues should
            be given to the function
        GivenValues: list of given values to plot over the results
        ReturnMostProbable: boolean to determine if the most probable values should be returned to the user or not
    """
    if PlotGivenValues and (GivenValues == [] or len(Parameters) != len(GivenValues)):
        print("GivenValues must have the same number of entries as Parameters, if you set PlotGivenValues to True."
              "Set Values to -1 if you wish a Parameter to not have a value value plotted for it.")
        return
    FileName = SessionName[:-3]
    SaveName = '../results/' + FileName[FileName.find('/',3):]
    session = {}
    if os.path.isfile(FileName + ".bak"):
        with shelve.open(FileName) as loadedData:
            for key in loadedData.keys():
                session[key] = loadedData[key]


    GridFile = "../data/" + GridType + "_Grid.csv"
    Grid = pd.read_csv(GridFile, delimiter=",/", engine='python')
    GridDictionary = Grid.set_index('Parameter')["Grid"].to_dict()
    for P in GridDictionary.keys():
        GridDictionary[P] = (np.array(GridDictionary[P].split(",")).astype(np.float))
    if PlotGivenValues:
        for i in range(len(Parameters)):
            if GivenValues[i] != -1:
                GivenValues[i] = np.asarray((np.abs(GridDictionary[Parameters[i]] - GivenValues[i])).argmin())
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
            currentSample = np.asarray(
                [[
                    (np.abs(GridDictionary[Parameters[i]] - k)).argmin() for k in samples[j, :, i]
                 ] for j in range(len(samples[:, 0, i]))
                ]
            )
            ax.plot(currentSample, "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            currentRange = [np.abs(GridDictionary[Parameters[i]] - session[Parameters[i]]).argmin(),
                            np.abs(GridDictionary[Parameters[i]] - session[Parameters[i]+"_up"]).argmin()]
            if currentRange[0] != 0:
                currentRange[0] -= 1
            if currentRange[1] < GridDictionary[Parameters[i]].max():
                currentRange[1] += 1

            ax.set_ylim(currentRange[0], currentRange[1])
            ax.set_ylabel(OutputParameterDic[Parameters[i]])
            yticks = ax.get_yticks()
            if Parameters[i] == 'finalDens' or Parameters[i] == 'rout':
                Yticks = np.asarray(['e' + "{:.2f}".format(np.log10(GridDictionary[Parameters[i]][int(k)])) if
                                     k < len(GridDictionary[Parameters[i]]) else None for k in yticks])
                ax.set_yticklabels(Yticks)
            else:
                Yticks = np.asarray([GridDictionary[Parameters[i]][int(k)] if
                                     k < len(GridDictionary[Parameters[i]]) else None for k in yticks])
                ax.set_yticklabels(Yticks)
            if PlotGivenValues and GivenValues[i] != -1:
                ax.axhline(GivenValues[i], color='b', linestyle='solid', linewidth=3, alpha=0.2)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
        plt.savefig(SaveName + "_Chain.png")
        plt.close(fig)
    if CornerPackagePlot:
        fig = corner.corner(flat_samples, labels=Parameters)
        fig.savefig(SaveName + "_Corner.png")
        plt.close(fig)
    else:
        fig, axs = plt.subplots(ndim, ndim, figsize=(10, 10))
        MostProbableParameters = []
        for i in range(0, ndim):
            currentYSample = np.asarray([(np.abs(GridDictionary[Parameters[i]] - k)).argmin() for k in flat_samples[:,i]])
            currentYRanges = [np.abs(GridDictionary[Parameters[i]] - session[Parameters[i]]).argmin(),
                              np.abs(GridDictionary[Parameters[i]] - session[Parameters[i]+"_up"]).argmin()]
            if currentYRanges[0] != 0:
                currentYRanges[0] -= 1
            if currentYRanges[1] < GridDictionary[Parameters[i]].max():
                currentYRanges[1] += 1
            currentYBins = np.linspace(currentYRanges[0]-0.5, currentYRanges[1]+0.5,
                                       int(currentYRanges[1] +2 - currentYRanges[0]))
            weights = np.ones_like(currentYSample) / len(currentYSample)
            for j in range(i + 1):
                axs[i, j].autoscale_on = False
                currentXSample = np.asarray([(np.abs(GridDictionary[Parameters[j]] - k)).argmin() for k in flat_samples[:, j]])
                currentXRanges = [np.abs(GridDictionary[Parameters[j]] - session[Parameters[j]]).argmin(),
                                  np.abs(GridDictionary[Parameters[j]] - session[Parameters[j]+"_up"]).argmin()]
                if currentXRanges[0] != 0:
                    currentXRanges[0] -= 1
                if currentXRanges[1] < GridDictionary[Parameters[j]].max():
                    currentXRanges[1] += 1
                currentXBins = np.linspace(currentXRanges[0]-0.5, currentXRanges[1]+0.5,
                                           int(currentXRanges[1] + 2 - currentXRanges[0]))
                if j == 0 and j == i:
                    yAxisLabel = "P(" + OutputParameterDic[Parameters[j]] + ")"
                elif j == 0:
                    yAxisLabel = OutputParameterDic[Parameters[i]]
                else:
                    yAxisLabel = ''
                if i + 1 == ndim:
                    if Parameters[j] == 'finalDens' and Parameters[j] == 'rout':
                        xAxisLabel = "log("+OutputParameterDic[Parameters[j]]+")"
                    else:
                        xAxisLabel = OutputParameterDic[Parameters[j]]
                else:
                    xAxisLabel = ''
                if j == i:
                    hist, edges = np.histogram(currentYSample, weights=weights, bins=currentYBins)
                    MostProbableParameters += [GridDictionary[Parameters[i]][int(currentYRanges[0] + hist.argmax())]]
                    axs[i, j].hist(currentYSample, weights=weights, bins=currentYBins)
                    if PlotGivenValues and GivenValues[j] != -1:
                        axs[i, j].axvline(GivenValues[j] - 0.5, color='k', linestyle='dashed', linewidth=1, alpha=0.5)
                        axs[i, j].axvline(GivenValues[j] + 0.5, color='k', linestyle='dashed', linewidth=1, alpha=0.5)
                    if ReturnMostProbable:
                        axs[i, j].axvline(int(currentYRanges[0] + hist.argmax()),
                            color='r', linestyle='dashed', linewidth=1, alpha=0.5)
                    if i + 1 != ndim:
                        axs[i, j].xaxis.set_ticks([])
                    else:
                        xTicks = axs[i, j].get_xticks()
                        axs[i, j].xaxis.set_ticks(xTicks)
                        if Parameters[i] == 'finalDens' or Parameters[i] == 'rout':
                            TickValues = np.asarray(
                                ['e' + "{:.2f}".format(np.log10(GridDictionary[Parameters[i]][int(k)])) if
                                 k < len(GridDictionary[Parameters[i]]) else None for k in xTicks])
                        else:
                            TickValues = np.asarray([GridDictionary[Parameters[i]][int(k)] if
                                                     k < len(GridDictionary[Parameters[i]]) else None
                                                     for k in xTicks])
                        axs[i, j].set_xticklabels(TickValues.copy())
                    axs[i, j].yaxis.set_ticks([])
                    axs[i, j].set_xlim(xmin=currentYBins.min(), xmax=currentYBins.max())
                    axs[i, j].set_ylim(ymin=0, ymax=hist.max()+0.1)
                else:
                    if i == 1:
                        h, x, y, image = axs[i, j].hist2d(currentXSample, currentYSample,
                                                          bins=[currentXBins, currentYBins], weights=weights)
                    else:
                        axs[i, j].hist2d(currentXSample, currentYSample,
                                         bins=[currentXBins, currentYBins], weights=weights)
                    if PlotGivenValues and GivenValues[j] != -1:
                        axs[i, j].axvline(GivenValues[j] - 0.5, color='w', linestyle='dashed', linewidth=1, alpha=0.5)
                        axs[i, j].axvline(GivenValues[j] + 0.5, color='w', linestyle='dashed', linewidth=1, alpha=0.5)
                    if PlotGivenValues and GivenValues[i] != -1:
                        axs[i, j].axhline(GivenValues[i] - 0.5, color='w', linestyle='dashed', linewidth=1, alpha=0.5)
                        axs[i, j].axhline(GivenValues[i] + 0.5, color='w', linestyle='dashed', linewidth=1, alpha=0.5)
                    if i + 1 != ndim:
                        axs[i, j].xaxis.set_ticks([])
                    else:
                        xTicks = axs[i, j].get_xticks()
                        axs[i, j].xaxis.set_ticks(xTicks)
                        if Parameters[j] == 'finalDens' or Parameters[j] == 'rout':
                            TickValues = np.asarray(
                                ['e' + "{:.2f}".format(np.log10(GridDictionary[Parameters[j]][int(k)])) if
                                                     k < len(GridDictionary[Parameters[j]]) else None for k in
                                 xTicks])
                        else:
                            TickValues = np.asarray([GridDictionary[Parameters[j]][int(k)] if
                                                     k < len(GridDictionary[Parameters[j]]) else None for k in xTicks])
                        axs[i, j].set_xticklabels(TickValues.copy())

                    if (j != 0 and j + 1 != ndim) or i == 0 or (j + 1 == ndim and i + 1 == ndim):
                        axs[i, j].yaxis.set_ticks([])
                    else:
                        yTicks = axs[i, j].get_yticks()
                        axs[i, j].yaxis.set_ticks(yTicks)
                        if Parameters[i] == 'finalDens' or Parameters[i] == 'rout':
                            TickValues = np.asarray(
                                ['e' + "{:.2f}".format(np.log10(GridDictionary[Parameters[i]][int(k)])) if
                                                     k < len(GridDictionary[Parameters[i]]) else None for k in
                                 yTicks])
                        else:
                            TickValues = np.asarray([GridDictionary[Parameters[i]][int(k)] if
                                                     k < len(GridDictionary[Parameters[i]]) else None for k in yTicks])
                        axs[i, j].set_yticklabels(TickValues.copy())
                    axs[i, j].set_ylim(ymin=currentYBins.min(), ymax=currentYBins.max())
                    axs[i, j].set_xlim(xmin=currentXBins.min(), xmax=currentXBins.max())
                axs[i, j].set_ylabel(yAxisLabel)
                axs[i, j].set_xlabel(xAxisLabel)
        for i in range(ndim):
            for j in range(ndim):
                if j > i:
                    axs[i, j].remove()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(image, cax=cbar_ax)
        plt.savefig(SaveName+".png")
        plt.close(fig)
    if ReturnMostProbable:
        return MostProbableParameters
    return


def MakeSyntheticData(ChangingParameters, LinesOfInterestDict, StandardDeviationSpread=0.1, WithError=False):
    """
    Creates synthetic data and applies noise to it

    Args:
        ChangingParameters: list of changing parameters
        LinesOfInterestDict: dictionary of each species of interest and the lines those species are emitting
        StandardDeviationSpread: the standard deviation to use for adding noise to the synthetic data
        WithError: boolean to determine if error bars should be included or not
    """
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
    if not all(ReturnLines) in UCLParamOut.columns:
        for i in ReturnLines:
            if i not in UCLParamOut.columns:
                UCLParamOut[i] = 0.0
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


def MakeRotationData(ChangingParameters, LinesOfInterestDict):
    """
    This function creates and returns values needed to create a rotation diagram like plot

    Args:
        ChangingParameters: list of changing parameters
        LinesOfInterestDict: dictionary of each species of interest and the lines those species are emitting
    """
    BaseDictionary = {"phase": 1, "switch": 1, "collapse": 1, "readAbunds": 0, "writeStep": 1, "points": 1, "desorb": 1,
                      "finalOnly": "True", "fr": 1.0}
    UCLCHEMDict = {**BaseDictionary, **ChangingParameters}
    for key in GUI.ParameterDefaults.keys():
        if key not in UCLCHEMDict:
            UCLCHEMDict[key] = GUI.ParameterDefaults[key]
    ParamDF, ChemDF = utils.UCLChemDataFrames(UCLChemDict=UCLCHEMDict, Queue=False)
    UCLParamOut = utils.RadexForGrid(UCLChemDict=UCLCHEMDict, UCLParamDF=ParamDF,
                                     UCLChemDF=ChemDF, Queue=False, RotDia=True)
    ReturnLines = []
    for chems in LinesOfInterestDict.keys():
        chemical = utils.chemDat(chems)[:-4]
        for line in LinesOfInterestDict[chems]:
            ChemicalLine = chemical + '_' + line
            ReturnLines += [ChemicalLine + '_Eu', ChemicalLine + '_T_r']
    ReturnLines = np.asarray(ReturnLines)
    return UCLParamOut[ReturnLines]
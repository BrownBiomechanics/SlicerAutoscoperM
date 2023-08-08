import contextlib
import glob
import logging
import os
import time
import zipfile

import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

#
# AutoscoperM
#


class AutoscoperM(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AutoscoperM"
        self.parent.categories = [
            "Tracking",
        ]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Anthony Lombardi (Kitware)",
            "Amy M Morton (Brown University)",
            "Bardiya Akhbari (Brown University)",
            "Beatriz Paniagua (Kitware)",
            "Jean-Christophe Fillion-Robin (Kitware)",
        ]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AutoscoperM">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def downloadAndExtract(source):
    try:
        logic = slicer.modules.SampleDataWidget.logic
    except AttributeError:
        import SampleData

        logic = SampleData.SampleDataLogic()

    logic.downloadFromSource(source)

    cache_dir = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()
    logic.logMessage(f"<b>Extracting archive</b> <i>{source.fileNames[0]}<i/> into {cache_dir} ...</b>")

    # Unzip the downloaded file
    with zipfile.ZipFile(os.path.join(cache_dir, source.fileNames[0]), "r") as zip_ref:
        zip_ref.extractall(cache_dir)

    logic.logMessage("<b>Done</b>")


def registerAutoscoperSampleData(dataType, version, checksum):
    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Tracking",
        sampleName=f"AutoscoperM - {dataType} BVR",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images"
        # set to "Single".
        thumbnailFileName=os.path.join(iconsPath, f"{dataType}.png"),
        # Download URL and target file name
        uris=f"https://github.com/BrownBiomechanics/Autoscoper/releases/download/sample-data/{version}-{dataType}.zip",
        fileNames=f"{version}-{dataType}.zip",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums=checksum,
        # This node name will be used when the data set is loaded
        # nodeNames=f"AutoscoperM - {dataType} BVR" # comment this line so the data is not loaded into the scene
        customDownloader=downloadAndExtract,
    )


def sampleDataConfigFile(dataType):
    """Return the trial config filename."""
    return {
        "2023-08-01-Wrist": "2023-07-20-Wrist.cfg",
        "2023-08-01-Knee": "2023-07-26-Knee.cfg",
        "2023-08-01-Ankle": "2023-07-20-Ankle.cfg",
    }.get(dataType)


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    registerAutoscoperSampleData(
        "Wrist", "2023-08-01", checksum="SHA256:86a914ec822d88d3cbd70135ac77212207856c71a244d18b0e150f246f0e8ab2"
    )
    registerAutoscoperSampleData(
        "Knee", "2023-08-01", checksum="SHA256:ffdba730e8792ee8797068505ae502ed6edafe26e70597ff10a2e017a4162767"
    )
    registerAutoscoperSampleData(
        "Ankle", "2023-08-01", checksum="SHA256:9e666e0dbca0c556072d2c9c18f4ddc74bfb328b98668c7f65347e4746431e33"
    )


#
# AutoscoperMWidget
#


class AutoscoperMWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AutoscoperM.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AutoscoperMLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # NA

        # Buttons
        self.ui.startAutoscoper.connect("clicked(bool)", self.lookupAndStartAutoscoper)
        self.ui.closeAutoscoper.connect("clicked(bool)", self.logic.stopAutoscoper)
        self.ui.loadConfig.connect("clicked(bool)", self.onLoadConfig)

        # Sample Data Buttons
        self.ui.wristSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Wrist"))
        self.ui.kneeSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Knee"))
        self.ui.ankleSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Ankle"))

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, _caller, _event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller, _event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # NA

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(
            self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode
        ):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, _caller=None, _event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        # NA

        # Update buttons states and tooltips
        # NA

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, _caller=None, _event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # NA

        self._parameterNode.EndModify(wasModified)

    def lookupAndStartAutoscoper(self):
        """Lookup autoscoper executable and start a new process

        This call waits that the process has been started and returns.
        """
        import shutil

        executablePath = shutil.which("autoscoper")
        if not executablePath:
            logging.error("autoscoper executable not found")
            return
        self.logic.startAutoscoper(executablePath)

    def onLoadConfig(self):
        self.loadConfig(self.ui.configSelector.currentPath)

    def loadConfig(self, configPath):
        if not configPath.endswith(".cfg"):
            logging.error(f"Failed to load config file: {configPath} is expected to have the .cfg extension")
            return

        if not os.path.exists(configPath):
            logging.error(f"Failed to load config file: {configPath} not found")
            return

        self.logic.AutoscoperSocket.loadTrial(configPath)

    def onSampleDataButtonClicked(self, dataType):

        # Ensure that the sample data is installed
        slicerCacheDir = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()
        sampleDataDir = os.path.join(slicerCacheDir, dataType)
        if not os.path.exists(sampleDataDir):
            logging.error(
                f"Sample data not found. Please install the {dataType} sample data set using the Sample Data module."
            )
            return

        # Ensure that autoscoper is running
        if self.logic.AutoscoperProcess.state() != qt.QProcess.Running and slicer.util.confirmYesNoDisplay(
            "Autoscoper is not running. Do you want to start Autoscoper?"
        ):
            self.lookupAndStartAutoscoper()

        if self.logic.AutoscoperProcess.state() != qt.QProcess.Running:
            logging.error("failed to load the Sample Data: Autoscoper is not running. ")
            return

        # Load the sample data
        configFile = os.path.join(sampleDataDir, sampleDataConfigFile(dataType))

        if not os.path.exists(configFile):
            logging.error(f"Failed to load config file: {configFile} not found")
            return

        self.loadConfig(configFile)

        # Load filter settings
        numCams = len(glob.glob(os.path.join(sampleDataDir, "Calibration", "*.txt")))
        filterSettings = os.path.join(sampleDataDir, "xParameters", "control_settings.vie")
        for cam in range(numCams):
            self.logic.AutoscoperSocket.loadFilters(cam, filterSettings)


#
# AutoscoperMLogic
#


class AutoscoperMLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

        self.AutoscoperProcess = qt.QProcess()
        self.AutoscoperProcess.setProcessChannelMode(qt.QProcess.ForwardedChannels)
        self.AutoscoperSocket = None

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        pass

    def connectToAutoscoper(self):
        """Connect to a running instance of Autoscoper."""

        if self.AutoscoperProcess.state() != qt.QProcess.Running:
            logging.error("failed to connect to Autoscoper: The process is not running")
            return

        try:
            from PyAutoscoper.connect import AutoscoperConnection
        except ImportError:
            slicer.util.pip_install("PyAutoscoper~=2.0.0")
            from PyAutoscoper.connect import AutoscoperConnection

        self.AutoscoperSocket = AutoscoperConnection()
        logging.info("connection to Autoscoper is established")

    def disconnectFromAutoscoper(self):
        """Disconnect from a running instance of Autoscoper."""
        if self.AutoscoperSocket is None:
            logging.warning("connection to Autoscoper is not established")
            return
        self.AutoscoperSocket.closeConnection()
        time.sleep(0.5)
        self.AutoscoperSocket = None
        logging.info("Autoscoper is disconnected from 3DSlicer")

    def startAutoscoper(self, executablePath):
        """Start Autoscoper executable in a new process

        This call waits the process has been started and returns.
        """
        if not os.path.exists(executablePath):
            logging.error(f"Specified executable {executablePath} does not exist")
            return

        if self.AutoscoperProcess.state() in [qt.QProcess.Starting, qt.QProcess.Running]:
            logging.error("Autoscoper executable already started")
            return

        @contextlib.contextmanager
        def changeCurrentDir(directory):
            currentDirectory = os.getcwd()
            try:
                os.chdir(directory)
                yield
            finally:
                os.chdir(currentDirectory)

        executableDirectory = os.path.dirname(executablePath)

        with changeCurrentDir(executableDirectory):
            logging.info(f"Starting Autoscoper {executablePath}")
            self.AutoscoperProcess.setProgram(executablePath)
            self.AutoscoperProcess.start()
            self.AutoscoperProcess.waitForStarted()

        slicer.app.processEvents()

        time.sleep(4)  # wait for autoscoper to boot up before connecting

        # Since calling "time.sleep()" prevents Slicer application from being
        # notified when the QProcess state changes (e.g Autoscoper is closed while
        # Slicer as asleep waiting), we are calling waitForFinished() explicitly
        # to ensure that the QProcess state is up-to-date.
        self.AutoscoperProcess.waitForFinished(1)

        self.connectToAutoscoper()

    def stopAutoscoper(self):
        """Stop Autoscoper process"""
        if self.AutoscoperProcess.state() == qt.QProcess.NotRunning:
            logging.error("Autoscoper executable is not running")
            return

        if self.AutoscoperSocket:
            self.disconnectFromAutoscoper()

        self.AutoscoperProcess.kill()

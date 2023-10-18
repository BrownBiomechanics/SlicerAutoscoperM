import contextlib
import glob
import logging
import os
import shutil
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

from AutoscoperMLib import IO, RadiographGeneration, SubVolumeExtraction

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

        # Pre-processing Library Buttons
        self.ui.tiffGenButton.connect("clicked(bool)", self.onGeneratePartialVolumes)
        self.ui.vrgGenButton.connect("clicked(bool)", self.onGenerateVRG)
        self.ui.configGenButton.connect("clicked(bool)", self.onGenerateConfig)
        self.ui.segmentationButton.connect("clicked(bool)", self.onSegmentation)

        self.ui.loadPVButton.connect("clicked(bool)", self.onLoadPV)

        # Default output directory
        self.ui.mainOutputSelector.setCurrentPath(
            os.path.join(slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory(), "AutoscoperM-Pre-Processing")
        )

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
            return False

        if not os.path.exists(configPath):
            logging.error(f"Failed to load config file: {configPath} not found")
            return False

        # Ensure that autoscoper is running
        if self.logic.AutoscoperProcess.state() != qt.QProcess.Running and slicer.util.confirmYesNoDisplay(
            "Autoscoper is not running. Do you want to start Autoscoper?"
        ):
            self.lookupAndStartAutoscoper()

        if self.logic.AutoscoperProcess.state() != qt.QProcess.Running:
            logging.error("failed to load the Sample Data: Autoscoper is not running. ")
            return False

        self.logic.AutoscoperSocket.loadTrial(configPath)

        return True

    def onSampleDataButtonClicked(self, dataType):

        # Ensure that the sample data is installed
        slicerCacheDir = slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory()
        sampleDataDir = os.path.join(slicerCacheDir, dataType)
        if not os.path.exists(sampleDataDir):
            logging.error(
                f"Sample data not found. Please install the {dataType} sample data set using the Sample Data module."
            )
            return

        # Load the sample data
        configFile = os.path.join(sampleDataDir, sampleDataConfigFile(dataType))

        if not os.path.exists(configFile):
            logging.error(f"Failed to load config file: {configFile} not found")
            return

        if not self.loadConfig(configFile):
            return

        # Load filter settings
        numCams = len(glob.glob(os.path.join(sampleDataDir, "Calibration", "*.txt")))
        filterSettings = os.path.join(sampleDataDir, "xParameters", "control_settings.vie")
        for cam in range(numCams):
            self.logic.AutoscoperSocket.loadFilters(cam, filterSettings)

    def onGeneratePartialVolumes(self):
        """
        This function creates partial volumes for each segment in the segmentation node for the selected volume node.
        """
        volumeNode = self.ui.volumeSelector.currentNode()
        if not volumeNode:
            logging.error("Failed to generate partial volume: no volume selected")
            return
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        if mainOutputDir is None or mainOutputDir == "":
            logging.error("Failed to generate partial volume: no output directory selected")
            return
        if not os.path.exists(mainOutputDir):
            os.makedirs(mainOutputDir)
        tiffSubDir = self.ui.tiffSubDir.text
        tfmSubDir = self.ui.tfmSubDir.text
        segmentationNode = self.ui.MRMLNodeComboBox.currentNode()
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setMaximum(100)
        self.logic.saveSubVolumesFromSegmentation(
            volumeNode,
            segmentationNode,
            mainOutputDir,
            volumeSubDir=tiffSubDir,
            transformSubDir=tfmSubDir,
            progressCallback=self.updateProgressBar,
        )

    def onGenerateVRG(self):
        """
        NOTE - This function is not currently used. It is a work in progress.

        This function optimizes the camera positions for a given volume and then
        generates a VRG file for each optimized camera.
        """

        self.updateProgressBar(0)

        volumeNode = self.ui.volumeSelector.currentNode()
        if not volumeNode:
            logging.error("Failed to generate VRG: no volume selected")
            return
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        if mainOutputDir is None or mainOutputDir == "":
            logging.error("Failed to generate VRG: no output directory selected")
            return
        if not os.path.exists(mainOutputDir):
            os.makedirs(mainOutputDir)

        width = self.ui.vrgRes_width.value
        height = self.ui.vrgRes_height.value

        nPossibleCameras = self.ui.posCamSpin.value
        nOptimizedCameras = self.ui.optCamSpin.value

        if nPossibleCameras < nOptimizedCameras:
            logging.error("Failed to generate VRG: more optimized cameras than possible cameras")
            return

        newVolumeNode = SubVolumeExtraction.automaticExtraction(
            volumeNode,
            self.ui.vrgGen_ThresholdSpinBox.value,
            segmentationName="Full Extraction",
            progressCallback=self.updateProgressBar,
            maxProgressValue=10,
        )
        bounds = [0, 0, 0, 0, 0, 0]
        newVolumeNode.GetBounds(bounds)

        # rename the volume node
        newVolumeNode.SetName(volumeNode.GetName() + " - Bone Subvolume")
        self.logic.removeFolderByName(volumeNode.GetName() + " split")

        # Generate all possible camera positions
        logging.info(f"Generating {nPossibleCameras} possible cameras...")
        camOffset = self.ui.camOffSetSpin.value
        cameras = RadiographGeneration.generateNCameras(nPossibleCameras, bounds, camOffset, [width, height])

        self.updateProgressBar(13)

        # Generate initial VRG for each camera
        tmpDir = self.ui.vrgTempDir.text
        for i, cam in enumerate(cameras):
            logging.info(f"Generating VRG for camera {i}")
            cameraDir = os.path.join(mainOutputDir, tmpDir, f"cam{cam.id}")
            if not os.path.exists(cameraDir):
                os.makedirs(cameraDir)
            RadiographGeneration.generateVRG(cam, newVolumeNode, os.path.join(cameraDir, "1.tif"), width, height)
            self.updateProgressBar(((i + 1) / nPossibleCameras) * 29 + 13)

        # Want to find the best nOptimizedCameras cameras that have the best DID
        bestCameras = RadiographGeneration.optimizeCameras(
            cameras, os.path.join(mainOutputDir, tmpDir), nOptimizedCameras, progressCallback=self.updateProgressBar
        )

        # Generate the camera calibration files and VRGs for the best cameras
        cameraSubDir = self.ui.cameraSubDir.text
        vrgSubDir = self.ui.vrgSubDir.text
        for i, cam in enumerate(bestCameras):
            logging.info(f"Generating camera calibration file and VRG for camera {i}")
            if not os.path.exists(os.path.join(mainOutputDir, cameraSubDir)):
                os.makedirs(os.path.join(mainOutputDir, cameraSubDir))
            IO.generateCameraCalibrationFile(cam, os.path.join(mainOutputDir, cameraSubDir, f"cam{cam.id}.yaml"))
            cameraDir = os.path.join(mainOutputDir, vrgSubDir, f"cam{cam.id}")
            if not os.path.exists(cameraDir):
                os.makedirs(cameraDir)
            # Copy the VRG to the final directory
            shutil.copyfile(
                os.path.join(mainOutputDir, tmpDir, f"cam{cam.id}", "1.tif"), os.path.join(cameraDir, "1.tif")
            )
            progress = ((i + 1) / nOptimizedCameras) * 29 + 71
            self.updateProgressBar(progress)

        # remove temp directory
        if self.ui.removeVrgTmp.isChecked():
            shutil.rmtree(os.path.join(mainOutputDir, tmpDir))

        # remove subvolume node
        # slicer.mrmlScene.RemoveNode(newVolumeNode)

    def onGenerateConfig(self):
        """
        NOTE - This function is not currently in use

        Generates a complete config file (including all partial volumes, radiographs,
        and camera calibration files) for Autoscoper.
        """
        volumeNode = self.ui.volumeSelector.currentNode()
        if not volumeNode:
            logging.error("Failed to generate config: no volume selected")
            return
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        if mainOutputDir is None or mainOutputDir == "":
            logging.error("Failed to generate config: no output directory selected")
            return
        if not os.path.exists(mainOutputDir):
            os.makedirs(mainOutputDir)

        trialName = self.ui.trialName.text

        width = self.ui.vrgRes_width.value
        height = self.ui.vrgRes_height.value

        optimizationOffsets = [
            self.ui.optOffX.value,
            self.ui.optOffY.value,
            self.ui.optOffZ.value,
            self.ui.optOffYaw.value,
            self.ui.optOffPitch.value,
            self.ui.optOffRoll.value,
        ]
        volumeFlip = [
            int(self.ui.flipX.isChecked()),
            int(self.ui.flipY.isChecked()),
            int(self.ui.flipZ.isChecked()),
        ]

        # Validate the directory structure
        tiffSubDir = self.ui.tiffSubDir.text
        vrgSubDir = self.ui.vrgSubDir.text
        calibrationSubDir = self.ui.cameraSubDir.text

        if not os.path.exists(os.path.join(mainOutputDir, tiffSubDir)):
            logging.error(f"Failed to generate config file: {tiffSubDir} not found")
            return
        if not os.path.exists(os.path.join(mainOutputDir, vrgSubDir)):
            logging.error(f"Failed to generate config file: {vrgSubDir} not found")
            return
        if not os.path.exists(os.path.join(mainOutputDir, calibrationSubDir)):
            logging.error(f"Failed to generate config file: {calibrationSubDir} not found")
            return

        # generate the config file
        logging.info("Generating config file")
        configFilePath = IO.generateConfigFile(
            mainOutputDir,
            [tiffSubDir, vrgSubDir, calibrationSubDir],
            trialName,
            volumeFlip=volumeFlip,
            voxelSize=volumeNode.GetSpacing(),
            renderResolution=[int(width / 2), int(height / 2)],
            optimizationOffsets=optimizationOffsets,
        )

        self.ui.configSelector.setCurrentPath(configFilePath)

    def onSegmentation(self):
        """
        Either launches the automatic segmentation process or loads in a set of segmentations from a directory
        """

        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setMaximum(100)

        volumeNode = self.ui.volumeSelector.currentNode()

        if not volumeNode:
            logging.error("No volume selected")
            return

        if self.ui.segGen_autoRadioButton.isChecked():
            segmentationNode = SubVolumeExtraction.automaticSegmentation(
                volumeNode,
                self.ui.segGen_ThresholdSpinBox.value,
                self.ui.segGen_marginSizeSpin.value,
                progressCallback=self.updateProgressBar,
            )
        elif self.ui.segGen_fileRadioButton.isChecked():
            segmentationFileDir = self.ui.segGen_lineEdit.currentPath
            if not segmentationFileDir or not os.path.exists(segmentationFileDir):
                logging.error("No segmentation directory selected")
                return
            segmentationFiles = glob.glob(os.path.join(segmentationFileDir, "*.*"))
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.CreateDefaultDisplayNodes()
            for i, file in enumerate(segmentationFiles):
                returnedNode = IO.loadSegmentation(segmentationNode, file)
                if returnedNode:
                    # get the segment from the returned node and add it to the segmentation node
                    segment = returnedNode.GetSegmentation().GetNthSegment(0)
                    segmentationNode.GetSegmentation().AddSegment(segment)
                    slicer.mrmlScene.RemoveNode(returnedNode)
                self.ui.progressBar.setValue((i + 1) / len(segmentationFiles) * 100)
        else:  # Should never happen but just in case
            logging.error("No segmentation method selected")
            return

    def updateProgressBar(self, value):
        """
        Progress bar callback function for use with AutoscoperMLib functions
        """
        self.ui.progressBar.setValue(value)
        slicer.app.processEvents()

    def onLoadPV(self):

        mainOutputDir = self.ui.mainOutputSelector.currentPath
        volumeSubDir = self.ui.tiffSubDir.text
        transformSubDir = self.ui.tfmSubDir.text

        vols = glob.glob(os.path.join(mainOutputDir, volumeSubDir, "*.tif"))
        tfms = glob.glob(os.path.join(mainOutputDir, transformSubDir, "*.tfm"))

        if len(vols) != len(tfms):
            logging.error("Number of volumes and transforms do not match")
            return

        if len(vols) == 0:
            logging.error("No data found")
            return

        for vol, tfm in zip(vols, tfms):
            volumeNode = slicer.util.loadVolume(vol)
            transformNode = slicer.util.loadTransform(tfm)
            volumeNode.SetAndObserveTransformNodeID(transformNode.GetID())
            self.logic.showVolumeIn3D(volumeNode)


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

    def saveSubVolumesFromSegmentation(
        self,
        volumeNode: slicer.vtkMRMLVolumeNode,
        segmentationNode: slicer.vtkMRMLSegmentationNode,
        outputDir: str,
        volumeSubDir: str = "Volumes",
        transformSubDir: str = "Transforms",
        progressCallback=None,
    ) -> bool:
        """
        Save subvolumes from segmentation to outputDir

        :param volumeNode: volume node
        :type volumeNode: slicer.vtkMRMLVolumeNode
        :param segmentationNode: segmentation node
        :type segmentationNode: slicer.vtkMRMLSegmentationNode
        :param outputDir: output directory
        :type outputDir: str
        :param progressCallback: progress callback, defaults to None
        :type progressCallback: callable, optional
        """

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        if not progressCallback:
            logging.warning(
                "[AutoscoperM.logic.saveSubVolumesFromSegmentation] "
                "No progress callback provided, progress bar will not be updated"
            )

            def progressCallback(x):
                return x

        segmentIDs = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
        numSegments = segmentIDs.GetNumberOfValues()
        for i in range(numSegments):
            segmentID = segmentIDs.GetValue(i)
            segmentName = segmentationNode.GetSegmentation().GetSegment(segmentID).GetName()
            segmentVolume = SubVolumeExtraction.extractSubVolume(volumeNode, segmentationNode, segmentID)
            segmentVolume.SetName(segmentName)
            filename = os.path.join(outputDir, volumeSubDir, segmentName + ".tif")
            IO.castVolumeForTIFF(segmentVolume)
            IO.writeVolume(segmentVolume, filename)
            spacing = segmentVolume.GetSpacing()
            origin = segmentVolume.GetOrigin()
            filename = os.path.join(outputDir, transformSubDir, segmentName + ".tfm")
            IO.writeTFMFile(filename, [1, 1, spacing[2]], origin)
            self.showVolumeIn3D(segmentVolume)
            # update progress bar
            progressCallback((i + 1) / numSegments * 100)
        # Set the  volumeNode to be the active volume
        slicer.app.applicationLogic().GetSelectionNode().SetActiveVolumeID(volumeNode.GetID())
        # Reset the slice field of views
        slicer.app.layoutManager().resetSliceViews()
        return True

    def showVolumeIn3D(self, volumeNode: slicer.vtkMRMLVolumeNode):
        logic = slicer.modules.volumerendering.logic()
        displayNode = logic.CreateVolumeRenderingDisplayNode()
        displayNode.UnRegister(logic)
        slicer.mrmlScene.AddNode(displayNode)
        volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())
        logic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
        slicer.mrmlScene.RemoveNode(slicer.util.getNode("Volume rendering ROI"))

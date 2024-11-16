import contextlib
import glob
import logging
import os
import shutil
import time
import zipfile
from typing import Optional, Union

import qt
import slicer
import vtk
import vtkAddon
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from AutoscoperMLib import IO, SubVolumeExtraction

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
        self.autoscoperExecutables = {}

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
        self.ui.startAutoscoper.connect("clicked(bool)", self.startAutoscoper)
        self.ui.closeAutoscoper.connect("clicked(bool)", self.logic.stopAutoscoper)
        self.ui.loadConfig.connect("clicked(bool)", self.onLoadConfig)

        # Lookup Autoscoper executables
        for backend in ["CUDA", "OpenCL"]:
            executableName = AutoscoperMWidget.autoscoperExecutableName(backend)
            logging.info(f"Looking up '{executableName}' executable")
            path = shutil.which(executableName)
            if path:
                self.autoscoperExecutables[backend] = path
                logging.info(f"Found '{path}'")
            else:
                logging.info("No executable found")

        if not self.autoscoperExecutables:
            logging.error("Failed to lookup autoscoper executables")

        # Available Autoscoper backends
        self.ui.autoscoperRenderingBackendComboBox.addItems(list(self.autoscoperExecutables.keys()))

        # Sample Data Buttons
        self.ui.wristSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Wrist"))
        self.ui.kneeSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Knee"))
        self.ui.ankleSampleButton.connect("clicked(bool)", lambda: self.onSampleDataButtonClicked("2023-08-01-Ankle"))

        # Pre-processing Library Buttons
        self.ui.tiffGenButton.connect("clicked(bool)", self.onGeneratePartialVolumes)
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

    @property
    def autoscoperExecutableToLaunchBackend(self):
        return self.ui.autoscoperRenderingBackendComboBox.currentText

    @autoscoperExecutableToLaunchBackend.setter
    def autoscoperExecutableToLaunchBackend(self, value):
        self.ui.autoscoperRenderingBackendComboBox.currentText = value

    @staticmethod
    def autoscoperExecutableName(backend=None):
        """Returns the Autoscoper executable name to lookup given a backend name."""
        suffix = f"-{backend}" if backend else ""
        return f"autoscoper{suffix}"

    def startAutoscoper(self):
        """Start a new process using the Autoscoper executable corresponding to the selected backend.

        This call waits that the process has been started and returns.
        """
        try:
            executablePath = self.autoscoperExecutables[self.autoscoperExecutableToLaunchBackend]
        except KeyError:
            logging.error("Autoscoper executable not found")
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
            self.startAutoscoper()

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
        with slicer.util.tryWithErrorDisplay("Failed to compute results", waitCursor=True):
            volumeNode = self.ui.volumeSelector.currentNode()
            mainOutputDir = self.ui.mainOutputSelector.currentPath
            tiffSubDir = self.ui.tiffSubDir.text
            tfmSubDir = self.ui.tfmSubDir.text
            trackingSubDir = self.ui.trackingSubDir.text
            modelSubDir = self.ui.modelSubDir.text
            segmentationNode = self.ui.pv_SegNodeComboBox.currentNode()
            if not self.logic.validateInputs(
                volumeNode=volumeNode,
                segmentationNode=segmentationNode,
                mainOutputDir=mainOutputDir,
                volumeSubDir=tiffSubDir,
                transformSubDir=tfmSubDir,
                trackingSubDir=trackingSubDir,
                modelSubDir=modelSubDir,
            ):
                raise ValueError("Invalid inputs")
                return

            self.logic.createPathsIfNotExists(
                mainOutputDir,
                os.path.join(mainOutputDir, tiffSubDir),
                os.path.join(mainOutputDir, tfmSubDir),
                os.path.join(mainOutputDir, trackingSubDir),
                os.path.join(mainOutputDir, modelSubDir),
            )
            self.ui.progressBar.setValue(0)
            self.ui.progressBar.setMaximum(100)
            self.logic.saveSubVolumesFromSegmentation(
                volumeNode,
                segmentationNode,
                mainOutputDir,
                volumeSubDir=tiffSubDir,
                transformSubDir=tfmSubDir,
                trackingSubDir=trackingSubDir,
                modelSubDir=modelSubDir,
                progressCallback=self.updateProgressBar,
            )
            # Load TIFFs and Transforms back for visualization
            self.onLoadPV()
            # onLoadPV has a call to the "success" display, remove the one here so the user doesn't get two.

    def onGenerateConfig(self):
        """
        Generates a complete config file (including all partial volumes, radiographs,
        and camera calibration files) for Autoscoper.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results", waitCursor=True):
            volumeNode = self.ui.volumeSelector.currentNode()
            mainOutputDir = self.ui.mainOutputSelector.currentPath
            trialName = self.ui.trialName.text
            width = self.ui.vrgRes_width.value
            height = self.ui.vrgRes_height.value

            tiffSubDir = self.ui.tiffSubDir.text
            vrgSubDir = self.ui.vrgSubDir.text
            calibrationSubDir = self.ui.cameraSubDir.text

            # Validate the inputs
            if not self.logic.validateInputs(
                volumeNode=volumeNode,
                mainOutputDir=mainOutputDir,
                trialName=trialName,
                width=width,
                height=height,
                volumeSubDir=tiffSubDir,
                vrgSubDir=vrgSubDir,
                calibrationSubDir=calibrationSubDir,
            ):
                raise ValueError("Invalid inputs")
                return
            if not self.logic.validatePaths(
                mainOutputDir=mainOutputDir,
                tiffDir=os.path.join(mainOutputDir, tiffSubDir),
                vrgDir=os.path.join(mainOutputDir, vrgSubDir),
                calibDir=os.path.join(mainOutputDir, calibrationSubDir),
            ):
                raise ValueError("Invalid paths")
                return

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

            voxel_spacing = self.logic.GetVolumeSpacing(volumeNode)
            # generate the config file
            configFilePath = IO.generateConfigFile(
                mainOutputDir,
                [tiffSubDir, vrgSubDir, calibrationSubDir],
                trialName,
                volumeFlip=volumeFlip,
                voxelSize=voxel_spacing,
                renderResolution=[int(width / 2), int(height / 2)],
                optimizationOffsets=optimizationOffsets,
            )

            self.ui.configSelector.setCurrentPath(configFilePath)
        slicer.util.messageBox("Success!")

    def onSegmentation(self):
        """
        Either launches the automatic segmentation process or loads in a set of segmentations from a directory
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results", waitCursor=True):
            self.ui.progressBar.setValue(0)
            self.ui.progressBar.setMaximum(100)

            volumeNode = self.ui.volumeSelector.currentNode()

            if not self.logic.validateInputs(voluemNode=volumeNode):
                raise ValueError("Invalid inputs")
                return

            if self.ui.segGen_autoRadioButton.isChecked():
                currentVolumeNode = volumeNode
                numFrames = 1
                if self.logic.IsSequenceVolume(volumeNode):
                    numFrames = volumeNode.GetNumberOfDataNodes()
                    currentVolumeNode = self.logic.getItemInSequence(volumeNode, 0)
                    segmentationSequenceNode = self.logic.createSequenceNodeInBrowser(
                        nodename=f"{volumeNode.GetName()}_Segmentation", sequenceNode=volumeNode
                    )
                for i in range(numFrames):
                    self.logic.cleanFilename(currentVolumeNode.GetName(), i)
                    segmentationNode = SubVolumeExtraction.automaticSegmentation(
                        currentVolumeNode,
                        self.ui.segGen_ThresholdSpinBox.value,
                        self.ui.segGen_marginSizeSpin.value,
                        progressCallback=self.updateProgressBar,
                    )
                    progress = (i + 1) / numFrames * 100
                    self.ui.progressBar.setValue(progress)
                    if self.logic.IsSequenceVolume(volumeNode):
                        segmentationSequenceNode.SetDataNodeAtValue(segmentationNode, str(i))
                        slicer.mrmlScene.RemoveNode(segmentationNode)
                        currentVolumeNode = self.logic.getNextItemInSequence(volumeNode)
            elif self.ui.segGen_fileRadioButton.isChecked():
                segmentationFileDir = self.ui.segGen_lineEdit.currentPath
                if not self.logic.validatePaths(segmentationFileDir=segmentationFileDir):
                    raise ValueError("Invalid paths")
                    return
                segmentationFiles = glob.glob(os.path.join(segmentationFileDir, "*.*"))
                segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
                segmentationNode.CreateDefaultDisplayNodes()
                for idx, file in enumerate(segmentationFiles):
                    returnedNode = IO.loadSegmentation(segmentationNode, file)
                    if returnedNode:
                        # get the segment from the returned node and add it to the segmentation node
                        segment = returnedNode.GetSegmentation().GetNthSegment(0)
                        segmentationNode.GetSegmentation().AddSegment(segment)
                        slicer.mrmlScene.RemoveNode(returnedNode)
                    self.ui.progressBar.setValue((idx + 1) / len(segmentationFiles) * 100)
            else:  # Should never happen but just in case
                raise Exception("No segmentation method selected")
                return
        slicer.util.messageBox("Success!")

    def updateProgressBar(self, value):
        """
        Progress bar callback function for use with AutoscoperMLib functions
        """
        self.ui.progressBar.setValue(value)
        slicer.app.processEvents()

    def onLoadPV(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            mainOutputDir = self.ui.mainOutputSelector.currentPath
            volumeSubDir = self.ui.tiffSubDir.text
            transformSubDir = self.ui.tfmSubDir.text

            # Check number of generated scale and translation transform files matches the number of volumes
            vols = glob.glob(os.path.join(mainOutputDir, volumeSubDir, "*.tif"))
            tfms_t = glob.glob(os.path.join(mainOutputDir, transformSubDir, "*_t.tfm"))
            tfms_scale = glob.glob(os.path.join(mainOutputDir, transformSubDir, "*_scale.tfm"))

            if len(vols) == 0:
                raise Exception("No data found")
                return

            if len(vols) != len(tfms_t) != len(tfms_scale):
                raise ValueError(
                    "Volume TFM mismatch, missing scale or translation tfm files! "
                    f"vols: ({len(vols)}) != tfms_t: ({len(tfms_t)}) != tfms_scale: ({len(tfms_scale)})"
                )
                return

            # get the IJK to RAS direction matrix from original input volume
            parentVolume = self.ui.volumeSelector.currentNode()
            parentIJKToRAS = vtk.vtkMatrix4x4()
            parentVolume.GetIJKToRASDirectionMatrix(parentIJKToRAS)

            # check 3 transform files have been generated (translation, scale and combined)
            # and load only the combined scale and translation transform for each generated partial volume
            for i in range(len(vols)):
                nodeName = os.path.splitext(os.path.basename(vols[i]))[0]
                volumeNode = slicer.util.loadVolume(vols[i])
                # ensure we maintain the original RAS/LPS directions from the parent volume
                volumeNode.SetIJKToRASDirectionMatrix(parentIJKToRAS)

                translationTransformFileName = os.path.join(mainOutputDir, transformSubDir, f"{nodeName}_t.tfm")
                scaleTranformFileName = os.path.join(mainOutputDir, transformSubDir, f"{nodeName}_scale.tfm")
                transformFileName = os.path.join(mainOutputDir, transformSubDir, f"{nodeName}.tfm")
                if not os.path.exists(translationTransformFileName):
                    raise ValueError(
                        f"Failed to load partial volume {nodeName}: "
                        "Corresponding translation transform file {translationTransformFileName} not found"
                    )
                if not os.path.exists(scaleTranformFileName):
                    raise ValueError(
                        f"Failed to load partial volume {nodeName}: "
                        "Corresponding scaling transform file {scaleTranformFileName} not found"
                    )
                if not os.path.exists(transformFileName):
                    raise ValueError(
                        f"Failed to load partial volume {nodeName}: "
                        "Corresponding combined transform file {transformFileName} not found"
                    )
                transformNode = slicer.util.loadTransform(transformFileName)

                volumeNode.SetAndObserveTransformNodeID(transformNode.GetID())
                self.logic.showVolumeIn3D(volumeNode)

        slicer.util.messageBox("Success!")


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

    @staticmethod
    def IsSequenceVolume(node: Union[slicer.vtkMRMLNode, None]) -> bool:
        return isinstance(node, slicer.vtkMRMLSequenceNode)

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
        trackingSubDir: str = "Tracking",
        modelSubDir: str = "Models",
        progressCallback: Optional[callable] = None,
    ) -> bool:
        """
        Save subvolumes from segmentation to outputDir

        :param volumeNode: volume node
        :param segmentationNode: segmentation node
        :param outputDir: output directory
        :param progressCallback: progress callback, defaults to None
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

        tfmFiles = glob.glob(os.path.join(outputDir, transformSubDir, "*.tfm"))
        tfms = [tfm if os.path.basename(tfm).split(".")[0] == "Origin2Dicom" else None for tfm in tfmFiles]
        try:
            origin2DicomTransformFile = next(item for item in tfms if item is not None)
        except StopIteration:
            origin2DicomTransformFile = None

        for idx in range(numSegments):
            segmentID = segmentIDs.GetValue(idx)
            segmentName = segmentationNode.GetSegmentation().GetSegment(segmentID).GetName()
            segmentVolume = SubVolumeExtraction.extractSubVolume(volumeNode, segmentationNode, segmentID)
            segmentVolume.SetName(segmentName)
            tifFilename = os.path.join(outputDir, volumeSubDir, segmentName + ".tif")
            IO.castVolumeForTIFF(segmentVolume)

            # Remove spacing before writing volume out
            segmentVolume.DisableModifiedEventOn()
            originalSpacing = segmentVolume.GetSpacing()
            segmentVolume.SetSpacing([1.0, 1.0, 1.0])
            IO.writeVolume(segmentVolume, tifFilename)
            # Restore spacing
            segmentVolume.SetSpacing(originalSpacing)
            segmentVolume.DisableModifiedEventOff()

            transformFilenameBase = os.path.join(outputDir, transformSubDir, segmentName)
            origin = segmentVolume.GetOrigin()
            IO.writeTFMFile(f"{transformFilenameBase}_t.tfm", [1.0, 1.0, 1.0], origin)
            spacing = segmentVolume.GetSpacing()
            IO.writeTFMFile(f"{transformFilenameBase}_scale.tfm", spacing, [0.0, 0.0, 0.0])
            IO.writeTFMFile(f"{transformFilenameBase}.tfm", spacing, origin)

            # Create PVOL2AUT transform
            pvol2autNode = self.createAndAddPVol2AutTransformNode(segmentVolume)
            pvol2autFilename = os.path.join(outputDir, transformSubDir, f"{segmentVolume.GetName()}-PVOL2AUT.tfm")
            slicer.util.saveNode(pvol2autNode, pvol2autFilename)

            # Create DICOM2AUT transform
            dicom2autNode = self.createAndAddDicom2AutTransformNode(origin, pvol2autNode)
            dicom2autFilename = os.path.join(outputDir, transformSubDir, f"{segmentVolume.GetName()}-DICOM2AUT.tfm")
            slicer.util.exportNode(dicom2autNode, dicom2autFilename)

            stlFilename = os.path.join(outputDir, modelSubDir, f"AUT_{segmentVolume.GetName()}.stl")
            self.exportSTLFromSegment(segmentationNode, segmentID, stlFilename, dicom2autNode.GetTransformToParent())

            slicer.mrmlScene.RemoveNode(dicom2autNode)

            # Create TRA file
            tfm = vtk.vtkMatrix4x4()
            tfm.SetElement(0, 3, origin[0])
            tfm.SetElement(1, 3, origin[1])
            tfm.SetElement(2, 3, origin[2])

            if origin2DicomTransformFile is not None:
                origin2DicomNode = self.loadTransformFromFile(origin2DicomTransformFile)
                origin2DicomNode.Inverse()
                tfm = self.applyOrigin2DicomTransform(tfm, origin2DicomNode)
                slicer.mrmlScene.RemoveNode(origin2DicomNode)

            tfm = self.applyPVol2AutTransform(tfm, pvol2autNode)
            slicer.mrmlScene.RemoveNode(pvol2autNode)

            tfmR = vtk.vtkMatrix3x3()
            vtkAddon.vtkAddonMathUtilities.GetOrientationMatrix(tfm, tfmR)

            # Apply RAS to LPS transform
            RAS2LPS = vtk.vtkMatrix3x3()
            RAS2LPS.SetElement(0, 0, -1)
            RAS2LPS.SetElement(1, 1, -1)

            vtk.vtkMatrix3x3.Multiply3x3(tfmR, RAS2LPS, tfmR)
            vtkAddon.vtkAddonMathUtilities.SetOrientationMatrix(tfmR, tfm)

            # Save TRA file
            filename = os.path.join(outputDir, trackingSubDir, f"{segmentName}.tra")
            IO.writeTRA(filename, [tfm])

            # update progress bar
            progressCallback((idx + 1) / numSegments * 100)

            # Remove the segment volume
            slicer.mrmlScene.RemoveNode(segmentVolume)
        # Set the  volumeNode to be the active volume
        slicer.app.applicationLogic().GetSelectionNode().SetActiveVolumeID(volumeNode.GetID())
        # Reset the slice field of views
        slicer.app.layoutManager().resetSliceViews()
        return True

    @staticmethod
    def exportSTLFromSegment(
        segmentationNode: slicer.vtkMRMLSegmentationNode,
        segmentID: str,
        filename: str,
        transform: Optional[vtk.vtkAbstractTransform] = None,
    ):
        """Utility functions for exporting a segment as an STL. Optionally takes a vtk transform."""
        if not segmentationNode.CreateClosedSurfaceRepresentation():
            raise ValueError(
                f"Failed to generate the closed surface representation from segmentation: {segmentationNode.GetName()}"
            )

        polyData = vtk.vtkPolyData()
        if not segmentationNode.GetClosedSurfaceRepresentation(segmentID, polyData):
            raise ValueError(f"Failed to get PolyData for segmentationNode {segmentationNode.GetName()}")

        if transform is not None:
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(polyData)
            transformFilter.SetTransform(transform)
            transformFilter.Update()
            polyData = transformFilter.GetOutput()

        stlNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        stlNode.SetAndObservePolyData(polyData)

        slicer.util.exportNode(stlNode, filename)
        slicer.mrmlScene.RemoveNode(stlNode)

    @staticmethod
    def showVolumeIn3D(volumeNode: slicer.vtkMRMLVolumeNode):
        logic = slicer.modules.volumerendering.logic()
        displayNode = logic.CreateVolumeRenderingDisplayNode()
        displayNode.UnRegister(logic)
        slicer.mrmlScene.AddNode(displayNode)
        volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())
        logic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
        slicer.mrmlScene.RemoveNode(slicer.util.getNode("Volume rendering ROI"))

    @staticmethod
    def validateInputs(*args: tuple, **kwargs: dict) -> bool:
        """
        Validates that the provided inputs are not None.

        :param args: list of inputs to validate
        :param kwargs: list of inputs to validate

        :return: True if all inputs are valid, False otherwise
        """
        statuses = []
        for arg in args:
            status = True
            if arg is None:
                logging.error(f"{arg} is None")
                status = False
            if isinstance(arg, str) and arg == "":
                logging.error(f"{arg} is an empty string")
                status = False
            statuses.append(status)

        for name, arg in kwargs.items():
            status = True
            if arg is None:
                logging.error(f"{name} is None")
                status = False
            if isinstance(arg, str) and arg == "":
                logging.error(f"{name} is an empty string")
                status = False
            statuses.append(status)

        return all(statuses)

    @staticmethod
    def validatePaths(*args: tuple, **kwargs: dict) -> bool:
        """
        Checks that the provided paths exist.

        :param args: list of paths to validate
        :param kwargs: list of paths to validate

        :return: True if all paths exist, False otherwise
        """
        statuses = []
        for arg in args:
            status = True
            if not os.path.exists(arg):
                logging.error(f"{arg} does not exist")
                status = False
            statuses.append(status)

        for name, path in kwargs.items():
            status = True
            if not os.path.exists(path):
                logging.error(f"{name} ({path}) does not exist")
                status = False
            statuses.append(status)

        return all(statuses)

    @staticmethod
    def createPathsIfNotExists(*args: tuple) -> None:
        """
        Creates a path if it does not exist.

        :param args: list of paths to create
        """
        for arg in args:
            if not os.path.exists(arg):
                os.makedirs(arg)

    @staticmethod
    def extractSubVolumeForVRG(
        volumeNode: slicer.vtkMRMLVolumeNode,
        segmentationNode: slicer.vtkMRMLSegmentationNode,
        cameraDebugMode: bool = False,
    ) -> tuple[vtk.vtkImageData, list[float]]:
        """
        Extracts a subvolume from the volumeNode that contains all of the segments in the segmentationNode

        :param volumeNode: volume node
        :param segmentationNode: segmentation node
        :param cameraDebugMode: Whether or not to keep the extracted volume in the scene, defaults to False

        :return: tuple containing the extracted volume and the bounds of the volume
        """
        mergedSegmentationNode = SubVolumeExtraction.mergeSegments(volumeNode, segmentationNode)
        newVolumeNode = SubVolumeExtraction.extractSubVolume(
            volumeNode, mergedSegmentationNode, mergedSegmentationNode.GetSegmentation().GetNthSegmentID(0)
        )
        newVolumeNode.SetName(volumeNode.GetName() + " - Bone Subvolume")

        bounds = [0, 0, 0, 0, 0, 0]
        newVolumeNode.GetBounds(bounds)

        # Copy the metadata from the original volume into the ImageData
        newVolumeImageData = vtk.vtkImageData()
        newVolumeImageData.DeepCopy(newVolumeNode.GetImageData())  # So we don't modify the original volume
        newVolumeImageData.SetSpacing(newVolumeNode.GetSpacing())
        origin = list(newVolumeNode.GetOrigin())
        origin[0:2] = [x * -1 for x in origin[0:2]]
        newVolumeImageData.SetOrigin(origin)

        # Ensure we are in the correct orientation (RAS vs LPS)
        imageReslice = vtk.vtkImageReslice()
        imageReslice.SetInputData(newVolumeImageData)

        axes = vtk.vtkMatrix4x4()
        axes.Identity()
        axes.SetElement(0, 0, -1)
        axes.SetElement(1, 1, -1)

        imageReslice.SetResliceAxes(axes)
        imageReslice.Update()
        newVolumeImageData = imageReslice.GetOutput()

        if not cameraDebugMode:
            slicer.mrmlScene.RemoveNode(newVolumeNode)
            slicer.mrmlScene.RemoveNode(mergedSegmentationNode)

        return newVolumeImageData, bounds

    @staticmethod
    def getItemInSequence(sequenceNode: slicer.vtkMRMLSequenceNode, idx: int) -> slicer.vtkMRMLNode:
        """
        Returns the item at the specified index in the sequence node

        :param sequenceNode: sequence node
        :param idx: index

        :return: item at the specified index
        """
        if not AutoscoperMLogic.IsSequenceVolume(sequenceNode):
            raise Exception("[AutoscoperM.logic.getItemInSequence] sequenceNode must be a sequence node")
            return None

        if idx >= sequenceNode.GetNumberOfDataNodes():
            raise Exception(f"[AutoscoperM.logic.getItemInSequence] index {idx} is out of range")
            return None

        browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(sequenceNode)
        browserNode.SetSelectedItemNumber(idx)
        return browserNode.GetProxyNode(sequenceNode), sequenceNode.GetNthDataNode(idx).GetName()

    @staticmethod
    def getNextItemInSequence(sequenceNode: slicer.vtkMRMLSequenceNode) -> slicer.vtkMRMLNode:
        """
        Returns the next item in the sequence

        :param sequenceNode: sequence node

        :return: next item in the sequence
        """
        if not AutoscoperMLogic.IsSequenceVolume(sequenceNode):
            raise Exception("[AutoscoperM.logic.getNextItemInSequence] sequenceNode must be a sequence node")

        browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(sequenceNode)
        browserNode.SelectNextItem()
        idx = browserNode.GetSelectedItemNumber()
        return browserNode.GetProxyNode(sequenceNode), sequenceNode.GetNthDataNode(idx).GetName()

    @staticmethod
    def cleanFilename(volumeName: str, index: Optional[int] = None) -> str:
        filename = slicer.qSlicerCoreIOManager().forceFileNameValidCharacters(volumeName.replace(" ", "_"))
        return f"{index:03d}_{filename}" if index is not None else filename

    @staticmethod
    def createSequenceNodeInBrowser(nodename, sequenceNode):
        if not AutoscoperMLogic.IsSequenceVolume(sequenceNode):
            raise Exception("[AutoscoperMLogic.createSequenceNodeInBrowser] sequenceNode must be a sequence node")

        browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(sequenceNode)
        newSeqenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", nodename)
        browserNode.AddSynchronizedSequenceNode(newSeqenceNode)
        browserNode.SetOverwriteProxyName(newSeqenceNode, True)
        browserNode.SetSaveChanges(newSeqenceNode, True)
        return newSeqenceNode

    @staticmethod
    def GetVolumeSpacing(node: Union[slicer.vtkMRMLVolumeNode, slicer.vtkMRMLSequenceNode]) -> list[float]:
        if AutoscoperMLogic.IsSequenceVolume(node):
            return AutoscoperMLogic.getItemInSequence(node, 0)[0].GetSpacing()
        return node.GetSpacing()

    @staticmethod
    def loadTransformFromFile(transformFileName: str) -> slicer.vtkMRMLLinearTransformNode:
        return slicer.util.loadNodeFromFile(transformFileName)

    @staticmethod
    def applyOrigin2DicomTransform(
        transform: vtk.vtkMatrix4x4,
        origin2DicomTransformNode: slicer.vtkMRMLLinearTransformNode,
    ) -> vtk.vtkMatrix4x4:
        """Utility function for converting a transform between the dicom centered and
        world centered coordinate systems."""
        origin2DicomTransformMatrix = vtk.vtkMatrix4x4()
        origin2DicomTransformNode.GetMatrixTransformToParent(origin2DicomTransformMatrix)

        vtk.vtkMatrix4x4.Multiply4x4(origin2DicomTransformMatrix, transform, transform)

        return transform

    @staticmethod
    def applyPVol2AutTransform(
        transform: vtk.vtkMatrix4x4,
        pVol2AutNode: slicer.vtkMRMLLinearTransformNode,
    ) -> vtk.vtkMatrix4x4:
        """Utility function for converting a transform between the Slicer and Autoscoper coordinate systems."""
        pvol2aut = vtk.vtkMatrix4x4()
        pVol2AutNode.GetMatrixTransformToParent(pvol2aut)

        # Extract the rotation matrices so we are not affecting the translation vector
        transformR = vtk.vtkMatrix3x3()
        vtkAddon.vtkAddonMathUtilities.GetOrientationMatrix(transform, transformR)

        pvol2autR = vtk.vtkMatrix3x3()
        vtkAddon.vtkAddonMathUtilities.GetOrientationMatrix(pvol2aut, pvol2autR)

        vtk.vtkMatrix3x3.Multiply3x3(pvol2autR, transformR, transformR)

        vtkAddon.vtkAddonMathUtilities.SetOrientationMatrix(transformR, transform)

        # Apply the translation vector
        for i in range(3):
            transform.SetElement(i, 3, transform.GetElement(i, 3) + pvol2aut.GetElement(i, 3))

        return transform

    @staticmethod
    def createAndAddPVol2AutTransformNode(
        volumeNode: slicer.vtkMRMLVolumeNode,
    ) -> slicer.vtkMRMLLinearTransformNode:
        """Utility function for creating a slicer2Autoscoper transform for the given volume."""
        bounds = [0] * 6
        volumeNode.GetRASBounds(bounds)
        volSize = [abs(bounds[i + 1] - bounds[i]) for i in range(0, len(bounds), 2)]

        pVol2Aut = vtk.vtkMatrix4x4()
        pVol2Aut.Identity()
        # Rotation matrix for a 180 x-axis rotation
        pVol2Aut.SetElement(1, 1, -pVol2Aut.GetElement(1, 1))
        pVol2Aut.SetElement(2, 2, -pVol2Aut.GetElement(2, 2))
        pVol2Aut.SetElement(1, 3, -volSize[1])  # Offset -Y

        pVol2AutNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
        pVol2AutNode.SetMatrixTransformToParent(pVol2Aut)
        return pVol2AutNode

    @staticmethod
    def createAndAddDicom2AutTransformNode(
        origin: list[float], pvol2autNode: slicer.vtkMRMLLinearTransformNode
    ) -> slicer.vtkMRMLLinearTransformNode:
        """Utility function that creates and adds a DICOM2AUT transform node"""
        dicom2aut = vtk.vtkMatrix4x4()
        dicom2aut.SetElement(0, 3, -origin[0])
        dicom2aut.SetElement(1, 3, origin[1])
        dicom2aut.SetElement(2, 3, origin[2])
        dicom2aut = AutoscoperMLogic.applyPVol2AutTransform(dicom2aut, pvol2autNode)

        dicom2autNode = slicer.vtkMRMLLinearTransformNode()
        dicom2autNode.SetMatrixTransformToParent(dicom2aut)
        slicer.mrmlScene.AddNode(dicom2autNode)
        return dicom2autNode

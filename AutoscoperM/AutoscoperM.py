import contextlib
import glob
import logging
import os
import shutil
import time
import zipfile
from typing import Optional

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
        self.parent.dependencies = [
            "CalculateDataIntensityDensity",
            "VirtualRadiographGeneration",
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
        self.ui.vrgGenButton.connect("clicked(bool)", self.onGenerateVRG)
        self.ui.manualVRGGenButton.connect("clicked(bool)", self.onManualVRGGen)
        self.ui.configGenButton.connect("clicked(bool)", self.onGenerateConfig)
        self.ui.segmentationButton.connect("clicked(bool)", self.onSegmentation)

        self.ui.loadPVButton.connect("clicked(bool)", self.onLoadPV)

        # Default output directory
        self.ui.mainOutputSelector.setCurrentPath(
            os.path.join(slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory(), "AutoscoperM-Pre-Processing")
        )

        # Dynamic camera frustum functions
        self.ui.mVRG_markupSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMarkupNodeChanged)
        self.ui.mVRG_ClippingRangeSlider.connect("valuesChanged(double,double)", self.updateClippingRange)
        self.ui.mVRG_viewAngleSpin.connect("valueChanged(int)", self.updateViewAngle)

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
        volumeNode = self.ui.volumeSelector.currentNode()
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        tiffSubDir = self.ui.tiffSubDir.text
        tfmSubDir = self.ui.tfmSubDir.text
        segmentationNode = self.ui.pv_SegNodeComboBox.currentNode()
        if not self.logic.validateInputs(
            volumeNode=volumeNode,
            segmentationNode=segmentationNode,
            mainOutputDir=mainOutputDir,
            volumeSubDir=tiffSubDir,
            transformSubDir=tfmSubDir,
        ):
            raise ValueError("Invalid inputs")
            return

        self.logic.createPathsIfNotExists(
            mainOutputDir, os.path.join(mainOutputDir, tiffSubDir), os.path.join(mainOutputDir, tfmSubDir)
        )
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
        This function optimizes the camera positions for a given volume and then
        generates a VRG file for each optimized camera.
        """

        self.updateProgressBar(0)

        # Set up and validate inputs
        volumeNode = self.ui.volumeSelector.currentNode()
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        width = self.ui.vrgRes_width.value
        height = self.ui.vrgRes_height.value
        nPossibleCameras = self.ui.posCamSpin.value
        nOptimizedCameras = self.ui.optCamSpin.value
        tmpDir = self.ui.vrgTempDir.text
        cameraSubDir = self.ui.cameraSubDir.text
        vrgSubDir = self.ui.vrgSubDir.text
        if not self.logic.validateInputs(
            volumeNode=volumeNode,
            mainOutputDir=mainOutputDir,
            width=width,
            height=height,
            nPossibleCameras=nPossibleCameras,
            nOptimizedCameras=nOptimizedCameras,
            tmpDir=tmpDir,
            cameraSubDir=cameraSubDir,
            vrgSubDir=vrgSubDir,
        ):
            raise ValueError("Invalid inputs")
            return
        if not self.logic.validatePaths(mainOutputDir=mainOutputDir):
            raise ValueError("Invalid paths")
            return
        if nPossibleCameras < nOptimizedCameras:
            logging.error("Failed to generate VRG: more optimized cameras than possible cameras")
            return

        bounds = [0] * 6
        volumeNode.GetBounds(bounds)

        # Generate all possible camera positions
        camOffset = self.ui.camOffSetSpin.value
        cameras = RadiographGeneration.generateNCameras(
            nPossibleCameras, bounds, camOffset, [width, height], self.ui.camDebugCheckbox.isChecked()
        )

        self.updateProgressBar(10)

        # Generate initial VRG for each camera
        self.logic.generateVRGForCameras(
            cameras,
            volumeNode,
            os.path.join(mainOutputDir, tmpDir),
            width,
            height,
            progressCallback=self.updateProgressBar,
        )

        # Optimize the camera positions
        bestCameras = RadiographGeneration.optimizeCameras(
            cameras, os.path.join(mainOutputDir, tmpDir), nOptimizedCameras, progressCallback=self.updateProgressBar
        )

        # Move the optimized VRGs to the final directory and generate the camera calibration files
        self.logic.generateCameraCalibrationFiles(
            bestCameras,
            os.path.join(mainOutputDir, tmpDir),
            os.path.join(mainOutputDir, vrgSubDir),
            os.path.join(mainOutputDir, cameraSubDir),
            progressCallback=self.updateProgressBar,
        )

        # Clean Up
        if self.ui.removeVrgTmp.isChecked():
            shutil.rmtree(os.path.join(mainOutputDir, tmpDir))

    def onGenerateConfig(self):
        """
        Generates a complete config file (including all partial volumes, radiographs,
        and camera calibration files) for Autoscoper.
        """
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

        # generate the config file
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

        if not self.logic.validateInputs(voluemNode=volumeNode):
            raise ValueError("Invalid inputs")
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

    def onManualVRGGen(self):
        markupsNode = self.ui.mVRG_markupSelector.currentNode()
        volumeNode = self.ui.volumeSelector.currentNode()
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        viewAngle = self.ui.mVRG_viewAngleSpin.value
        clippingRange = (self.ui.mVRG_ClippingRangeSlider.minimumValue, self.ui.mVRG_ClippingRangeSlider.maximumValue)
        width = self.ui.vrgRes_width.value
        height = self.ui.vrgRes_height.value
        vrgDir = self.ui.vrgSubDir.text
        cameraDir = self.ui.cameraSubDir.text
        if not self.logic.validateInputs(
            markupsNode=markupsNode,
            volumeNode=volumeNode,
            mainOutputDir=mainOutputDir,
            viewAngle=viewAngle,
            clippingRange=clippingRange,
            width=width,
            height=height,
            vrgDir=vrgDir,
            cameraDir=cameraDir,
        ):
            logging.error("Failed to generate VRG: invalid inputs")
            return
        if not self.logic.validatePaths(mainOutputDir=mainOutputDir):
            logging.error("Failed to generate VRG: invalid output directory")
            return
        self.logic.createPathsIfNotExists(os.path.join(mainOutputDir, vrgDir), os.path.join(mainOutputDir, cameraDir))

        if self.logic.vrgManualCameras is None:
            self.onMarkupNodeChanged(markupsNode)  # create the cameras

        self.logic.generateVRGForCameras(
            self.logic.vrgManualCameras,
            volumeNode,
            os.path.join(mainOutputDir, vrgDir),
            width,
            height,
            progressCallback=self.updateProgressBar,
        )

        self.updateProgressBar(100)

        for cam in self.logic.vrgManualCameras:
            IO.generateCameraCalibrationFile(cam, os.path.join(mainOutputDir, cameraDir, f"cam{cam.id}.json"))

    def onMarkupNodeChanged(self, node):
        if node is None:
            if self.logic.vrgManualCameras is not None:
                # clean up
                for cam in self.logic.vrgManualCameras:
                    slicer.mrmlScene.RemoveNode(cam.FrustumModel)
                self.logic.vrgManualCameras = None
            return
        if self.logic.vrgManualCameras is not None:
            # clean up
            for cam in self.logic.vrgManualCameras:
                slicer.mrmlScene.RemoveNode(cam.FrustumModel)
            self.logic.vrgManualCameras = None
        # get the volume nodes
        volumeNode = self.ui.volumeSelector.currentNode()
        self.logic.validateInputs(volumeNode=volumeNode)
        bounds = [0] * 6
        volumeNode.GetBounds(bounds)
        self.logic.vrgManualCameras = RadiographGeneration.generateCamerasFromMarkups(
            node,
            bounds,
            (self.ui.mVRG_ClippingRangeSlider.minimumValue, self.ui.mVRG_ClippingRangeSlider.maximumValue),
            self.ui.mVRG_viewAngleSpin.value,
            [self.ui.vrgRes_width.value, self.ui.vrgRes_height.value],
            True,
        )

    def updateClippingRange(self, min, max):
        for cam in self.logic.vrgManualCameras:
            cam.vtkCamera.SetClippingRange(min, max)
            RadiographGeneration._updateFrustumModel(cam)

    def updateViewAngle(self, value):
        for cam in self.logic.vrgManualCameras:
            cam.vtkCamera.SetViewAngle(value)
            RadiographGeneration._updateFrustumModel(cam)


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
        self.vrgManualCameras = None

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
        for idx in range(numSegments):
            segmentID = segmentIDs.GetValue(idx)
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
            progressCallback((idx + 1) / numSegments * 100)
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

    def validateInputs(self, *args: tuple, **kwargs: dict) -> bool:
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

    def validatePaths(self, *args: tuple, **kwargs: dict) -> bool:
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

    def createPathsIfNotExists(self, *args: tuple) -> None:
        """
        Creates a path if it does not exist.

        :param args: list of paths to create
        """
        for arg in args:
            if not os.path.exists(arg):
                os.makedirs(arg)

    def extractSubVolumeForVRG(
        self,
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

    def generateVRGForCameras(
        self,
        cameras: list[RadiographGeneration.Camera],
        volumeNode: slicer.vtkMRMLVolumeNode,
        outputDir: str,
        width: int,
        height: int,
        progressCallback: Optional[callable] = None,
    ) -> None:
        """
        Generates VRG files for each camera in the cameras list

        :param cameras: list of cameras
        :param volumeNode: volume node
        :param outputDir: output directory
        :param width: width of the radiographs
        :param height: height of the radiographs
        :param progressCallback: progress callback, defaults to None
        """
        self.createPathsIfNotExists(outputDir)

        if not progressCallback:
            logging.warning(
                "[AutoscoperM.logic.generateVRGForCameras] "
                "No progress callback provided, progress bar will not be updated"
            )

            def progressCallback(x):
                return x

        # Apply a thresh of 0 to the volume to remove air from the volume
        thresholdScalarVolume = slicer.modules.thresholdscalarvolume
        parameters = {
            "InputVolume": volumeNode.GetID(),
            "OutputVolume": volumeNode.GetID(),
            "ThresholdValue": 0,
            "ThresholdType": "Below",
            "Lower": 0,
        }
        slicer.cli.runSync(thresholdScalarVolume, None, parameters)

        # write a temporary volume to disk
        volumeFileName = "AutoscoperM_VRG_GEN_TEMP.mhd"
        IO.writeTemporyFile(volumeFileName, self.convertNodeToData(volumeNode))

        # Execute CLI for each camera
        cliModule = slicer.modules.virtualradiographgeneration
        cliNodes = []
        for cam in cameras:
            cameraDir = os.path.join(outputDir, f"cam{cam.id}")
            self.createPathsIfNotExists(cameraDir)
            camera = cam.vtkCamera
            parameters = {
                "inputVolumeFileName": os.path.join(slicer.app.temporaryPath, volumeFileName),
                "cameraPosition": [camera.GetPosition()[0], camera.GetPosition()[1], camera.GetPosition()[2]],
                "cameraFocalPoint": [camera.GetFocalPoint()[0], camera.GetFocalPoint()[1], camera.GetFocalPoint()[2]],
                "cameraViewUp": [camera.GetViewUp()[0], camera.GetViewUp()[1], camera.GetViewUp()[2]],
                "cameraViewAngle": camera.GetViewAngle(),
                "clippingRange": [camera.GetClippingRange()[0], camera.GetClippingRange()[1]],
                "outputWidth": width,
                "outputHeight": height,
                "outputFileName": os.path.join(cameraDir, "1.tif"),
            }
            cliNode = slicer.cli.run(cliModule, None, parameters)  # run asynchronously
            cliNodes.append(cliNode)

        # Note: CLI nodes are currently not executed in parallel. See https://github.com/Slicer/Slicer/pull/6723
        # This just allows the UI to remain responsive while the CLI nodes are running for now.

        # Wait for all the CLI nodes to finish
        for i, cliNode in enumerate(cliNodes):
            while cliNodes[i].GetStatusString() != "Completed":
                slicer.app.processEvents()
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                # error
                errorText = cliNode.GetErrorText()
                slicer.mrmlScene.RemoveNode(cliNode)
                raise ValueError("CLI execution failed: " + errorText)
            slicer.mrmlScene.RemoveNode(cliNode)
            progress = ((i + 1) / len(cameras)) * 30 + 10
            progressCallback(progress)

    def generateCameraCalibrationFiles(
        self,
        bestCameras: list[RadiographGeneration.Camera],
        tmpDir: str,
        finalDir: str,
        calibDir: str,
        progressCallback: Optional[callable] = None,
    ) -> None:
        """
        Copies the optimized VRGs from the temporary directory to the final directory
        and generates the camera calibration files

        :param bestCameras: list of optimized cameras
        :param tmpDir: temporary directory
        :param finalDir: final directory
        :param calibDir: calibration directory
        :param progressCallback: progress callback, defaults to None
        """
        self.validatePaths(tmpDir=tmpDir)
        self.createPathsIfNotExists(finalDir, calibDir)
        if not progressCallback:
            logging.warning(
                "[AutoscoperM.logic.generateCameraCalibrationFiles] "
                "No progress callback provided, progress bar will not be updated"
            )

            def progressCallback(x):
                return x

        for idx, cam in enumerate(bestCameras):
            IO.generateCameraCalibrationFile(cam, os.path.join(calibDir, f"cam{cam.id}.json"))
            cameraDir = os.path.join(finalDir, f"cam{cam.id}")
            self.createPathsIfNotExists(cameraDir)
            # Copy all tif files from the tmp to the final directory
            for file in glob.glob(os.path.join(tmpDir, f"cam{cam.id}", "*.tif")):
                shutil.copy(file, cameraDir)

            progress = ((idx + 1) / len(bestCameras)) * 10 + 90
            progressCallback(progress)

    def convertNodeToData(self, volumeNode: slicer.vtkMRMLVolumeNode) -> vtk.vtkImageData:
        """
        Converts a volume node to a vtkImageData object
        """
        imageData = vtk.vtkImageData()
        imageData.DeepCopy(volumeNode.GetImageData())
        imageData.SetSpacing(volumeNode.GetSpacing())
        origin = list(volumeNode.GetOrigin())
        imageData.SetOrigin(origin)

        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)
        if mat.GetElement(0, 0) < 0 and mat.GetElement(1, 1) < 0:
            origin[0:2] = [x * -1 for x in origin[0:2]]
            imageData.SetOrigin(origin)

            # Ensure we are in the correct orientation (RAS vs LPS)
            imageReslice = vtk.vtkImageReslice()
            imageReslice.SetInputData(imageData)

            axes = vtk.vtkMatrix4x4()
            axes.Identity()
            axes.SetElement(0, 0, -1)
            axes.SetElement(1, 1, -1)

            imageReslice.SetResliceAxes(axes)
            imageReslice.Update()
            imageData = imageReslice.GetOutput()

        return imageData

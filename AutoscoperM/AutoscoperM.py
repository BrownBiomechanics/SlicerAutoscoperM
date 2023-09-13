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
        self.is_4d = False

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
        self.ui.manualVRGGenButton.connect("clicked(bool)", self.onManualVRGGen)
        self.ui.configGenButton.connect("clicked(bool)", self.onGenerateConfig)
        self.ui.segmentationButton.connect("clicked(bool)", self.onSegmentation)

        self.ui.loadPVButton.connect("clicked(bool)", self.onLoadPV)

        self.ui.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.check4D)

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
        if self.ui.volumeSelector.currentNode() is not None:
            self.check4D(self.ui.volumeSelector.currentNode())
        if self.ui.mVRG_markupSelector.currentNode() is not None:
            self.onMarkupNodeChanged(self.ui.mVRG_markupSelector.currentNode())

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

    def onGeneratePartialVolumes(self):
        """
        This function creates partial volumes for each segment in the segmentation node for the selected volume node.
        """
        volumeNode = self.ui.volumeSelector.currentNode()
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        tiffSubDir = self.ui.tiffSubDir.text
        tfmSubDir = self.ui.tfmSubDir.text
        segmentationNode = self.ui.pv_SegNodeComboBox.currentNode()
        self.logic.validateInputs(
            volumeNode=volumeNode,
            segmentationNode=segmentationNode,
            mainOutputDir=mainOutputDir,
            volumeSubDir=tiffSubDir,
            transformSubDir=tfmSubDir,
        )
        self.logic.createPathsIfNotExists(
            mainOutputDir, os.path.join(mainOutputDir, tiffSubDir), os.path.join(mainOutputDir, tfmSubDir)
        )
        self.logic.centerVolume(volumeNode, os.path.join(mainOutputDir, tfmSubDir), self.is_4d)

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
        segmentationNode = self.ui.vrg_SegNodeComboBox.currentNode()
        width = self.ui.vrgRes_width.value
        height = self.ui.vrgRes_height.value
        nPossibleCameras = self.ui.posCamSpin.value
        nOptimizedCameras = self.ui.optCamSpin.value
        tmpDir = self.ui.vrgTempDir.text
        cameraSubDir = self.ui.cameraSubDir.text
        vrgSubDir = self.ui.vrgSubDir.text
        tfmSubDir = self.ui.tfmSubDir.text
        self.logic.validateInputs(
            volumeNode=volumeNode,
            segmentationNode=segmentationNode,
            mainOutputDir=mainOutputDir,
            width=width,
            height=height,
            nPossibleCameras=nPossibleCameras,
            nOptimizedCameras=nOptimizedCameras,
            tmpDir=tmpDir,
            cameraSubDir=cameraSubDir,
            vrgSubDir=vrgSubDir,
        )
        self.logic.validatePaths(mainOutputDir=mainOutputDir)
        self.logic.createPathsIfNotExists(os.path.join(mainOutputDir, tfmSubDir))
        self.logic.centerVolume(volumeNode, os.path.join(mainOutputDir, tfmSubDir), self.is_4d)

        if nPossibleCameras < nOptimizedCameras:
            logging.error("Failed to generate VRG: more optimized cameras than possible cameras")
            return

        # Extract the subvolume for the radiographs
        volumeImageData, bounds = self.logic.extractSubVolumeForVRG(
            volumeNode, segmentationNode, cameraDebugMode=self.ui.camDebugCheckbox.isChecked()
        )

        # Generate all possible camera positions
        camOffset = self.ui.camOffSetSpin.value
        cameras = RadiographGeneration.generateNCameras(
            nPossibleCameras, bounds, camOffset, [width, height], self.ui.camDebugCheckbox.isChecked()
        )

        self.updateProgressBar(10)

        # Generate initial VRG for each camera
        self.logic.generateVRGForCameras(
            cameras,
            volumeImageData,
            os.path.join(mainOutputDir, tmpDir),
            [width, height],
            frameNum=1,
            progressCallback=self.updateProgressBar,
        )

        # Optimize the camera positions
        bestCameras = RadiographGeneration.optimizeCameras(
            cameras, os.path.join(mainOutputDir, tmpDir), nOptimizedCameras, progressCallback=self.updateProgressBar
        )

        # Move the optimized VRGs to the final directory and generate the camera calibration files
        self.logic.moveOptimizedVRGsAndGenCalibFiles(
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
        self.logic.validateInputs(
            volumeNode=volumeNode,
            mainOutputDir=mainOutputDir,
            trialName=trialName,
            width=width,
            height=height,
            volumeSubDir=tiffSubDir,
            vrgSubDir=vrgSubDir,
            calibrationSubDir=calibrationSubDir,
        )
        self.logic.validatePaths(
            mainOutputDir=mainOutputDir,
            tiffDir=os.path.join(mainOutputDir, tiffSubDir),
            vrgDir=os.path.join(mainOutputDir, vrgSubDir),
            calibDir=os.path.join(mainOutputDir, calibrationSubDir),
        )

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
        mainOutputDir = self.ui.mainOutputSelector.currentPath
        tfmSubDir = self.ui.tfmSubDir.text

        self.logic.validateInputs(voluemNode=volumeNode, mainOutputDir=mainOutputDir, tfmSubDir=tfmSubDir)
        self.logic.validatePaths(mainOutputDir=mainOutputDir)
        self.logic.createPathsIfNotExists(os.path.join(mainOutputDir, tfmSubDir))

        self.logic.centerVolume(volumeNode, os.path.join(mainOutputDir, tfmSubDir), self.is_4d)

        if self.ui.segGen_autoRadioButton.isChecked():
            currentVolumeNode = volumeNode
            numFrames = 1
            if self.is_4d:
                numFrames = volumeNode.GetNumberOfDataNodes()
                browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(volumeNode)
                browserNode.SetSelectedItemNumber(0)
                currentVolumeNode = browserNode.GetProxyNode(volumeNode)
                segmentationSequenceNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSequenceNode", f"{volumeNode.GetName()}_Segmentation"
                )
                browserNode.AddSynchronizedSequenceNode(segmentationSequenceNode)
                browserNode.SetOverwriteProxyName(segmentationSequenceNode, True)
                browserNode.SetSaveChanges(segmentationSequenceNode, True)
            for i in range(numFrames):
                segmentationNode = SubVolumeExtraction.automaticSegmentation(
                    currentVolumeNode,
                    self.ui.segGen_ThresholdSpinBox.value,
                    self.ui.segGen_marginSizeSpin.value,
                )
                progress = (i + 1) / numFrames * 100
                self.ui.progressBar.setValue(progress)
                if self.is_4d:
                    segmentationSequenceNode.SetDataNodeAtValue(segmentationNode, str(i))
                    slicer.mrmlScene.RemoveNode(segmentationNode)
                    currentVolumeNode = self.logic.getItemInSequence(volumeNode, i + 1)
        elif self.ui.segGen_fileRadioButton.isChecked():
            segmentationFileDir = self.ui.segGen_lineEdit.currentPath
            self.logic.validatePaths(segmentationFileDir=segmentationFileDir)
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

    def onManualVRGGen(self):
        markupsNode = self.ui.mVRG_markupSelector.currentNode()
        volumeNode = self.ui.volumeSelector.currentNode()
        segmentationNode = self.ui.mVRG_segmentationSelector.currentNode()
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
            segmentationNode=segmentationNode,
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

        # Check if the volume is centered at the origin
        bounds = [0] * 6
        if self.is_4d:
            # get the bounds of the first frame
            volumeNode.GetNthDataNode(0).GetRASBounds(bounds)
        else:
            volumeNode.GetRASBounds(bounds)

        center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        center = [round(x) for x in center]
        if center != [0, 0, 0]:
            logging.warning("Volume is not centered at the origin. This may cause issues with Autoscoper.")

        numFrames = 1
        currentNode = volumeNode
        currentSegmentationNode = segmentationNode
        if self.is_4d:
            numFrames = volumeNode.GetNumberOfDataNodes()

        for i in range(numFrames):
            if self.is_4d:
                currentNode = self.logic.getItemInSequence(volumeNode, i)
                currentSegmentationNode = self.logic.getItemInSequence(segmentationNode, i)

            volumeImageData, _ = self.logic.extractSubVolumeForVRG(
                currentNode, currentSegmentationNode, cameraDebugMode=self.ui.camDebugCheckbox.isChecked()
            )

            self.logic.generateVRGForCameras(
                self.logic.vrgManualCameras,
                volumeImageData,
                os.path.join(mainOutputDir, vrgDir),
                [width, height],
                frameNum=i,
                progressCallback=self.updateProgressBar,
            )

        self.updateProgressBar(100)

        for cam in self.logic.vrgManualCameras:
            IO.generateCameraCalibrationFile(cam, os.path.join(mainOutputDir, cameraDir, f"cam{cam.id}.yaml"))

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
        # get the volume and segmentation nodes
        segmentationNode = self.ui.mVRG_segmentationSelector.currentNode()
        if not self.logic.validateInputs(segmentationNode=segmentationNode):
            return
        bounds = [0] * 6
        if self.is_4d:
            # calculate the average bounds
            tmp = [0] * 6
            for i in range(segmentationNode.GetNumberOfDataNodes()):
                segmentationNode.GetNthDataNode(i).GetBounds(tmp)
                for j in range(6):
                    bounds[j] += tmp[j]
            for j in range(6):
                bounds[j] /= segmentationNode.GetNumberOfDataNodes()
        else:
            segmentationNode.GetRASBounds(bounds)
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

    def check4D(self, node):
        self.is_4d = type(node) == slicer.vtkMRMLSequenceNode


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

    def validateInputs(self, *args: tuple, **kwargs: dict) -> bool:
        """
        Validates that the provided inputs are not None.

        :param args: list of inputs to validate
        :type args: tuple

        :param kwargs: list of inputs to validate
        :type kwargs: dict

        :return: True if all inputs are valid, False otherwise
        :rtype: bool
        """
        for arg in args:
            if arg is None:
                logging.error(f"{arg} is None")
                return False
            if isinstance(arg, str) and arg == "":
                logging.error(f"{arg} is an empty string")
                return False

        for name, arg in kwargs.items():
            if arg is None:
                logging.error(f"{name} is None")
                return False
            if isinstance(arg, str) and arg == "":
                logging.error(f"{name} is an empty string")
                return False
        return True

    def validatePaths(self, *args: tuple, **kwargs: dict) -> bool:
        """
        Ensures that the provided paths exist.

        :param args: list of paths to validate
        :type args: tuple

        :param kwargs: list of paths to validate
        :type kwargs: dict

        :return: True if all paths exist, False otherwise
        :rtype: bool
        """
        for arg in args:
            if not os.path.exists(arg):
                logging.error(f"{arg} does not exist")
                return False
        for name, path in kwargs.items():
            if not os.path.exists(path):
                logging.error(f"{name} does not exist! \n {path}")
                return False
        return True

    def createPathsIfNotExists(self, *args: tuple) -> None:
        """
        Creates a path if it does not exist.

        :param args: list of paths to create
        :type args: tuple
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
        :type volumeNode: slicer.vtkMRMLVolumeNode
        :param segmentationNode: segmentation node
        :type segmentationNode: slicer.vtkMRMLSegmentationNode
        :param cameraDebugMode: Whether or not to keep the extracted volume in the scene, defaults to False
        :type cameraDebugMode: bool, optional

        :return: tuple containing the extracted volume and the bounds of the volume
        :rtype: tuple[vtk.vtkImageData, list[float]]
        """
        mergedSegmentationNode = SubVolumeExtraction.mergeSegments(volumeNode, segmentationNode)
        newVolumeNode = SubVolumeExtraction.extractSubVolume(
            volumeNode, mergedSegmentationNode, mergedSegmentationNode.GetSegmentation().GetNthSegmentID(0)
        )
        newVolumeNode.SetName(volumeNode.GetName() + " - Bone Subvolume")

        bounds = [0, 0, 0, 0, 0, 0]
        newVolumeNode.GetRASBounds(bounds)

        # Copy the metadata from the original volume into the ImageData
        newVolumeImageData = vtk.vtkImageData()
        newVolumeImageData.DeepCopy(newVolumeNode.GetImageData())  # So we don't modify the original volume
        newVolumeImageData.SetSpacing(newVolumeNode.GetSpacing())
        origin = list(newVolumeNode.GetOrigin())
        newVolumeImageData.SetOrigin(origin)

        mat = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(mat)

        if mat.GetElement(0, 0) < 0 and mat.GetElement(1, 1) < 0:
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
        volumeImageData: vtk.vtkImageData,
        outputDir: str,
        imageSize: list[int],
        frameNum: int = 1,
        progressCallback=None,
    ) -> None:
        """
        Generates VRG files for each camera in the cameras list

        :param cameras: list of cameras
        :type cameras: list[RadiographGeneration.Camera]
        :param volumeImageData: volume image data
        :type volumeImageData: vtk.vtkImageData
        :param outputDir: output directory
        :type outputDir: str
        :param imageSize: image size
        :type imageSize: list[int]
        :param frameNum: frame number, defaults to 1
        :type frameNum: int, optional
        :param progressCallback: progress callback, defaults to None
        :type progressCallback: callable, optional
        """
        self.createPathsIfNotExists(outputDir)

        if not progressCallback:
            logging.warning(
                "[AutoscoperM.logic.generateVRGForCameras] "
                "No progress callback provided, progress bar will not be updated"
            )

            def progressCallback(x):
                return x

        # write a temporary volume to disk
        volumeFName = "AutoscoperM_VRG_GEN_TEMP.mhd"
        IO.writeTemporyFile(volumeFName, volumeImageData)

        # Start a CLI node for each camera
        cliModule = slicer.modules.virtualradiographgeneration
        cliNodes = []
        for cam in cameras:
            cameraDir = os.path.join(outputDir, f"cam{cam.id}")
            self.createPathsIfNotExists(cameraDir)
            camera = cam.vtkCamera
            parameters = {
                "inputVolumeFName": os.path.join(slicer.app.temporaryPath, volumeFName),
                "cameraPosition": [camera.GetPosition()[0], camera.GetPosition()[1], camera.GetPosition()[2]],
                "cameraFocalPoint": [camera.GetFocalPoint()[0], camera.GetFocalPoint()[1], camera.GetFocalPoint()[2]],
                "cameraViewUp": [camera.GetViewUp()[0], camera.GetViewUp()[1], camera.GetViewUp()[2]],
                "cameraViewAngle": camera.GetViewAngle(),
                "clippingRange": [camera.GetClippingRange()[0], camera.GetClippingRange()[1]],
                "width": imageSize[0],
                "height": imageSize[1],
                "outputFName": os.path.join(cameraDir, f"{frameNum}.tif"),
            }
            cliNode = slicer.cli.run(cliModule, None, parameters)  # run asynchronously
            cliNodes.append(cliNode)

        # Wait for all the CLI nodes to finish
        for i, cliNode in enumerate(cliNodes):
            while cliNodes[i].GetStatusString() != "Completed":
                slicer.app.processEvents()
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                # error
                errorText = cliNode.GetErrorText()
                slicer.mrmlScene.RemoveNode(cliNode)
                raise ValueError("CLI execution failed: " + errorText)
            # get the output
            slicer.mrmlScene.RemoveNode(cliNode)
            progress = ((i + 1) / len(cameras)) * 30 + 10
            progressCallback(progress)

        IO.removeTemporyFile(volumeFName)

    def moveOptimizedVRGsAndGenCalibFiles(
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
        :type bestCameras: list[RadiographGeneration.Camera]
        :param tmpDir: temporary directory
        :type tmpDir: str
        :param finalDir: final directory
        :type finalDir: str
        :param calibDir: calibration directory
        :type calibDir: str
        :param progressCallback: progress callback, defaults to None
        :type progressCallback: callable, optional
        """
        self.validatePaths(tmpDir=tmpDir)
        self.createPathsIfNotExists(finalDir, calibDir)
        if not progressCallback:
            logging.warning(
                "[AutoscoperM.logic.moveOptimizedVRGsAndGenCalibFiles] "
                "No progress callback provided, progress bar will not be updated"
            )

            def progressCallback(x):
                return x

        for i, cam in enumerate(bestCameras):
            IO.generateCameraCalibrationFile(cam, os.path.join(calibDir, f"cam{cam.id}.yaml"))
            cameraDir = os.path.join(finalDir, f"cam{cam.id}")
            self.createPathsIfNotExists(cameraDir)
            # Copy all tif files from the tmp to the final directory
            for file in glob.glob(os.path.join(tmpDir, f"cam{cam.id}", "*.tif")):
                shutil.copy(file, cameraDir)

            progress = ((i + 1) / len(bestCameras)) * 10 + 90
            progressCallback(progress)

    def centerVolume(self, volumeNode: slicer.vtkMRMLVolumeNode, transformPath: str, is_4d: bool) -> None:
        """
        A requirement for Autoscoper is that the center of the volume is at the origin.
        This method will center the volume and save the transform to the transformPath

        :param volumeNode: volume node
        :type volumeNode: slicer.vtkMRMLVolumeNode
        :param transformPath: path to save the transform to
        :type transformPath: str
        :param is_4d: whether or not the volume is a 4D volume
        :type is_4d: bool
        :return: None
        """

        # Get the bounds of the volume
        bounds = [0] * 6
        if is_4d:
            volumeNode.GetNthDataNode(0).GetRASBounds(bounds)
        else:
            volumeNode.GetRASBounds(bounds)

        # Get the center of the volume
        center = [0] * 3
        for i in range(3):
            center[i] = (bounds[i * 2] + bounds[i * 2 + 1]) / 2

        center_rounded = [round(x) for x in center]  # don't want to move the volume if its off by a small amount
        if center_rounded == [0, 0, 0]:
            return  # Already centered

        # Create a transform node
        transformNode = slicer.vtkMRMLTransformNode()
        transformNode.SetName("CenteringTransform")
        slicer.mrmlScene.AddNode(transformNode)

        # Get the transform matrix
        matrix = vtk.vtkMatrix4x4()

        # Move the center of the volume to the origin
        matrix.SetElement(0, 3, -center[0])
        matrix.SetElement(1, 3, -center[1])
        matrix.SetElement(2, 3, -center[2])

        # Set the transform matrix
        transformNode.SetMatrixTransformToParent(matrix)

        # Apply the transform to the volume
        num_frames = 1
        curVol = volumeNode
        if is_4d:
            num_frames = volumeNode.GetNumberOfDataNodes()
        for i in range(num_frames):
            if is_4d:
                curVol = self.getItemInSequence(volumeNode, i)
            curVol.SetAndObserveTransformNodeID(transformNode.GetID())

            # Harden the transform
            slicer.modules.transforms.logic().hardenTransform(curVol)
            curVol.SetAndObserveTransformNodeID(None)

        # # Invert and save the transform
        matrix.Invert()
        transformNode.SetMatrixTransformToParent(matrix)
        slicer.util.exportNode(transformNode, os.path.join(transformPath, "Origin2DICOMCenter.tfm"))

    def getItemInSequence(self, sequenceNode: slicer.vtkMRMLSequenceNode, idx: int) -> slicer.vtkMRMLNode:
        """
        Returns the item at the specified index in the sequence node

        :param sequenceNode: sequence node
        :type sequenceNode: slicer.vtkMRMLSequenceNode
        :param idx: index
        :type idx: int
        :return: item at the specified index
        :rtype: slicer.vtkMRMLNode
        """
        if type(sequenceNode) != slicer.vtkMRMLSequenceNode:
            logging.error("[AutoscoperM.logic.getItemInSequence] sequenceNode must be a sequence node")
            return None

        if idx >= sequenceNode.GetNumberOfDataNodes():
            logging.error(f"[AutoscoperM.logic.getItemInSequence] index {idx} is out of range")
            return None

        browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(sequenceNode)
        browserNode.SetSelectedItemNumber(idx)
        return browserNode.GetProxyNode(sequenceNode)

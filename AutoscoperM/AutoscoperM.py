import contextlib
import os, time, subprocess
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

try:  # from https://discourse.slicer.org/t/install-python-library-with-extension/10110/2
    from PyAutoscoper.connect import AutoscoperConnection
except:
    slicer.util.pip_install("PyAutoscoper~=1.1.0")
    from PyAutoscoper.connect import AutoscoperConnection

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
        self.parent.categories = ["Tracking"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Anthony Lombardi (Kitware), Bardiya Akhbari (Brown University), Amy Morton (Brown University), Beatriz Paniagua (Kitware), Jean-Christophe Fillion-Robin (Kitware)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
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


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # AutoscoperM1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AutoscoperM",
        sampleName="AutoscoperM1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "AutoscoperM1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="AutoscoperM1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="AutoscoperM1",
    )

    # AutoscoperM2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AutoscoperM",
        sampleName="AutoscoperM2",
        thumbnailFileName=os.path.join(iconsPath, "AutoscoperM2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="AutoscoperM2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="AutoscoperM2",
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
        # self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        # self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.closeAutoscoper.connect("clicked(bool)", self.onCloseAutoscoper)
        self.ui.loadConfig.connect("clicked(bool)", self.onLoadConfig)
        self.ui.saveTracking.connect("clicked(bool)", self.onSaveTracking)
        self.ui.loadTracking.connect("clicked(bool)", self.onLoadTracking)
        self.ui.startTrack.connect("clicked(bool)", self.onStartTrack)

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
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
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
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

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
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        # self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        # self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
        # self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
        # self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
        # self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = True  # False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        # get the full path from the root path
        environmentPath = os.path.join(os.getcwd(), "Autoscoper-build\\Autoscoper-build\\bin\\autoscoper_set_env")
        # get all sub directories from the bin
        sub = os.listdir(os.path.join(os.getcwd(), "Autoscoper-build\\Autoscoper-build\\bin"))
        # remove the set_env file regardless of the extension/OS
        if "autoscoper_set_env.bat" in sub:
            sub.remove("autoscoper_set_env.bat")
        if "autoscoper_set_env.sh" in sub:
            sub.remove("autoscoper_set_env.sh")
        config = sub[0]
        executablePath = os.path.join(
            os.getcwd(),
            f"Autoscoper-build\\Autoscoper-build\\bin\\{config}\\autoscoper",
        )
        if slicer.app.os == "win":
            environmentPath = environmentPath + ".bat"
            executablePath = executablePath + ".exe"
        else:
            environmentPath = environmentPath + ".sh"

        # this rips the environment variables from the batch file -> this is a little hacky
        with open(environmentPath, "r") as f:
            for line in f:
                if line.startswith("@set"):
                    line = line.replace("@set Path=", "")
                    envs = line.split(";")
                    environment = []
                    for env in envs:
                        if env != "%PATH%":
                            environment.append(env)
                    continue

        self.logic.setEnvironment(environment)
        self.logic.startAutoscoper(executablePath)

        self.sampleDir = os.path.join(
            os.getcwd(),
            f"Autoscoper-build\\Autoscoper-build\\bin\\{config}\\sample_data",
        )

    def onCloseAutoscoper(self):
        if self.logic.AutoscoperProcess.state() == qt.QProcess.NotRunning or not self.logic.isAutoscoperOpen:
            logging.info("Autoscoper is not open")
            return
        self.logic.stopAutoscoper()

    def onLoadConfig(self):
        configPath = self.ui.configSelector.currentPath
        if configPath.endswith(".cfg") and os.path.exists(configPath):
            self.logic.AutoscoperSocket.loadTrial(configPath)
            frames = self.logic.AutoscoperSocket.getNumFrames()
            volumes = self.logic.AutoscoperSocket.getNumVolumes()
            self.ui.selectedVolume.maximum = volumes - 1
            self.ui.trackingVolume.maximum = volumes - 1
            self.ui.endFrame.maximum = frames - 1
            self.ui.endFrame.value = frames - 1
            self.ui.startFrame.maximum = frames - 1
        else:
            logging.info("Invalid config file")

    def onSaveTracking(self):
        trackingPath = self.ui.trackingSelector.currentPath
        volume = self.ui.selectedVolume.value
        if trackingPath.endswith(".tra"):
            save_as_matrix = self.ui.matrixRadio.checked
            save_as_rows = self.ui.rowRadio.checked
            save_with_commas = self.ui.commaRadio.checked
            convert_to_cm = self.ui.cmRadio.checked
            convert_to_rad = self.ui.radRadio.checked
            interpolate = self.ui.splineRadio.checked
            self.logic.AutoscoperSocket.saveTracking(
                volume,
                trackingPath,
                save_as_matrix,
                save_as_rows,
                save_with_commas,
                convert_to_cm,
                convert_to_rad,
                interpolate,
            )
        else:
            logging.info("Invalid tracking file")

    def onLoadTracking(self):
        trackingPath = self.ui.trackingSelector.currentPath
        volume = self.ui.selectedVolume.value
        if trackingPath.endswith(".tra") and os.path.exists(trackingPath):
            is_maxtrix = self.ui.matrixRadio.checked
            is_rows = self.ui.rowRadio.checked
            has_commas = self.ui.commaRadio.checked
            is_cm = self.ui.cmRadio.checked
            is_rad = self.ui.radRadio.checked
            is_spline = self.ui.splineRadio.checked
            self.logic.AutoscoperSocket.loadTrackingData(
                volume,
                trackingPath,
                is_maxtrix,
                is_rows,
                has_commas,
                is_cm,
                is_rad,
                is_spline,
            )
        else:
            logging.info("Invalid tracking file")

    def onStartTrack(self):
        volume = self.ui.trackingVolume.value
        startFrame = self.ui.startFrame.value
        endFrame = self.ui.endFrame.value
        skipFrame = self.ui.skipFrame.value
        reverse = self.ui.reverse.checked
        optMethod = self.ui.downhillRadio.checked
        refinements = self.ui.refinements.value
        minLim = self.ui.minLim.value
        maxLim = self.ui.maxLim.value
        maxEpochs = self.ui.maxEpoch.value
        maxStall = self.ui.maxStall.value
        cf = self.ui.sadRadio.checked

        self.ui.progressBar.value = 0

        for frame in range(startFrame, endFrame, skipFrame):
            self.logic.AutoscoperSocket.setFrame(frame)
            if reverse:
                frame = endFrame - (frame - startFrame)
            if frame != startFrame:
                pose = self.logic.AutoscoperSocket.getPose(volume, frame - 1)
                self.logic.AutoscoperSocket.setPose(volume, frame, pose)
            self.ui.progressBar.value = ((frame - startFrame + 1) / (endFrame - startFrame)) * 100
            self.ui.progressBar.repaint()
            self.logic.AutoscoperSocket.optimizeFrame(
                volume, frame, refinements, maxEpochs, minLim, maxLim, maxStall, skipFrame, optMethod, cf
            )


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
        self.AutoscoperEnvironment = qt.QProcessEnvironment()
        self.AutoscoperProcess.setProcessChannelMode(qt.QProcess.ForwardedChannels)
        self.isAutoscoperOpen = False
        self.AutoscoperSocket = None

    def setEnvironment(self, environment):
        """
        Add environment paths to the PATH variable
        """
        curPath = ""
        for env in environment:
            curPath = curPath + ";" + env
        self.AutoscoperEnvironment.insert("PATH", curPath)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def connectToAutoscoper(self):
        """Connect to a running instance of Autoscoper."""
        if self.AutoscoperSocket:
            logging.warning("connection to Autoscoper is already established")
            return
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
            logging.error("Specified executable %s does not exist" % executablePath)
            return

        if self.AutoscoperProcess.state() in [
            qt.QProcess.Starting,
            qt.QProcess.Running,
        ]:
            logging.error("Autoscoper executable already started")
            return

        self.isAutoscoperOpen = True

        @contextlib.contextmanager
        def changeCurrentDir(directory):
            currentDirectory = os.getcwd()
            try:
                os.chdir(directory)
                yield
            finally:
                os.chdir(currentDirectory)

        with changeCurrentDir(os.path.dirname(executablePath)):
            logging.info("Starting Autoscoper %s" % executablePath)
            self.AutoscoperProcess.setProcessEnvironment(self.AutoscoperEnvironment)
            self.AutoscoperProcess.start(executablePath)
            self.AutoscoperProcess.waitForStarted()

        slicer.app.processEvents()

        time.sleep(2)  # wait for autoscoper to boot up before connecting

        self.connectToAutoscoper()

    def stopAutoscoper(self, force=True):
        """Stop Autoscoper process"""
        if self.AutoscoperProcess.state() == qt.QProcess.NotRunning:
            logging.error("Autoscoper executable is not running")
            return

        if self.AutoscoperSocket:
            self.disconnectFromAutoscoper()
        if force:
            self.AutoscoperProcess.kill()
        else:
            self.AutoscoperProcess.terminate()

        self.isAutoscoperOpen = False

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            cliParams,
            wait_for_completion=True,
            update_display=showResult,
        )
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info("Processing completed in {0:.2f} seconds".format(stopTime - startTime))


#
# AutoscoperMTest
#


class AutoscoperMTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_AutoscoperM1()

    def test_AutoscoperM1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("AutoscoperM1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AutoscoperMLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")

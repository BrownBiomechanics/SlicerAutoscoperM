from enum import Enum
from typing import Optional

import slicer
import vtk
from slicer import vtkMRMLLinearTransformNode, vtkMRMLScalarVolumeNode, vtkMRMLSequenceNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

from AutoscoperM import AutoscoperMLogic
from Hierarchical3DRegistrationLib.TreeNode import TreeNode


#
# Hierarchical3DRegistration
#
class Hierarchical3DRegistration(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Hierarchical3DRegistration")
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
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
        This is an example of scripted loadable module bundled in an extension.
        """
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
        This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
        and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
        """
        )


class Hierarchical3DRegistrationRunStatus(Enum):
    NOT_RUNNING = 0
    INITIALIZING = 1
    IN_PROGRESS = 2
    CANCELING = 3


#
# Hierarchical3DRegistrationParameterNode
#
@parameterNodeWrapper
class Hierarchical3DRegistrationParameterNode:
    """
    The parameters needed by the module.

    volumeSequence: The volume sequence (target) to be registered
    sourceVolume: The scalar volume to register from (from which input hierarchy were generated)
    hierarchyRootID: The ID associated with the root of the model hierarchy
    startFrameIdx: The frame index in volumeSequence from which registration is to start
    endFrameIdx: The frame index in volumeSequence up to which registration is to be performed
    trackOnlyRoot: Whether to only register the root node in the hierarchy
    skipManualTfmAdjustments: Whether to skip manual user intervention for the
                              initial guess transform for each registration

    totalBones: The total number of bones to track in each frame, saved for progress bar
    currentFrameIdx: The current target frame being registered
    currentBoneID: The current bone in the frame being registered
    runSatus: The current state of the registration module
    statusMsg: The message displayed to the user indicating the current workflow step
    """

    # UI fields
    volumeSequence: vtkMRMLSequenceNode
    sourceVolume: vtkMRMLScalarVolumeNode
    hierarchyRootID: int
    startFrameIdx: int
    endFrameIdx: int
    trackOnlyRoot: bool
    skipManualTfmAdjustments: bool

    # Registration parameters
    totalBones: int
    currentFrameIdx: int
    currentBoneID: int # TODO: use this to reconstruct TreeNode obj from scene?
    runSatus: Hierarchical3DRegistrationRunStatus
    statusMsg: str


#
# Hierarchical3DRegistrationWidget
#
class Hierarchical3DRegistrationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.rootBone = None
        self.currentBone = None
        self.bonesToTrack = None

        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Hierarchical3DRegistration.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = Hierarchical3DRegistrationLogic()
        self.logic.parameterFile = self.resourcePath("ParameterFiles/rigid.txt")

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Sets the frame slider range to be the number of nodes within the sequence
        self.ui.inputCTSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateFrameSlider)

        # Buttons
        self.ui.initializeButton.connect("clicked(bool)", self.onInitializeButton)
        self.ui.registerButton.connect("clicked(bool)", self.onRegisterButton)
        self.ui.abortButton.connect("clicked(bool)", self.onAbortButton)
        self.ui.importButton.connect("clicked(bool)", self.onImportButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.cleanupRegistrationProcess()
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        pass

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        pass

    def onSceneStartClose(self, _caller, _event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller, _event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        self.setParameterNode(self.logic.getParameterNode())

        self._parameterNode.currentFrameIdx = -1
        self._parameterNode.currentBoneID = -1
        self._parameterNode.totalBones = 0
        self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.NOT_RUNNING

        if not self._parameterNode.volumeSequence:
            firstSequenceNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceNode")
            if firstSequenceNode:
                self._parameterNode.volumeSequence = firstSequenceNode

    def setParameterNode(self, inputParameterNode: Optional[Hierarchical3DRegistrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateRegistrationButtonsState)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateRegistrationButtonsState)
            self.updateRegistrationButtonsState()

    def updateRegistrationButtonsState(self, _caller=None, _event=None):
        """Set the button text and whether the buttons are enabled."""
        # update the abort and initialize buttons
        if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.NOT_RUNNING:
            self.ui.abortButton.enabled = False
            self.ui.initializeButton.enabled = True
            self.updateProgressBar(0)
        else:
            self.ui.abortButton.enabled = True
            self.ui.initializeButton.enabled = False

        # update the register button
        if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.IN_PROGRESS:
            self.ui.registerButton.enabled = True
            if self._parameterNode.skipManualTfmAdjustments:
                self.ui.registerButton.text = "Register"
            else:
                self.ui.registerButton.text = "Set Initial Guess And Register"
        else:
            self.ui.registerButton.enabled = False
            self.ui.registerButton.text = "Set Initial Guess And Register"

        slicer.app.processEvents()

    def updateProgressBar(self, value):
        """
        Update the progress bar to indicate the total bones in each frame registered so far
        """
        self.ui.progressBar.setValue(value)
        slicer.app.processEvents()

    def onInitializeButton(self):
        """Initializes a new registration process from the UI configuration parameters"""
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self._parameterNode.statusMsg = "Initializing registration process..."
            if self._parameterNode.runSatus != Hierarchical3DRegistrationRunStatus.NOT_RUNNING:
                raise ValueError("Cannot initialize registration process, as one is already ongoing!")

            self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.INITIALIZING
            self.updateRegistrationButtonsState()

            # TODO: Remove this once this is working with the parameterNodeWrapper
            #  see Slicer issue: https://github.com/Slicer/Slicer/issues/7905
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem()
            if currentRootIDStatus == 0:
                raise ValueError("Invalid hierarchy object selected!")
            self._parameterNode.hierarchyRootID = currentRootIDStatus

            if self._parameterNode.sourceVolume.GetID() == self._parameterNode.volumeSequence.GetID():
                raise ValueError("The source volume must not be the same as the input sequence selected!")

            # initialize representation of the bone hierarchy to be registered
            self.rootBone = TreeNode(
                hierarchyID=self._parameterNode.hierarchyRootID,
                ctSequence=self._parameterNode.volumeSequence,
                sourceVolume=self._parameterNode.sourceVolume,
                isRoot=True,
            )

            # calculate total number of bones and frames for the progress bar display
            nodeCount = 1
            if not self._parameterNode.trackOnlyRoot:
                nodeCount += self.rootBone.shNode.GetNumberOfItemChildren(self._parameterNode.hierarchyRootID, True)
            self._parameterNode.totalBones = nodeCount

            # initialize the registration status variables
            self._parameterNode.currentFrameIdx = self._parameterNode.startFrameIdx
            self.bonesToTrack = [self.rootBone]
            self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.IN_PROGRESS
            self.updateRegistrationButtonsState()

            # prepare for the next step in the workflow
            nextFrame = AutoscoperMLogic.getItemInSequence(self._parameterNode.volumeSequence, self._parameterNode.currentFrameIdx)[0]
            nextFrame.CreateDefaultDisplayNodes()
            nextFrameDisplayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(nextFrame)
            nextFrameDisplayNode.SetVisibility(1)
            slicer.util.setSliceViewerLayers(background=nextFrame, fit=True)

            if self._parameterNode.skipManualTfmAdjustments:
                self.onRegisterButton()
            else:
                nextBone = self.bonesToTrack[0]
                nextBone.startInteraction(self._parameterNode.currentFrameIdx)
                self._parameterNode.statusMsg = "Adjust the initial guess transform for the bone " \
                                                    f"'{nextBone.name}' in frame {self._parameterNode.currentFrameIdx}"
                slicer.app.processEvents()

    def onRegisterButton(self):
        """Sets up and performs the registration step for the next bone in frame"""
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            # TODO: Remove this once this is working with the parameterNodeWrapper.
            #  It's currently commented out due to bug with parameter node, see
            #  Slice issue: https://github.com/Slicer/Slicer/issues/7905
            if not currentRootIDStatus:
                raise ValueError("Invalid hierarchy object selected!")
            if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.CANCELING:
                raise ValueError("Canceling registration...")

            # get bone and target frame for the current step
            targetFrameIdx = self._parameterNode.currentFrameIdx
            self.currentBone = self.bonesToTrack.pop(0)

            # update UI to prepare for automated registration
            self.currentBone.stopInteraction(targetFrameIdx)
            slicer.util.forceRenderAllViews()
            self._parameterNode.statusMsg = f"Registering bone '{self.currentBone.name}' in frame {targetFrameIdx}"

            # crop the target frame based on the source ROI and initial guess transform
            target_volume = AutoscoperMLogic.getItemInSequence(self._parameterNode.volumeSequence, targetFrameIdx)[0]
            self.currentBone.setupFrame(targetFrameIdx, target_volume)

            # perform the registration and update the hierarchy
            self.logic.registerBoneInFrame(
                self.currentBone,
                targetFrameIdx,
                self._parameterNode.trackOnlyRoot
            )

            # If there is a next frame to register, copy the result transform as the next initial guess
            if targetFrameIdx != self._parameterNode.endFrameIdx:
                self.currentBone.copyTransformToNextFrame(targetFrameIdx)

            # append all child bones to queue of bones to register next in this frame
            if not self._parameterNode.trackOnlyRoot:
                self.bonesToTrack.extend(self.currentBone.childNodes)

            # report progress
            totalFrames = self._parameterNode.endFrameIdx - self._parameterNode.startFrameIdx + 1
            progress = self.ui.progressBar.value + 1 / (self._parameterNode.totalBones * totalFrames) * 100
            self.updateProgressBar(progress)

            # prepare for the next step in the workflow
            if len(self.bonesToTrack) == 0:
                # we just finished registering all the bones in the current
                # frame, so now we can move on to the next frame
                self._parameterNode.currentFrameIdx += 1

                self.rootBone.setModelsVisibility(False)
                nextFrame = AutoscoperMLogic.getItemInSequence(self._parameterNode.volumeSequence, self._parameterNode.currentFrameIdx)[0]
                nextFrame.CreateDefaultDisplayNodes()
                nextFrameDisplayNode = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(nextFrame)
                nextFrameDisplayNode.SetVisibility(1)
                slicer.util.setSliceViewerLayers(background=nextFrame, fit=True)
                slicer.app.processEvents()
                self.bonesToTrack = [self.rootBone]

            if self._parameterNode.currentFrameIdx > self._parameterNode.endFrameIdx:
                # we successfully finished the registration process
                browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(self._parameterNode.volumeSequence)
                browserNode.SetSelectedItemNumber(self._parameterNode.endFrameIdx)
                self.cleanupRegistrationProcess()
                self.updateProgressBar(100)
                slicer.util.messageBox("Success! Registration Complete.")
            elif self._parameterNode.skipManualTfmAdjustments:
                # if user doesn't need to adjust anything, continue to next step right away
                self.onRegisterButton()
            else:
                # prepare for user interaction of transform adjustments for the next bone
                nextBone = self.bonesToTrack[0]
                nextFrameIdx = self._parameterNode.currentFrameIdx
                self._parameterNode.statusMsg = "Adjust the initial guess transform for the bone " \
                                                      f"'{nextBone.name}' in frame {nextFrameIdx}"

                # advance the sequence browser to visualize the next frame
                browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(self._parameterNode.volumeSequence)
                browserNode.SetSelectedItemNumber(nextFrameIdx)
                nextBone.startInteraction(nextFrameIdx)
                slicer.app.processEvents()

    def onAbortButton(self):
        """Stop ongoing registration process and restores all module parameters to default"""
        self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.CANCELING
        self.updateRegistrationButtonsState()
        self.cleanupRegistrationProcess()

    def cleanupRegistrationProcess(self):
        """Reset all parameters and UI components at the end of a registration run"""
        # remove all visual aids from the registration process
        if self.rootBone:
            self.rootBone.setModelsVisibility(True)
            nodesList = [self.rootBone]
            for node in nodesList:
                node.stopInteraction(self._parameterNode.currentFrameIdx)
                nodesList.extend(node.childNodes)

        # reset current parameters, effectively wiping the current status saved
        self.setParameterNode(None)
        self.initializeParameterNode()
        self.rootBone = None
        self.currentBone = None
        self.bonesToTrack = None

        # update UI to reflect process no longer in progress
        self._parameterNode.statusMsg = ""
        self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.NOT_RUNNING
        self.updateRegistrationButtonsState()

    def onImportButton(self):  # TODO: revisit
        """UI button for reading the TRA files into sequences."""
        import glob
        import logging
        import os

        with slicer.util.tryWithErrorDisplay("Failed to import transforms", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            if not currentRootIDStatus:  # TODO: Remove this once this is working with the parameterNodeWrapper
                raise ValueError("Invalid hierarchy object selected!")

            CT = self.ui.inputCTSequenceSelector.currentNode()
            rootID = self.ui.SubjectHierarchyComboBox.currentItem()
            rootNode = TreeNode(hierarchyID=rootID, ctSequence=CT, isRoot=True)

            importDir = self.ui.ioDir.currentPath
            if importDir == "":
                raise ValueError("Import directory not set!")

            node_list = [rootNode]
            for node in node_list:
                node.model.SetAndObserveTransformNodeID(node.getTransform(0).GetID())
                node_list.extend(node.childNodes)

                foundFiles = glob.glob(os.path.join(importDir, f"{node.name}*.tra"))
                if len(foundFiles) == 0:
                    logging.warning(f"No files found matching the '{node.name}*.tra' pattern")
                    return

                if len(foundFiles) > 1:
                    logging.warning(
                        f"Found multiple tra files matching the '{node.name}*.tra' pattern, using {foundFiles[0]}"
                    )

                node.importTransfromsFromTRAFile(foundFiles[0])
        slicer.util.messageBox("Success!")

    def onExportButton(self):  # TODO: revisit
        """UI button for writing the sequences as TRA files."""
        with slicer.util.tryWithErrorDisplay("Failed to export transforms.", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            if not currentRootIDStatus:  # TODO: Remove this once this is working with the parameterNodeWrapper
                raise ValueError("Invalid hierarchy object selected!")

            CT = self.ui.inputCTSequenceSelector.currentNode()
            rootID = self.ui.SubjectHierarchyComboBox.currentItem()
            rootNode = TreeNode(hierarchyID=rootID, ctSequence=CT, isRoot=True)

            exportDir = self.ui.ioDir.currentPath
            if exportDir == "":
                raise ValueError("Export directory not set!")

            node_list = [rootNode]
            for node in node_list:
                node.exportTransformsAsTRAFile(exportDir)
                node_list.extend(node.childNodes)
        slicer.util.messageBox("Success!")

    def updateFrameSlider(self, CTSelectorNode: slicer.vtkMRMLNode):
        """Update the slider and spin boxes when a new sequence is selected."""
        # TODO: fix slider misbehaving
        if AutoscoperMLogic.IsSequenceVolume(CTSelectorNode):
            numNodes = CTSelectorNode.GetNumberOfDataNodes()
            maxFrame = numNodes - 1
        elif CTSelectorNode is None:
            maxFrame = 0
        self.ui.frameSlider.maximum = maxFrame
        self.ui.startFrame.maximum = maxFrame
        self.ui.endFrame.maximum = maxFrame
        self.ui.endFrame.value = maxFrame


#
# Hierarchical3DRegistrationLogic
#


class Hierarchical3DRegistrationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._itk = None

    @property
    def itk(self):
        import logging

        if self._itk is None:
            logging.info("Importing itk...")
            self._itk = self.importITKElastix()
        return self._itk

    def importITKElastix(self):
        import logging

        try:
            # Since running hasattr(itk, "ElastixRegistrationMethod") is slow,
            # check if Elastix is installed by attempting to import ElastixRegistrationMethod
            from itk import ElastixRegistrationMethod

            del ElastixRegistrationMethod
        except ImportError:
            self.installITKElastix()

        import itk

        logging.info(f"ITK imported correctly: itk {itk.__version__}")
        return itk

    @staticmethod
    def installITKElastix():
        import logging

        if not slicer.util.confirmOkCancelDisplay(
            "ITK-elastix will be downloaded and installed now. This process may take a minute",
            dontShowAgainSettingsKey="Hierarchical3DRegistration/DontShowITKElastixInstallWarning",
        ):
            logging.info("ITK-elasitx install aborted by user.")
            return None
        slicer.util.pip_install("itk-elastix")
        import itk

        logging.info(f"Installed itk version {itk.__version__}")
        return itk

    def getParameterNode(self):
        return Hierarchical3DRegistrationParameterNode(super().getParameterNode())

    @staticmethod
    def parameterObject2SlicerTransform(paramObj) -> slicer.vtkMRMLLinearTransformNode:
        from math import cos, sin

        import numpy as np

        transformParameters = [float(val) for val in paramObj.GetParameter(0, "TransformParameters")]
        rx, ry, rz = transformParameters[0:3]
        tx, ty, tz = transformParameters[3:]
        centerOfRotation = [float(val) for val in paramObj.GetParameter(0, "CenterOfRotationPoint")]

        rotX = np.array([[1.0, 0.0, 0.0], [0.0, cos(rx), -sin(rx)], [0.0, sin(rx), cos(rx)]])
        rotY = np.array([[cos(ry), 0.0, sin(ry)], [0.0, 1.0, 0.0], [-sin(ry), 0.0, cos(ry)]])
        rotZ = np.array([[cos(rz), -sin(rz), 0.0], [sin(rz), cos(rz), 0.0], [0.0, 0.0, 1.0]])

        fixedToMovingDirection = np.dot(np.dot(rotZ, rotX), rotY)

        fixedToMoving = np.eye(4)
        fixedToMoving[0:3, 0:3] = fixedToMovingDirection

        offset = np.array([tx, ty, tz]) + np.array(centerOfRotation)
        offset[0] -= np.dot(fixedToMovingDirection[0, :], np.array(centerOfRotation))
        offset[1] -= np.dot(fixedToMovingDirection[1, :], np.array(centerOfRotation))
        offset[2] -= np.dot(fixedToMovingDirection[2, :], np.array(centerOfRotation))
        fixedToMoving[0:3, 3] = offset
        ras2lps = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        fixedToMoving = np.dot(np.dot(ras2lps, fixedToMoving), ras2lps)

        tfmNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        tfmNode.SetMatrixTransformToParent(slicer.util.vtkMatrixFromArray(fixedToMoving))
        return tfmNode

    def registerRigidBody(
        self,
        sourceVolume: vtkMRMLScalarVolumeNode,
        targetVolume: vtkMRMLScalarVolumeNode,
        transformNode: vtkMRMLLinearTransformNode,
    ) -> None:
        """
        Registers a source volume to a target volume using ITKElastix.
        The output of this function is written into transformNode such that,
        when applied on the source image, will align it with the target image.

        :param sourceVolume: the source input to be registered, aka the "moving image"
        :param targetVolume: the target input to be registered, aka the "fixed image"
        :param transformNode: the initial guess transform, node will be overwitten to
                              the composition of its value and the ITKElastix result
                              (the total transform from sourceVolume to targetVolume)
        """
        from tempfile import NamedTemporaryFile

        # Apply the initial guess if there is one
        sourceVolume.SetAndObserveTransformNodeID(None)
        sourceVolume.SetAndObserveTransformNodeID(transformNode.GetID())
        sourceVolume.HardenTransform()

        # Register with Elastix
        with NamedTemporaryFile(suffix=".mha") as movingTempFile, NamedTemporaryFile(suffix=".mha") as fixedTempFile:
            slicer.util.saveNode(sourceVolume, movingTempFile.name)
            slicer.util.saveNode(targetVolume, fixedTempFile.name)

            movingITKImage = self.itk.imread(movingTempFile.name, self.itk.F)
            fixedITKImage = self.itk.imread(fixedTempFile.name, self.itk.F)

            paramObj = self.itk.ParameterObject.New()
            paramObj.AddParameterMap(paramObj.GetDefaultParameterMap("rigid"))
            # paramObj.AddParameterFile(self.parameterFile)

            elastixObj = self.itk.ElastixRegistrationMethod.New(fixedITKImage, movingITKImage)
            elastixObj.SetParameterObject(paramObj)
            elastixObj.SetNumberOfThreads(16)
            elastixObj.LogToConsoleOn()  # TODO: Update this to log to file instead
            try:
                elastixObj.UpdateLargestPossibleRegion()
            except Exception:
                # Remove the hardened initial guess and then throw the exception
                transformNode.Inverse()
                sourceVolume.SetAndObserveTransformNodeID(transformNode.GetID())
                sourceVolume.HardenTransform()
                transformNode.Inverse()
                raise

        resultTransform = self.parameterObject2SlicerTransform(elastixObj.GetTransformParameterObject())
        # The elastix result represents the transformation from the fixed to the moving
        # image, we so invert it to get the transform from the moving to the fixed
        resultTransform.Inverse()

        # Remove the hardened initial guess
        transformNode.Inverse()
        sourceVolume.SetAndObserveTransformNodeID(transformNode.GetID())
        sourceVolume.HardenTransform()
        transformNode.Inverse()

        # Combine the initial and result transforms
        transformNode.SetAndObserveTransformNodeID(resultTransform.GetID())
        transformNode.HardenTransform()

        # Clean up intermediate result node that was just added to the scene
        slicer.mrmlScene.RemoveNode(resultTransform)

    def registerBoneInFrame(
        self,
        boneNode: TreeNode,
        targetFrameIdx: int,
        trackOnlyRoot: bool = False,
    ) -> None:
        """
        Performs hierarchical registration for a specific bone in a sequence frame.

        :param boneNode: the object representing the bone to be registered
        :param targetFrameIdx: the target frame index the bone will be registered to
        :param trackOnlyRoot: whether to propagate the result to child node in the hierarchy
        """
        import logging
        import time

        # get the moving and fixed images for registration, as well as the initial guess transform
        source_cropped_volume = boneNode.croppedSourceVolume
        target_cropped_volume = boneNode.getCroppedFrame(targetFrameIdx)
        bone_tfm = boneNode.getTransform(targetFrameIdx)

        # calculate the relative transform from the source to the target volume
        logging.info(f"Registering: {boneNode.name} to frame {targetFrameIdx}")
        start = time.time()
        self.registerRigidBody(
            source_cropped_volume,
            target_cropped_volume,
            bone_tfm,
        )
        end = time.time()
        logging.info(f"{boneNode.name} took {end-start:.3f}s for frame {targetFrameIdx}.")

        # propagate the result transform to all child nodes of the current bone in the hierarchy
        if not trackOnlyRoot:
            boneNode.applyTransformToChildren(targetFrameIdx, bone_tfm)
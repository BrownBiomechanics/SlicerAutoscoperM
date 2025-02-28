from enum import Enum
from typing import Optional

import slicer
import vtk
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSequenceNode, vtkMRMLLinearTransformNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

import AutoscoperM
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
    The parameters needed by module.

    hierarchyRootID: The ID associated with the root of the model hierarchy
    volumeSequence: The volume sequence to be registered
    startFrame: The frame index from which registration is to start
    endFrame: The frame index up to which registration is to be performed
    trackOnlyRoot: Whether to only register the root node in the hierarchy
    skipManualTfmAdjustments: Whether to skip manual user intervention for the
                              initial guess for each registration

    currentFrame: The current target frame being registered
    currentBone: The current bone in the frame being registered
    runSatus: # TODO: running (in progress), not running, aborting
    """

    # UI fields
    hierarchyRootID: int
    volumeSequence: vtkMRMLSequenceNode
    startFrame: int
    endFrame: int
    trackOnlyRoot: bool
    skipManualTfmAdjustments: bool
    statusMsg: str

    # Registration parameters
    currentFrame: int
    currentBoneID: int # save this for the scene metadata, but for actual session, would be more convenient to save TreeNode obj
    runSatus: Hierarchical3DRegistrationRunStatus


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
        self.ui.inputSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.updateFrameSlider)

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
        # Make sure parameter node exists and observed
        #self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        #if self._parameterNode:
        #    self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        #    self._parameterNodeGuiTag = None
        #    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateRegistrationButtonsState)

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

        self._parameterNode.currentFrame = -1
        self._parameterNode.currentBoneID = -1
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
        """Sets the text and whether the buttons are enabled."""
        # update the abort and initialize buttons
        if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.NOT_RUNNING:
            self.ui.abortButton.enabled = False
            self.ui.initializeButton.enabled = True
        else:
            self.ui.abortButton.enabled = True
            self.ui.initializeButton.enabled = False

        # update the register button
        if (self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.CANCELING or
            self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.NOT_RUNNING):
            self.ui.registerButton.enabled = False
            self.ui.registerButton.text = "Set Initial Guess And Register"
        else:
            self.ui.registerButton.enabled = True
            if self._parameterNode.skipManualTfmAdjustments:
                self.ui.registerButton.text = "Register"
            elif self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.INITIALIZING:
                self.ui.registerButton.text = "Adjust Initial Frame"
            else:
                self.ui.registerButton.text = "Set Initial Guess And Register"
        # tooltip?: "Adjust the initial guess transform for the registration of this bone in the current frame."

        slicer.app.processEvents()

    def onInitializeButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            self._parameterNode.statusMsg = "Initializing registration process"
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

            self.rootBone = TreeNode(
                hierarchyID=self._parameterNode.hierarchyRootID,
                ctSequence=self._parameterNode.volumeSequence,
                isRoot=True,
            )
            self._parameterNode.currentFrame = self._parameterNode.startFrame
            self.bonesToTrack = [self.rootBone]

            """if self._parameterNode.skipManualTfmAdjustments:
                self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.IN_PROGRESS
                self.updateRegistrationButtonsState()
            else:"""
            nextBone = self.rootBone
            initial_tfm = nextBone.getTransform(self._parameterNode.currentFrame)
            nextBone.model.SetAndObserveTransformNodeID(initial_tfm.GetID())
            nextBone.startInteraction(self._parameterNode.currentFrame)
            self._parameterNode.statusMsg = "Adjust the initial guess transform for the bone " \
                                                f"'{nextBone.name}' in frame {self._parameterNode.currentFrame}"
            slicer.app.processEvents()

    def onRegisterButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            # TODO: Remove this once this is working with the parameterNodeWrapper.
            #  It's currently commented out due to bug with parameter node, see
            #  Slice issue: https://github.com/Slicer/Slicer/issues/7905
            if not currentRootIDStatus:
                raise ValueError("Invalid hierarchy object selected!")
            if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.CANCELING:
                raise ValueError("Canceling registration...")

            self.currentBone = self.bonesToTrack.pop(0)
            self._parameterNode.currentBoneID = -1  # TOOD: want this meaningfully saved to scene and re-imported later

            self.currentBone.stopInteraction(self._parameterNode.currentFrame)

            source_frame = self._parameterNode.currentFrame
            target_frame = self._parameterNode.currentFrame + 1

            slicer.util.forceRenderAllViews()

            if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.INITIALIZING:
                assert source_frame == self._parameterNode.startFrame

                self._parameterNode.statusMsg = f"Setting up bone '{self.currentBone.name}' in initial frame"
                # TODO: remove transform interaction here
                source_volume = AutoscoperMLogic.getItemInSequence(self._parameterNode.volumeSequence, source_frame)[0]
                self.currentBone.setupFrame(source_frame, source_volume)
                # ^^^ if self._parameterNode.skipManualTfmAdjustments: ???
            else:
                self._parameterNode.statusMsg = f"Registering bone '{self.currentBone.name}' in frame {target_frame}"
                target_volume = AutoscoperMLogic.getItemInSequence(self._parameterNode.volumeSequence, target_frame)[0]

                elastix_tfm = self.currentBone.setupFrame(target_frame, target_volume)

                self.logic.registerBoneInFrame(
                    self.currentBone,
                    elastix_tfm,
                    source_frame,
                    target_frame,
                    self._parameterNode.trackOnlyRoot
                )

                # If there is a next frame to register, propagate the transform generated from this frame
                if target_frame != self._parameterNode.endFrame:
                    self.currentBone.copyTransformToNextFrame(target_frame)

            if not self._parameterNode.trackOnlyRoot:
                self.bonesToTrack.extend(self.currentBone.childNodes)

            if len(self.bonesToTrack) == 0:
                if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.INITIALIZING:
                    # we just finished setting up all the cropped volumes in the
                    # source frame, so now we can actually start registering
                    self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.IN_PROGRESS
                else:
                    # we just finished registering all the bones in the current
                    # frame, so now we move on to the next frame
                    self._parameterNode.currentFrame += 1

                self.rootBone.setModelsVisibility(False)
                slicer.app.processEvents()
                self.bonesToTrack = [self.rootBone]

        if self._parameterNode.currentFrame == self._parameterNode.endFrame:
            # if we just finished registering the last frame, cleanup
            self.cleanupRegistrationProcess()
            slicer.util.messageBox("Success! Registration Complete.")
        elif self._parameterNode.skipManualTfmAdjustments:
            self.onRegisterButton()
        else:
            nextBone = self.bonesToTrack[0]
            nextFrame = self._parameterNode.currentFrame+1
            if self._parameterNode.runSatus == Hierarchical3DRegistrationRunStatus.INITIALIZING:
                nextFrame = self._parameterNode.currentFrame
                initial_tfm = nextBone.getTransform(nextFrame)
                nextBone.model.SetAndObserveTransformNodeID(initial_tfm.GetID())
                # TODO: add tfm interaction object here
            self._parameterNode.statusMsg = "Adjust the initial guess transform for the bone " \
                                                  f"'{nextBone.name}' in frame {nextFrame}"

            # advance frame in sequence browser to visualize next frame
            browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(self._parameterNode.volumeSequence)
            browserNode.SetSelectedItemNumber(nextFrame)
            nextBone.startInteraction(nextFrame)
            slicer.app.processEvents()

    def onAbortButton(self):
        self._parameterNode.runSatus = Hierarchical3DRegistrationRunStatus.CANCELING
        self.updateRegistrationButtonsState()
        self.cleanupRegistrationProcess()

    def cleanupRegistrationProcess(self):
        #TODO: turn off all tfm interaction
        # reset current parameters, effectively wiping index of current progress
        if self.rootBone:
            self.rootBone.setModelsVisibility(True)
        self.setParameterNode(None)
        self.initializeParameterNode()
        self.rootBone = None
        self.currentBone = None
        self.bonesToTrack = None

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

            CT = self.ui.inputSelectorCT.currentNode()
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

            CT = self.ui.inputSelectorCT.currentNode()
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
        """Updates the slider and spin boxes when a new sequence is selected."""
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
    ):
        """
        Registers a source volume to a target volume using ITKElastix.
        The output of this function is written into transformNode such that,
        when applied on the source image, will align it with the target image.

        :param sourceVolume: the source input to be registered, aka the "moving image"
        :param targetVolume: the target input to be registered, aka the "fixed image"
        :param transformNode: the node in which the results transform
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

        # Clean up
        slicer.mrmlScene.RemoveNode(resultTransform)

    def registerBoneInFrame(
        self,
        boneNode: TreeNode,
        elastixTfm: vtkMRMLLinearTransformNode,
        source_frame: int,
        target_frame: int,
        trackOnlyRoot: bool = False,
    ) -> None:
        import logging
        import time

        # get initial guess, then crop target volume
        source_cropped_volume = boneNode.getCroppedFrame(source_frame)
        target_cropped_volume = boneNode.getCroppedFrame(target_frame)

        logging.info(f"Registering: {boneNode.name} for frame {target_frame}")
        start = time.time()
        self.registerRigidBody(
            source_cropped_volume,
            target_cropped_volume,
            elastixTfm,
        )
        end = time.time()
        logging.info(f"{boneNode.name} took {end-start} for frame {target_frame}.")

        boneNode.saveRegistrationTransform(target_frame, elastixTfm)

        if not trackOnlyRoot:
            boneNode.applyTransformToChildren(target_frame, elastixTfm)

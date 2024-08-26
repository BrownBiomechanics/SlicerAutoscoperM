from typing import Optional

import slicer
import vtk
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSequenceNode, vtkMRMLTransformNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

import AutoscoperM
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


#
# Hierarchical3DRegistrationParameterNode
#
@parameterNodeWrapper
class Hierarchical3DRegistrationParameterNode:
    """
    The parameters needed by module.

    inputVolumeSequence - The volume sequence.

    """

    inputHierarchyRootID: int
    inputVolumeSequence: vtkMRMLSequenceNode


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
        self.inProgress = False
        self._parameterNode = None
        self._parameterNodeGuiTag = None
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
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.importButton.connect("clicked(bool)", self.onImportButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)
        self.ui.initHierarchyButton.connect("clicked(bool)", self.onInitHierarchyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateApplyButtonState)

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

        if not self._parameterNode.inputVolumeSequence:
            firstSequenceNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceNode")
            if firstSequenceNode:
                self._parameterNode.inputVolumeSequence = firstSequenceNode

    def setParameterNode(self, inputParameterNode: Optional[Hierarchical3DRegistrationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateApplyButtonState)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateApplyButtonState)
            self.updateApplyButtonState()

    def updateApplyButtonState(self, _caller=None, _event=None):
        """Sets the text and whether the button is enabled."""
        if self.inProgress or self.logic.isRunning:
            if self.logic.cancelRequested:
                self.ui.applyButton.text = "Cancelling..."
                self.ui.applyButton.enabled = False
            else:
                self.ui.applyButton.text = "Cancel"
                self.ui.applyButton.enabled = True
        else:
            currentCTStatus = self.ui.inputSelectorCT.currentNode() is not None
            # currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            # Unsure of the type for the parameterNodeWrapper
            if currentCTStatus:  # and currentRootIDStatus:
                self.ui.applyButton.text = "Apply"
                self.ui.applyButton.enabled = True
            elif not currentCTStatus:  # or not currentRootIDStatus:
                self.ui.applyButton.text = "Please select a Sequence and Hierarchy"
                self.ui.applyButton.enabled = False
        slicer.app.processEvents()

    def onApplyButton(self):
        """UI button for running the hierarchical registration."""
        if self.inProgress:
            self.logic.cancelRequested = True
            self.inProgress = False
        else:
            with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
                currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
                if not currentRootIDStatus:  # TODO: Remove this once this is working with the parameterNodeWrapper
                    raise ValueError("Invalid hierarchy object selected!")
                try:
                    self.inProgress = True
                    self.updateApplyButtonState()

                    CT = self.ui.inputSelectorCT.currentNode()
                    rootID = self.ui.SubjectHierarchyComboBox.currentItem()

                    startFrame = self.ui.startFrame.value
                    endFrame = self.ui.endFrame.value

                    trackOnlyRoot = self.ui.onlyTrackRootNodeCheckBox.isChecked()

                    self.logic.registerSequence(CT, rootID, startFrame, endFrame, trackOnlyRoot)
                finally:
                    self.inProgress = False
            slicer.util.messageBox("Success!")
        self.updateApplyButtonState()

    def onImportButton(self):
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
                node.dataNode.SetAndObserveTransformNodeID(node.getTransform(0).GetID())
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

    def onExportButton(self):
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
        if self.logic.autoscoperLogic.IsSequenceVolume(CTSelectorNode):
            numNodes = CTSelectorNode.GetNumberOfDataNodes()
            self.ui.frameSlider.maximum = numNodes
            self.ui.startFrame.maximum = numNodes
            self.ui.endFrame.maximum = numNodes
            self.ui.endFrame.value = numNodes
        elif CTSelectorNode is None:
            self.ui.frameSlider.maximum = 0
            self.ui.startFrame.maximum = 0
            self.ui.endFrame.maximum = 0
            self.ui.endFrame.value = 0

    def onInitHierarchyButton(self):
        """UI Button for initializing the hierarchy transforms."""
        with slicer.util.tryWithErrorDisplay("Failed initialize transforms.", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            if not currentRootIDStatus:  # TODO: Remove this once this is working with the parameterNodeWrapper
                raise ValueError("Invalid hierarchy object selected!")

            CT = self.ui.inputSelectorCT.currentNode()
            rootID = self.ui.SubjectHierarchyComboBox.currentItem()
            TreeNode(hierarchyID=rootID, ctSequence=CT, isRoot=True)


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
        self.autoscoperLogic = AutoscoperM.AutoscoperMLogic()
        self.cancelRequested = False
        self.isRunning = False
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
    def parameterObject2SlicerTransform(paramObj) -> slicer.vtkMRMLTransformNode:
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
        CT: vtkMRMLScalarVolumeNode,
        partialVolume: vtkMRMLScalarVolumeNode,
        transformNode: vtkMRMLTransformNode,
    ):
        """Registers a partial volume to a CT scan, uses ITKElastix."""
        from tempfile import NamedTemporaryFile

        # Apply the initial guess if there is one
        partialVolume.SetAndObserveTransformNodeID(None)
        partialVolume.SetAndObserveTransformNodeID(transformNode.GetID())
        partialVolume.HardenTransform()

        # Register with Elastix
        with NamedTemporaryFile(suffix=".mha") as movingTempFile, NamedTemporaryFile(suffix=".mha") as fixedTempFile:
            slicer.util.saveNode(CT, movingTempFile.name)
            slicer.util.saveNode(partialVolume, fixedTempFile.name)

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
                partialVolume.SetAndObserveTransformNodeID(transformNode.GetID())
                partialVolume.HardenTransform()
                transformNode.Inverse()
                raise

        resultTransform = self.parameterObject2SlicerTransform(elastixObj.GetTransformParameterObject())

        # Remove the hardened initial guess
        transformNode.Inverse()
        partialVolume.SetAndObserveTransformNodeID(transformNode.GetID())
        partialVolume.HardenTransform()
        transformNode.Inverse()

        # Combine the initial and result transforms
        transformNode.SetAndObserveTransformNodeID(resultTransform.GetID())
        transformNode.HardenTransform()

        # Clean up
        slicer.mrmlScene.RemoveNode(resultTransform)

    def registerSequence(
        self,
        ctSequence: vtkMRMLSequenceNode,
        rootID: int,
        startFrame: int,
        endFrame: int,
        trackOnlyRoot: bool = False,
    ) -> None:
        """Performs hierarchical registration on a ct sequence."""
        import logging
        import time

        rootNode = TreeNode(hierarchyID=rootID, ctSequence=ctSequence, isRoot=True)

        try:
            self.isRunning = True
            for idx in range(startFrame, endFrame):
                nodeList = [rootNode]
                for node in nodeList:
                    slicer.app.processEvents()
                    if self.cancelRequested:
                        logging.info("User canceled")
                        self.cancelRequested = False
                        self.isRunning = False
                        return
                    # register
                    logging.info(f"Registering: {node.name} for frame {idx}")
                    start = time.time()
                    self.registerRigidBody(
                        self.autoscoperLogic.getItemInSequence(ctSequence, idx)[0],
                        node.dataNode,
                        node.getTransform(idx),
                    )
                    end = time.time()
                    logging.info(f"{node.name} took {end-start} for frame {idx}.")

                    # Add children to node_list
                    if not trackOnlyRoot:
                        node.applyTransformToChildren(idx)
                        nodeList.extend(node.childNodes)

                    node.dataNode.SetAndObserveTransformNodeID(node.getTransform(idx).GetID())

                if idx != endFrame - 1:  # Unless its the last frame
                    rootNode.copyTransformToNextFrame(idx)
        finally:
            self.isRunning = False
            self.cancelRequested = False

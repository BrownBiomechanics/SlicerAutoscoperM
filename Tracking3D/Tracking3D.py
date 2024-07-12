from typing import Optional

import Elastix
import slicer
import vtk
from slicer import vtkMRMLMarkupsROINode, vtkMRMLScalarVolumeNode, vtkMRMLSequenceNode, vtkMRMLTransformNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import parameterNodeWrapper
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin
from Tracking3DLib.TreeNode import TreeNode

import AutoscoperM


#
# Tracking3D
#
class Tracking3D(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Tracking3D")
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
# Tracking3DParameterNode
#
@parameterNodeWrapper
class Tracking3DParameterNode:
    """
    The parameters needed by module.

    inputVolumeSequence - The volume sequence.

    """

    # inputHierarchyRootID: str
    inputVolumeSequence: vtkMRMLSequenceNode


#
# Tracking3DWidget
#
class Tracking3DWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Tracking3D.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = Tracking3DLogic()
        self.logic.parameterFile = self.resourcePath("ParameterFiles/rigid.txt")

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Sets the frame slider range to be the number of nodes within the sequence
        self.ui.inputSelectorCT.connect("currentNodeChanged(vtkMRMLNode*)", self.updateFrameSlider)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.exportButton.connect("clicked(bool)", self.onExportButton)

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

    def setParameterNode(self, inputParameterNode: Optional[Tracking3DParameterNode]) -> None:
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

                    self.logic.registerSequence(CT, rootID, startFrame, endFrame)
                finally:
                    self.inProgress = False
        self.updateApplyButtonState()
        slicer.util.messageBox("Success!")

    def onExportButton(self):
        """UI button for writing the sequences as TRA files."""
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            currentRootIDStatus = self.ui.SubjectHierarchyComboBox.currentItem() != 0
            if not currentRootIDStatus:  # TODO: Remove this once this is working with the parameterNodeWrapper
                raise ValueError("Invalid hierarchy object selected!")

            rootID = self.ui.SubjectHierarchyComboBox.currentItem()
            rootNode = TreeNode(hierarchyID=rootID, ctSequence=None, isRoot=True)

            node_list = rootNode.childNodes.copy()
            for child in node_list:
                child.exportTransformsAsTRAFile()
                node_list.extend(child.childNodes)
        slicer.util.messageBox("Success!")

    def updateFrameSlider(self, CTSelectorNode: slicer.vtkMRMLNode):
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


#
# Tracking3DLogic
#


class Tracking3DLogic(ScriptedLoadableModuleLogic):
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
        self.elastixLogic = Elastix.ElastixLogic()
        self.autoscoperLogic = AutoscoperM.AutoscoperMLogic()
        self.cancelRequested = False
        self.isRunning = False

    def getParameterNode(self):
        return Tracking3DParameterNode(super().getParameterNode())

    def createROIFromPV(self, body: vtkMRMLScalarVolumeNode, padding: int = 0) -> vtkMRMLMarkupsROINode:
        """Creates a ROI from a volume."""
        # TODO: Expose padding to GUI -> Maybe independent X,Y,Z values?
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetInputVolumeNodeID(body.GetID())
        cropVolumeParameters.SetROINodeID(roiNode.GetID())
        slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(cropVolumeParameters)
        slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(cropVolumeParameters)

        size = list(roiNode.GetSize())
        size = [s + padding for s in size]
        roiNode.SetSize(size)
        return roiNode

    def cropCT(self, CT: vtkMRMLScalarVolumeNode, roi: vtkMRMLMarkupsROINode) -> vtkMRMLScalarVolumeNode:
        """Crops a volume with a ROI."""
        cropVolumeLogic = slicer.modules.cropvolume.logic()
        cropVolumeParameterNode = slicer.vtkMRMLCropVolumeParametersNode()
        cropVolumeParameterNode.SetROINodeID(roi.GetID())
        cropVolumeParameterNode.SetInputVolumeNodeID(CT.GetID())
        cropVolumeParameterNode.SetVoxelBased(True)
        cropVolumeLogic.Apply(cropVolumeParameterNode)
        return slicer.mrmlScene.GetNodeByID(cropVolumeParameterNode.GetOutputVolumeNodeID())

    def registerRigidBody(
        self,
        CT: vtkMRMLScalarVolumeNode,
        body: vtkMRMLScalarVolumeNode,
        outputTransform: vtkMRMLTransformNode,
    ):
        """Registers a partial volume to a CT scan, uses SlicerElastix."""
        roi = self.createROIFromPV(body)
        body.SetAndObserveTransformNodeID(outputTransform.GetID())
        roi.SetAndObserveTransformNodeID(outputTransform.GetID())

        croppedCT = self.cropCT(CT, roi)

        self.elastixLogic.registerVolumes(
            fixedVolumeNode=croppedCT,
            movingVolumeNode=body,
            parameterFilenames=[self.parameterFile],
            outputVolumeNode=None,
            outputTransformNode=outputTransform,
            fixedVolumeMaskNode=None,
            movingVolumeMaskNode=None,
            forceDisplacementFieldOutputTransform=False,
            initialTransformNode=None,
        )

        # Clean up
        slicer.mrmlScene.RemoveNode(croppedCT)
        slicer.mrmlScene.RemoveNode(roi)

    def registerSequence(
        self,
        ctSequence: vtkMRMLSequenceNode,
        rootID: int,
        startFrame: int,
        endFrame: int,
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
                    node.dataNode.SetAndObserveTransformNodeID(None)
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
                    node.applyTransformToChildren(idx)
                    nodeList.extend(node.childNodes)

                    node.dataNode.SetAndObserveTransformNodeID(node.getTransform(idx).GetID())

                # Use the output of the roots children as the initial guess for next frame
                if idx != endFrame - 1:  # Unless its the last frame
                    [node.copyTransformToNextFrame(idx) for node in rootNode.childNodes]

        finally:
            self.isRunning = False
            self.cancelRequested = False

    def evalulateMetric(
        self, fixedImage: vtkMRMLScalarVolumeNode, movingImage: vtkMRMLScalarVolumeNode
    ) -> dict[str, float]:
        """Computes several metrics for the similarity of the fixed and moving images."""
        import SimpleITK as sitk
        import sitkUtils

        fixedSITKImg = sitkUtils.PullVolumeFromSlicer(fixedImage)
        movingSITKImg = sitkUtils.PullVolumeFromSlicer(movingImage)
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkFloat64)
        fixedSITKImg = castFilter.Execute(fixedSITKImg)
        movingSITKImg = castFilter.Execute(movingSITKImg)
        R = sitk.ImageRegistrationMethod()
        results = {}

        R.SetMetricAsMattesMutualInformation(64)  # 64 bins
        results["mutualInformation"] = R.MetricEvaluate(fixedSITKImg, movingSITKImg)

        R.SetMetricAsMeanSquares()
        results["meanSquares"] = R.MetricEvaluate(fixedSITKImg, movingSITKImg)

        R.SetMetricAsCorrelation()
        results["correlation"] = R.MetricEvaluate(fixedSITKImg, movingSITKImg)

        R.SetMetricAsJointHistogramMutualInformation(64)
        results["histogramMutualInformation"] = R.MetricEvaluate(fixedSITKImg, movingSITKImg)

        return results

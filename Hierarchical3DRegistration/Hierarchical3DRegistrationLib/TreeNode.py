from __future__ import annotations

import logging
import os

import slicer
import vtk

from AutoscoperM import IO, AutoscoperMLogic


class TreeNode:
    """
    Data structure to store a basic tree hierarchy.
    """

    def __init__(
        self,
        hierarchyID: int,
        ctSequence: slicer.vtkMRMLSequenceNode,
        parent: TreeNode | None = None,
        isRoot: bool = False,
    ):
        self.hierarchyID = hierarchyID
        self.isRoot = isRoot
        self.parent = parent

        if self.parent is not None and self.isRoot:
            raise ValueError("Node cannot be root and have a parent")

        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.name = self.shNode.GetItemName(self.hierarchyID)
        self.model = self.shNode.GetItemDataNode(self.hierarchyID)

        if self.model.GetClassName() != "vtkMRMLModelNode":
            raise ValueError(f"Hierarchy item '{self.name}' is not a model!")

        self.roi = self._generateRoiFromModel()
        self.transformSequence = self._initializeTransforms(ctSequence)
        self.croppedCtSequence = dict() #self._initializeCroppedCT(ctSequence)

        children_ids = []
        self.shNode.GetItemChildren(self.hierarchyID, children_ids)
        self.childNodes = [
            TreeNode(hierarchyID=child_id, parent=self, ctSequence=ctSequence) for child_id in children_ids
        ]

    def _generateRoiFromModel(self) -> slicer.vtkMRMLMarkupsROINode:
        """Creates a region of interest node from this TreeNode's model."""
        mBounds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.model.GetBounds(mBounds)

        import numpy as np

        # construct min and max coordinates of bounding box
        bb_min = np.array([mBounds[0], mBounds[2], mBounds[4]])
        bb_max = np.array([mBounds[1], mBounds[3], mBounds[5]])

        bb_center = (bb_min + bb_max) / 2
        bb_size = bb_min - bb_max

        # Create ROI node
        modelROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        modelROI.SetCenter(bb_center.tolist())
        modelROI.SetSize(bb_size.tolist())
        modelROI.SetDisplayVisibility(0)
        # modelROI.CreateDefaultDisplayNodes()  # only needed for display, TODO: hide
        # modelROI.SetAttribute("Markups.MovingInSliceView", "");
        # modelROI.SetAttribute("Markups.MovingMarkupIndex", "");
        modelROI.SetName(f"{self.name}_roi")

        return modelROI

    def _initializeTransforms(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new transform sequence in the same browser as the CT sequence."""
        # TODO: make sure output transforms start at the appropriate if startIdx != 0.

        newSequenceNode = AutoscoperMLogic.createSequenceNodeInBrowser(f"{self.name}_transform_sequence", ctSequence)
        identityTfm = slicer.mrmlScene.CreateNodeByClass("vtkMRMLLinearTransformNode")

        # batch the processing event for the addition of the transforms, for speedup
        slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)

        for i in range(ctSequence.GetNumberOfDataNodes()):
            idxValue = ctSequence.GetNthIndexValue(i)
            newSequenceNode.SetDataNodeAtValue(identityTfm, idxValue)

        slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
        slicer.app.processEvents()
        return newSequenceNode
        """
        nodes = []
        for i in range(ctSequence.GetNumberOfDataNodes()):
            curTfm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{self.name}-{i}")
            newSequenceNode.SetDataNodeAtValue(curTfm, f"{i}")
            nodes.append(curTfm)
        [slicer.mrmlScene.RemoveNode(node) for node in nodes]
        # Bit of a strange issue but the browser doesn't seem to update unless it moves to a new index,
        # so we force it to update here
        AutoscoperMLogic.getItemInSequence(newSequenceNode, 1)
        AutoscoperMLogic.getItemInSequence(newSequenceNode, 0)
        slicer.app.processEvents()
        return newSequenceNode
        """

    def _initializeCroppedCT(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new (but empty) volume sequence in the same browser as the CT sequence."""

        return AutoscoperMLogic.createSequenceNodeInBrowser(f"{self.name}_cropped_CT_sequence", ctSequence)

    def setupFrame(self, frameIdx, ctFrame) -> None:
        """
        Return a cropped volume from the given CT frame based on the initial guess
        transform for the given frame and this model's ROI.
        """
        initial_tfm = self.getTransform(frameIdx)  # for the root node, this will just be the identity
        # generate cropped volume from the given frame
        self.model.SetAndObserveTransformNodeID(initial_tfm.GetID())
        self.roi.SetAndObserveTransformNodeID(initial_tfm.GetID())
        self.cropFrameFromRoi(frameIdx, ctFrame)

    def cropFrameFromRoi(self, frame_idx, targetFrame) -> None:
        """
        Returns a cropped volume from the given target volume, based its
        corresponding initial guess transform and this model's ROI.

        :param frame_idx: the frame index corresponding to the target frame
        :param targetFrame: the target volume to be cropped from the ROI

        :return: the output cropped volume node
        """
        # create volume node for the output of the cropping, and add it to the sequence
        #outputVolumeNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLScalarVolumeNode")

        # initialize croppping configuration
        cvpn = slicer.vtkMRMLCropVolumeParametersNode()
        cvpn.SetROINodeID(self.roi.GetID())
        cvpn.SetInputVolumeNodeID(targetFrame.GetID())
        #cvpn.SetOutputVolumeNodeID(outputVolumeNode.GetID())
        cvpn.SetVoxelBased(True)

        # apply the cropping
        cropLogic = slicer.modules.cropvolume.logic()
        cropLogic.Apply(cvpn)

        outputVolumeNodeID = cvpn.GetOutputVolumeNodeID()
        outputVolumeNode = slicer.mrmlScene.GetNodeByID(outputVolumeNodeID)

        outputVolumeNode.SetName(f"{targetFrame.GetName()}_{frame_idx}_{self.name}_cropped")
        #self.croppedCtSequence.SetDataNodeAtValue(outputVolumeNode, str(frame_idx))
        self.croppedCtSequence[frame_idx] = outputVolumeNode
        # TODO: remove new node from scene (we want it visible just inside the sequence...)?

        return outputVolumeNode

    def getTransform(self, idx: int) -> slicer.vtkMRMLTransformNode:
        """Returns the transform at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]

    def getCroppedFrame(self, idx: int) -> slicer.vtkMRMLScalarVolumeNode:
        """if idx >= self.croppedCtSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return self.croppedCtSequence.GetNthDataNode(idx)"""
        if idx not in self.croppedCtSequence:
            logging.warning(f"Provided index {idx} not found in the sequence od cropped volumes.")
            return None
        return self.croppedCtSequence[idx]

    def _applyTransform(self, idx: int, transform: slicer.vtkMRMLTransformNode) -> None:
        """Applies and hardends a transform node to the transform sequence at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetAndObserveTransformNodeID(transform.GetID()) # TODO: fails for parent! parent transform cannot be self or a child transform
        current_transform.HardenTransform()

    def setTransformFromMatrix(self, transform: vtk.vtkMatrix4x4, idx: int) -> None:  # TODO: revisit for import
        """TODO description"""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetMatrixTransformToParent(transform)

    def applyTransformToChildren(self, idx: int, transform: slicer.vtkMRMLLinearTransformNode) -> None:
        """Applies the transform at the provided index to this node and all its children."""
        for childNode in self.childNodes:
            childNode._applyTransform(idx, transform)
            # recurse down all child nodes and apply it to them as well
            childNode.applyTransformToChildren(idx, transform)

    def copyTransformToNextFrame(self, currentIdx: int) -> None:
        """Copies the transform at the provided index to the next frame."""
        import vtk

        currentTransform = self.getTransform(currentIdx)
        transformMatrix = vtk.vtkMatrix4x4()
        currentTransform.GetMatrixTransformToParent(transformMatrix)

        nextIdx = currentIdx + 1
        nextTransform = self.getTransform(nextIdx)
        if nextTransform is not None:
            nextTransform.SetMatrixTransformToParent(transformMatrix)
        else:
            logging.error(f"DEBUGGING copyTransformToNextFrame: nextTransform is None at nextIdx={nextIdx}")

    def exportTransformsAsTRAFile(self, exportDir: str):  # TODO: revisit for export
        """Exports the sequence as a TRA file for reading into Autoscoper."""
        # Convert the sequence to a list of vtkMatrices
        transforms = []
        for idx in range(self.transformSequence.GetNumberOfDataNodes()):
            mat = vtk.vtkMatrix4x4()
            node = self.getTransform(idx)
            node.GetMatrixTransformToParent(mat)
            transforms.append(mat)

        if not os.path.exists(exportDir):
            os.mkdir(exportDir)
        filename = os.path.join(exportDir, f"{self.name}-abs-RAS.tra")
        IO.writeTRA(filename, transforms)

    def importTransfromsFromTRAFile(self, filename: str):  # TODO: revisit for import
        """TODO description"""
        import numpy as np

        tra = np.loadtxt(filename, delimiter=",")
        tra.resize(tra.shape[0], 4, 4)
        for idx in range(tra.shape[0]):
            self.setTransformFromMatrix(slicer.util.vtkMatrixFromArray(tra[idx, :, :]), idx)

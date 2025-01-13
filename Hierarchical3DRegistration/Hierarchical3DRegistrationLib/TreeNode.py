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

        self.roi = self._generateRoiFromModel()
        self.transformSequence = self._initializeTransforms(ctSequence)
        self.croppedCtSequence = self._initializeCroppedCT(ctSequence)

        children_ids = []
        self.shNode.GetItemChildren(self.hierarchyID, children_ids)
        self.childNodes = [
            TreeNode(hierarchyID=child_id, parent=self, ctSequence=ctSequence) for child_id in children_ids
        ]

    def _generateRoiFromModel(self) -> slicer.vtkMRMLMarkupsROINode:
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
        #modelROI.CreateDefaultDisplayNodes()  # only needed for display, TODO: hide
        #modelROI.SetAttribute("Markups.MovingInSliceView", "");
        #modelROI.SetAttribute("Markups.MovingMarkupIndex", "");
        modelROI.SetName(f"{self.name}_roi")

        return modelROI

    def _initializeTransforms(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new transform sequence in the same browser as the CT sequence."""
        #TODO: make sure output transforms start at the appropriate if startIdx != 0.

        newSequenceNode = AutoscoperMLogic.createSequenceNodeInBrowser(
            f"{self.name}_transform_sequence", ctSequence
        )
        identityTfm = slicer.mrmlScene.CreateNodeByClass("vtkMRMLLinearTransformNode")

        # batch the processing event for the addition of the transforms, for speedup
        slicer.mrmlScene.StartState(slicer.vtkMRMLScene.BatchProcessState)

        for i in range(ctSequence.GetNumberOfDataNodes()):
            idxValue = ctSequence.GetNthIndexValue(i)
            newSequenceNode.SetDataNodeAtValue(identityTfm, idxValue)

        slicer.mrmlScene.EndState(slicer.vtkMRMLScene.BatchProcessState)
        slicer.app.processEvents()
        return newSequenceNode

    def _initializeCroppedCT(self, ctSequence) -> slicer.vtkMRMLSequenceNode:
        """Creates a new (but empty) volume sequence in the same browser as the CT sequence."""

        newSequenceNode = AutoscoperMLogic.createSequenceNodeInBrowser(
            f"{self.name}_cropped_CT_sequence", ctSequence
        )

        return newSequenceNode

    def setupFrame(self, frameIdx, ctFrame) -> None:
        """
        TODO description
        """
        # prompt user to adjust initial guess, then crop target volume
        initial_tfm = self.getTransform(frameIdx)  # for root, this will just be the identity
        self.model.SetAndObserveTransformNodeID(initial_tfm.GetID())
        #initial_tfm.user_adjust_model() # optionally user adjusts initial guess for this bone in initial frame

        # generate cropped volume from frame
        self.roi.SetAndObserveTransformNodeID(initial_tfm.GetID())
        self.cropFrameFromRoi(frameIdx, ctFrame)

        return initial_tfm

    def cropFrameFromRoi(self, frame_idx, targetFrame) -> None:
        """
        TODO description
        """
        outputVolumeNode = self.croppedCtSequence.GetDataNodeAtValue(frame_idx, exactMatchRequired=True)
        if not outputVolumeNode:
            # create volume node for the output of the cropping, and add it to the sequence
            outputVolumeNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeNode")
            #outputVolumeNode.SetName(f"{targetFrame.GetName()}_{self.name}_cropped")
            self.croppedCtSequence.SetDataNodeAtValue(outputVolumeNode, frame_idx)

            # initialize croppping configuration
            cvpn = slicer.vtkMRMLCropVolumeParametersNode()
            cvpn.SetROINodeID(self.roi.GetID())
            cvpn.SetInputVolumeNodeID(targetFrame.GetID())
            cvpn.SetOutputVolumeNodeID(outputVolumeNode.GetID())
            cvpn.SetVoxelBased(True)

            # apply the cropping
            cropLogic = slicer.modules.cropvolume.logic()
            cropLogic.Apply(cvpn)

            # display pretty:
            # https://www.slicer.org/wiki/Documentation/4.3/Developers/Python_scripting
            views = slicer.app.layoutManager().sliceViewNames()
            for view in views:
                view_logic = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
                view_cn = view_logic.GetSliceCompositeNode()
                view_cn.SetBackgroundVolumeID(outputVolumeNode.GetID())
                view_logic.FitSliceToAll()
            # TODO: hide?

        return outputVolumeNode

    def getTransform(self, idx: int) -> slicer.vtkMRMLTransformNode:
        """Returns the transform at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]

    def _applyTransform(self, idx: int, transform: slicer.vtkMRMLTransformNode) -> None:
        """Applies and hardends a transform node to the transform in the sequence at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetAndObserveTransformNodeID(transform.GetID())
        current_transform.HardenTransform()

    def setTransformFromNode(self, transform: slicer.vtkMRMLLinearTransformNode, idx: int) -> None:
        """Sets the transform for the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        mat = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(mat)
        current_transform = AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetMatrixTransformToParent(mat)

    def setTransformFromMatrix(self, transform: vtk.vtkMatrix4x4, idx: int) -> None: # TODO: revisit for import
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = AutoscoperMLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetMatrixTransformToParent(transform)

    def applyTransformToTree(self, idx: int, transform: slicer.vtkMRMLLinearTransformNode) -> None:
        """Applies the transform at the provided index to this node and all its children."""
        # apply the transform first to this node
        self._applyTransform(idx, transform)
        # recurse down all child nodes and apply it to them as well
        [childNode.applyTransformToTree(idx, transform) for childNode in self.childNodes]

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

    def exportTransformsAsTRAFile(self, exportDir: str): # TODO: revisit for export
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

    def importTransfromsFromTRAFile(self, filename: str): # TODO: revisit for import
        import numpy as np

        tra = np.loadtxt(filename, delimiter=",")
        tra.resize(tra.shape[0], 4, 4)
        for idx in range(tra.shape[0]):
            self.setTransformFromMatrix(slicer.util.vtkMatrixFromArray(tra[idx, :, :]), idx)

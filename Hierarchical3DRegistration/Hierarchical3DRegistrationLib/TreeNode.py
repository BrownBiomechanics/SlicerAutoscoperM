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
        self.ctSequence = ctSequence

        if self.parent is not None and self.isRoot:
            raise ValueError("Node cannot be root and have a parent")

        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.autoscoperLogic = AutoscoperMLogic()

        self.name = self.shNode.GetItemName(self.hierarchyID)
        self.dataNode = self.shNode.GetItemDataNode(self.hierarchyID)
        self.transformSequence = self._initializeTransforms()

        children_ids = []
        self.shNode.GetItemChildren(self.hierarchyID, children_ids)
        self.childNodes = [
            TreeNode(hierarchyID=child_id, ctSequence=self.ctSequence, parent=self) for child_id in children_ids
        ]

    def _initializeTransforms(self) -> slicer.vtkMRMLSequenceNode:
        """Creates a new transform sequence in the same browser as the CT sequence."""
        import logging

        try:
            logging.info(f"Searching for {self.name} transforms")
            newSequenceNode = slicer.util.getNode(f"{self.name}_transform_sequence")
        except slicer.util.MRMLNodeNotFoundException:
            try:  # Loading the sequence transforms from seq and tfm files can be wacky
                logging.info(f"Searching for {self.name} transforms")
                newSequenceNode = slicer.util.getNode(
                    f"{self.name}_transform_sequence-{self.name}_transform_sequence-Seq"
                )
            except slicer.util.MRMLNodeNotFoundException:
                logging.info(f"Transforms not found, Initializing {self.name}")
                newSequenceNode = self.autoscoperLogic.createSequenceNodeInBrowser(
                    f"{self.name}_transform_sequence", self.ctSequence
                )
                nodes = []
                for i in range(self.ctSequence.GetNumberOfDataNodes()):
                    curTfm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{self.name}-{i}")
                    newSequenceNode.SetDataNodeAtValue(curTfm, f"{i}")
                    nodes.append(curTfm)
                [slicer.mrmlScene.RemoveNode(node) for node in nodes]

                # Bit of a strange issue but the browser doesn't seem to update unless it moves to a new index,
                # so we force it to update here
                self.autoscoperLogic.getItemInSequence(newSequenceNode, 1)
                self.autoscoperLogic.getItemInSequence(newSequenceNode, 0)

                slicer.app.processEvents()
        return newSequenceNode

    def _applyTransform(self, transform: slicer.vtkMRMLTransformNode, idx: int) -> None:
        """Applies and hardends a transform node to the transform in the sequence at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetAndObserveTransformNodeID(transform.GetID())
        current_transform.HardenTransform()

    def getTransform(self, idx: int) -> slicer.vtkMRMLTransformNode:
        """Returns the transform at the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return None
        return self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]

    def setTransformFromNode(self, transform: slicer.vtkMRMLLinearTransformNode, idx: int) -> None:
        """Sets the transform for the provided index."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        mat = vtk.vtkMatrix4x4()
        transform.GetMatrixTransformToParent(mat)
        current_transform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetMatrixTransformToParent(mat)

    def setTransformFromMatrix(self, transform: vtk.vtkMatrix4x4, idx: int) -> None:
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        current_transform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
        current_transform.SetMatrixTransformToParent(transform)

    def applyTransformToChildren(self, idx: int) -> None:
        """Applies the transform at the provided index to all children of this node."""
        if idx >= self.transformSequence.GetNumberOfDataNodes():
            logging.warning(f"Provided index {idx} is greater than number of data nodes in the sequence.")
            return
        applyTransform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
        [childNode.setTransformFromNode(applyTransform, idx) for childNode in self.childNodes]

    def copyTransformToNextFrame(self, currentIdx: int) -> None:
        """Copies the transform at the provided index to the next frame."""
        import vtk

        currentTransform = self.getTransform(currentIdx)
        transformMatrix = vtk.vtkMatrix4x4()
        currentTransform.GetMatrixTransformToParent(transformMatrix)
        nextTransform = self.getTransform(currentIdx + 1)
        if nextTransform is not None:
            nextTransform.SetMatrixTransformToParent(transformMatrix)

    def exportTransformsAsTRAFile(self, exportDir: str):
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

    def importTransfromsFromTRAFile(self, filename: str):
        import numpy as np

        tra = np.loadtxt(filename, delimiter=",")
        tra.resize(tra.shape[0], 4, 4)
        for idx in range(tra.shape[0]):
            self.setTransformFromMatrix(slicer.util.vtkMatrixFromArray(tra[idx, :, :]), idx)

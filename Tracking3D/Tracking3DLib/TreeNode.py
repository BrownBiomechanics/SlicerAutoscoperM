from __future__ import annotations

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
        if idx < self.transformSequence.GetNumberOfDataNodes():
            current_transform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
            current_transform.SetAndObserveTransformNodeID(transform.GetID())
            current_transform.HardenTransform()

    def getTransform(self, idx: int) -> slicer.vtkMRMLTransformNode:
        """Returns the transform at the provided index."""
        if idx < self.transformSequence.GetNumberOfDataNodes():
            return self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
        return None

    def setTransform(self, transform: slicer.vtkMRMLLinearTransformNode, idx: int) -> None:
        """Sets the transform for the provided index."""
        if idx < self.transformSequence.GetNumberOfDataNodes():
            mat = vtk.vtkMatrix4x4()
            transform.GetMatrixTransformToParent(mat)
            current_transform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
            current_transform.SetMatrixTransformToParent(mat)

    def applyTransformToChildren(self, idx: int) -> None:
        """Applies the transform at the provided index to all children of this node."""
        if idx < self.transformSequence.GetNumberOfDataNodes():
            applyTransform = self.autoscoperLogic.getItemInSequence(self.transformSequence, idx)[0]
            [childNode.setTransform(applyTransform, idx) for childNode in self.childNodes]

    def copyTransformToNextFrame(self, currentIdx: int) -> None:
        """Copies the transform at the provided index to the next frame."""
        import vtk

        currentTransform = self.getTransform(currentIdx)
        transformMatrix = vtk.vtkMatrix4x4()
        currentTransform.GetMatrixTransformToParent(transformMatrix)
        nextTransform = self.getTransform(currentIdx + 1)
        if nextTransform is not None:
            nextTransform.SetMatrixTransformToParent(transformMatrix)

    def exportTransformsAsTRAFile(self):
        """Exports the sequence as a TRA file for reading into Autoscoper."""
        # Convert the sequence to a list of vtkMatrices
        transforms = []
        for idx in range(self.transformSequence.GetNumberOfDataNodes()):
            mat = vtk.vtkMatrix4x4()
            node = self.getTransform(idx)
            node.GetMatrixTransformToParent(mat)
            transforms.append(mat)

        # Apply the Slicer2Autoscoper Transform to the neutral frame
        bounds = [0] * 6
        self.dataNode.GetRASBounds(bounds)
        volSize = [abs(bounds[i + 1] - bounds[i]) for i in range(0, len(bounds), 2)]
        origin = self.dataNode.GetOrigin()
        transforms[0] = self.autoscoperLogic.applySlicer2AutoscoperTransform(transforms[0], volSize, origin)

        # Since each additional transform is the change from the neutral position
        # we need to apply each additional transform to the neutral to get the final transform
        neutralArray = self.autoscoperLogic.vtkToNumpy(transforms[0])
        for idx in range(1, self.transformSequence.GetNumberOfDataNodes()):
            currentArray = self.autoscoperLogic.vtkToNumpy(transforms[idx])
            currentArray = currentArray @ neutralArray
            transforms[idx] = self.autoscoperLogic.numpyToVtk(currentArray)

        # Move each transform from the DICOM space into AUT space
        # TODO: Get this file from user
        o2dFile = r"AutoscoperM-Pre-Processing\Autoscoper Scene\Transforms\Origin2Dicom.tfm"
        o2dFile = os.path.join(slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory(), o2dFile)
        transforms = [self.autoscoperLogic.applyOrigin2DicomTransform(tfm, o2dFile) for tfm in transforms]

        # Write out tra
        # TODO: Make the directory user defined
        exportDir = r"AutoscoperM-Pre-Processing\Tracking"
        exportDir = os.path.join(slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory(), exportDir)
        if not os.path.exists(exportDir):
            os.mkdir(exportDir)
        filename = os.path.join(exportDir, f"{self.name}.tra")
        IO.writeTRA(filename, transforms)

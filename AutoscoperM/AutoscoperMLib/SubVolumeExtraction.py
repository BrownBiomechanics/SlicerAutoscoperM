import logging
from typing import Optional

import slicer
import vtk


def automaticSegmentation(
    volumeNode: slicer.vtkMRMLVolumeNode,
    threshold: int,
    marginSize: int,
    segmentationName: Optional[str] = None,
    progressCallback: Optional[callable] = None,
    maxProgressValue: int = 100,
) -> slicer.vtkMRMLSegmentationNode:
    """
    Automatic segmentation of the volume node using the threshold value.

    :param volumeNode: Volume node
    :param threshold: Threshold value
    :param marginSize: Margin size
    :param segmentationName: Segmentation name. Default is None.
    :param progressCallback: Progress callback. Default is None.
    :param maxProgressValue: Maximum progress value. Default is 100.

    :return: Segmentation node
    """
    if progressCallback is None:
        logging.warning("[AutoscoperMLib.SubVolumeExtraction.automaticSegmentation] No progress bar callback given.")

        def progressCallback(x):
            return x

    # Create segmentation node
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    if segmentationName:
        segmentationNode.SetName(segmentationName)
    segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
    if segmentationName:  # Add an empty segment with the given name
        segmentationNode.GetSegmentation().AddEmptySegment(segmentationName)
    else:
        segmentationNode.GetSegmentation().AddEmptySegment()

    # Create segment editor to get access to effects
    segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentationEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentationEditorWidget.setSegmentationNode(segmentationNode)
    segmentationEditorWidget.setSourceVolumeNode(volumeNode)

    # Thresholding
    segmentationEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold", threshold)
    effect.self().onApply()

    progressCallback(5 / 100 * maxProgressValue)

    # Island - Split Islands into Segments
    segmentationEditorWidget.setActiveEffectByName("Islands")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("Operation", "SPLIT_ISLANDS_TO_SEGMENTS")
    effect.self().onApply()

    progressCallback(10 / 100 * maxProgressValue)

    inputSegmentIDs = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(inputSegmentIDs)

    # Fill Holes
    segmentEditorNode.SetOverwriteMode(slicer.vtkMRMLSegmentEditorNode.OverwriteNone)
    segmentEditorNode.SetMaskMode(slicer.vtkMRMLSegmentationNode.EditAllowedEverywhere)

    numSegments = inputSegmentIDs.GetNumberOfValues()
    for i in range(numSegments):
        segmentID = inputSegmentIDs.GetValue(i)
        _fillHole(segmentID, segmentationEditorWidget, marginSize)
        progress = ((i + 1) / numSegments) * 90 + 10
        progress = progress / 100 * maxProgressValue
        progressCallback(progress)

    # Clean up
    segmentationEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    return segmentationNode


def extractSubVolume(
    volumeNode: slicer.vtkMRMLVolumeNode,
    segmentationNode: slicer.vtkMRMLVolumeNode,
    segmentID: Optional[str] = None,
) -> slicer.vtkMRMLVolumeNode:
    """
    Extracts the subvolume from the volume node using the segmentation node.

    :param volumeNode: Volume node
    :param segmentationNode: Segmentation node
    :param segmentID: Segment ID. Default is None.

    :return: Subvolume node.
    """
    # Create segment editor to get access to effects
    segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentationEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentationEditorWidget.setSegmentationNode(segmentationNode)
    segmentationEditorWidget.setSourceVolumeNode(volumeNode)

    if segmentID:
        segmentationEditorWidget.setCurrentSegmentID(segmentID)
    else:
        segmentIDs = vtk.vtkStringArray()
        segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(segmentIDs)
        segmentID = segmentIDs.GetValue(0)

    segmentationEditorWidget.setActiveEffectByName("Split volume")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("PaddingVoxels", 0)
    effect.setParameter("ApplyToAllVisibleSegments", 0)
    effect.self().onApply()

    folderName = volumeNode.GetName() + " split"

    return _getItemFromFolder(folderName)


def mergeSegments(
    volumeNode: slicer.vtkMRMLVolumeNode,
    segmentationNode: slicer.vtkMRMLSegmentationNode,
    newSegmentationNode: bool = True,
) -> Optional[slicer.vtkMRMLSegmentationNode]:
    """ "
    Merges all segments in the segmentation node into one segment.
    If newSegmentationNode is True, a new segmentation node is created.
    Otherwise the merge is performed in place on the given segmentation node.

    :param volumeNode: Volume node
    :param segmentationNode: Segmentation node.
    :param newSegmentationNode: If True, a new node is created. Otherwise it is performed in place. Default is True.

    :return: Segmentation node with the merged segments. If newSegmentationNode is False, None is returned.
    """
    mergeNode = segmentationNode
    if newSegmentationNode:
        mergeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        mergeNode.CreateDefaultDisplayNodes()  # only needed for display
        mergeNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        # copy segments over
        for i in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
            segment = segmentationNode.GetSegmentation().GetNthSegment(i)
            mergeNode.GetSegmentation().CopySegmentFromSegmentation(
                segmentationNode.GetSegmentation(), segment.GetName()
            )
        mergeNode.SetName(segmentationNode.GetName() + " merged")

    # Create segment editor to get access to effects
    segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentationEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentationEditorWidget.setSegmentationNode(mergeNode)
    segmentationEditorWidget.setSourceVolumeNode(volumeNode)

    # Merge Segments
    inputSegmentIDs = vtk.vtkStringArray()
    mergeNode.GetDisplayNode().GetVisibleSegmentIDs(inputSegmentIDs)

    segmentationEditorWidget.setCurrentSegmentID(inputSegmentIDs.GetValue(0))
    for i in range(1, inputSegmentIDs.GetNumberOfValues()):
        segmentID_to_add = inputSegmentIDs.GetValue(i)

        # Combine all segments into one
        segmentationEditorWidget.setActiveEffectByName("Logical operators")
        effect = segmentationEditorWidget.activeEffect()
        effect.setParameter("Operation", "UNION")
        effect.setParameter("BypassMasking", 1)
        effect.setParameter("ModifierSegmentID", segmentID_to_add)
        effect.self().onApply()

        # delete the segment
        mergeNode.GetSegmentation().RemoveSegment(segmentID_to_add)

    # Clean up
    segmentationEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)

    if newSegmentationNode:
        return mergeNode
    return None


def _fillHole(segmentID: str, segmentationEditorWidget: slicer.qMRMLSegmentEditorWidget, marginSize: int) -> None:
    """
     Fills internal holes in the segment.

    :param segmentID: Segment ID
    :param segmentationEditorWidget: Segment editor widget
    :param marginSize: Margin size.
    """
    segmentationEditorWidget.setCurrentSegmentID(segmentID)

    segmentationEditorWidget.setActiveEffectByName("Margin")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("MarginSizeMm", marginSize)
    effect.self().onApply()

    # Logical operators - Invert
    segmentationEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("Operation", "INVERT")
    effect.self().onApply()

    # Island - Keep Largest Island
    segmentationEditorWidget.setActiveEffectByName("Islands")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    effect.self().onApply()

    # Margin
    segmentationEditorWidget.setActiveEffectByName("Margin")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("MarginSizeMm", marginSize)
    effect.self().onApply()

    # Logical operators - Invert
    segmentationEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentationEditorWidget.activeEffect()
    effect.setParameter("Operation", "INVERT")
    effect.self().onApply()


def _getItemFromFolder(folderName: str) -> slicer.vtkMRMLNode:
    """
    Gets the item from the folder and removes the folder.

    :param folderName: Name of the folder

    :return: Node in the folder
    """
    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    folderPlugin = pluginHandler.pluginByName("Folder")
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    folderId = shNode.GetItemByName(folderName)
    nodeId = shNode.GetItemByPositionUnderParent(folderId, 0)

    folderPlugin.setDisplayVisibility(folderId, 1)
    slicer.mrmlScene.RemoveNode(shNode.GetDisplayNodeForItem(folderId))  # remove the folder

    return shNode.GetItemDataNode(nodeId)  # return the node in the folder

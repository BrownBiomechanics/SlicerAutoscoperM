import logging
from typing import Optional

import slicer
import vtk


def automaticExtraction(
    volumeNode: slicer.vtkMRMLVolumeNode,
    threshold: int,
    segmentationName: Optional[str] = None,
    progressCallback: Optional[callable] = None,
    maxProgressValue: int = 100,
) -> slicer.vtkMRMLVolumeNode:
    """
    Automatic extraction of a sub volume node using the threshold value.

    :param volumeNode: Volume node
    :type volumeNode: slicer.vtkMRMLVolumeNode

    :param threshold: Threshold value
    :type threshold: int

    :param progressCallback: Progress callback. Default is None.
    :type progressCallback: callable

    :param maxProgressValue: Maximum progress value. Default is 100.
    :type maxProgressValue: int

    :return: SubVolume node
    :rtype: slicer.vtkMRMLVolumeNode
    """
    segmentationNode = automaticSegmentation(
        volumeNode, threshold, segmentationName, progressCallback, maxProgressValue
    )
    _mergeSegments(volumeNode, segmentationNode)
    return extractSubVolume(volumeNode, segmentationNode)


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
    :type volumeNode: slicer.vtkMRMLVolumeNode

    :param threshold: Threshold value
    :type threshold: int

    :param marginSize: Margin size
    :type marginSize: int

    :param segmentationName: Segmentation name. Default is None.
    :type segmentationName: str

    :param progressCallback: Progress callback. Default is None.
    :type progressCallback: callable

    :param maxProgressValue: Maximum progress value. Default is 100.
    :type maxProgressValue: int

    :return: Segmentation node
    :rtype: slicer.vtkMRMLSegmentationNode
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
    :type volumeNode: slicer.vtkMRMLVolumeNode

    :param segmentationNode: Segmentation node
    :type segmentationNode: slicer.vtkMRMLVolumeNode

    :param segmentID: Segment ID. Default is None.
    :type segmentID: str

    :return: Subvolume node.
    :rtype: slicer.vtkMRMLVolumeNode
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


def _mergeSegments(volumeNode: slicer.vtkMRMLVolumeNode, segmentationNode: slicer.vtkMRMLSegmentationNode) -> None:
    """ "
    Merges all segments in the segmentation node.

    :param volumeNode: Volume node
    :type volumeNode: slicer.vtkMRMLVolumeNode

    :param segmentationNode: Segmentation node.
    :type segmentationNode: slicer.vtkMRMLSegmentationNode
    """
    # Create segment editor to get access to effects
    segmentationEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentationEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentationEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentationEditorWidget.setSegmentationNode(segmentationNode)
    segmentationEditorWidget.setSourceVolumeNode(volumeNode)

    # Merge Segments
    inputSegmentIDs = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(inputSegmentIDs)

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
        segmentationNode.GetSegmentation().RemoveSegment(segmentID_to_add)

    # Clean up
    segmentationEditorWidget = None
    slicer.mrmlScene.RemoveNode(segmentEditorNode)


def _fillHole(segmentID: str, segmentationEditorWidget: slicer.qMRMLSegmentEditorWidget, marginSize: int) -> None:
    """
     Fills internal holes in the segment.

    :param segmentID: Segment ID
    :type segmentID: str

    :param segmentationEditorWidget: Segment editor widget
    :type segmentationEditorWidget: slicer.qMRMLSegmentEditorWidget

    :param marginSize: Margin size.
    :type marginSize: int
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
    :type folderName: str

    :return: Node in the folder
    :rtype: slicer.vtkMRMLNode
    """
    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    folderPlugin = pluginHandler.pluginByName("Folder")
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    folderId = shNode.GetItemByName(folderName)
    nodeId = shNode.GetItemByPositionUnderParent(folderId, 0)
    nodeName = shNode.GetItemName(nodeId)

    folderPlugin.setDisplayVisibility(folderId, 1)
    slicer.mrmlScene.RemoveNode(slicer.util.getNode(folderName))  # remove the folder

    return slicer.util.getNode(nodeName)  # return the node in the folder
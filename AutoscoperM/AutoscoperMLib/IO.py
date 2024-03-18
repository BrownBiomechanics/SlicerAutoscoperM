import glob
import logging
import os

import numpy as np
import slicer
import vtk

from .RadiographGeneration import Camera


def loadSegmentation(segmentationNode: slicer.vtkMRMLSegmentationNode, filename: str):
    """
    Load a segmentation file

    :param segmentationNode: Segmentation node
    :param filename: File name
    """
    # Get the extension
    extension = os.path.splitext(filename)[1]

    if extension == ".iv":  # Load this as an Open Inventor File
        modelNode = slicer.util.loadNodeFromFile(filename, "OpenInventorMesh")  # Requires SlicerSandBox extension
        # Import the model into the segmentation node
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
        # Clean up
        slicer.mrmlScene.RemoveNode(modelNode)
        return None

    try:  # If the filetype is not known try to load it as a segmentation
        return slicer.util.loadSegmentation(filename)
    except Exception as e:
        logging.error(f"Could not load {filename} \n {e}")
        return None


def generateCameraCalibrationFile(camera: Camera, filename: str):
    """
    Generates a VTK camera calibration json file from the given camera.

    :param camera: Camera
    :param filename: Output file name
    """
    import json

    contents = {}
    contents["@schema"] = "https://autoscoperm.slicer.org/vtkCamera-schema-1.0.json"
    contents["version"] = 1.0
    contents["focal-point"] = camera.vtkCamera.GetFocalPoint()
    contents["camera-position"] = camera.vtkCamera.GetPosition()
    contents["view-up"] = camera.vtkCamera.GetViewUp()
    contents["view-angle"] = camera.vtkCamera.GetViewAngle()
    contents["image-width"] = camera.imageSize[0]
    contents["image-height"] = camera.imageSize[1]
    contents["clipping-range"] = camera.vtkCamera.GetClippingRange()

    contents_json = json.dumps(contents, indent=4)

    with open(filename, "w+") as f:
        f.write(contents_json)


def generateConfigFile(
    mainDirectory: str,
    subDirectories: list[str],
    trialName: str,
    volumeFlip: list[int],
    voxelSize: list[float],
    renderResolution: list[int],
    optimizationOffsets: list[float],
) -> str:
    """
    Generates the v1.1 config file for the trial

    :param mainDirectory: Main directory
    :param subDirectories: Sub directories
    :param trialName: Trial name
    :param volumeFlip: Volume flip
    :param voxelSize: Voxel size
    :param renderResolution: Render resolution
    :param optimizationOffsets: Optimization offsets

    :return: Path to the config file
    """
    import datetime

    # Get the camera calibration files, camera root directories, and volumes
    volumes = glob.glob(os.path.join(mainDirectory, subDirectories[0], "*.tif"))
    cameraRootDirs = glob.glob(os.path.join(mainDirectory, subDirectories[1], "*"))
    calibrationFiles = glob.glob(os.path.join(mainDirectory, subDirectories[2], "*.json"))

    # Check that we have the same number of camera calibration files and camera root directories
    if len(calibrationFiles) != len(cameraRootDirs):
        logging.error(
            "Number of camera calibration files and camera root directories do not match: "
            " {len(calibrationFiles)} != {len(cameraRootDirs)}"
        )
        return None

    # Check that we have at least one volume
    if len(volumes) == 0:
        logging.error("No volumes found!")
        return None

    # Transform the paths to be relative to the main directory
    calibrationFiles = [os.path.relpath(calibrationFile, mainDirectory) for calibrationFile in calibrationFiles]
    cameraRootDirs = [os.path.relpath(cameraRootDir, mainDirectory) for cameraRootDir in cameraRootDirs]
    volumes = [os.path.relpath(volume, mainDirectory) for volume in volumes]

    with open(os.path.join(mainDirectory, trialName + ".cfg"), "w+") as f:
        # Trial Name as comment
        f.write(f"# {trialName} configuration file\n")
        f.write(
            "# This file was automatically generated by AutoscoperM on " + datetime.datetime.now().strftime("%c") + "\n"
        )
        f.write("\n")

        # Version of the cfg file
        f.write("Version 1.1\n")
        f.write("\n")

        # Camera Calibration Files
        f.write("# Camera Calibration Files\n")
        for calibrationFile in calibrationFiles:
            f.write("mayaCam_csv " + calibrationFile + "\n")
        f.write("\n")

        # Camera Root Directories
        f.write("# Camera Root Directories\n")
        for cameraRootDir in cameraRootDirs:
            f.write("CameraRootDir " + cameraRootDir + "\n")
        f.write("\n")

        # Volumes
        f.write("# Volumes\n")
        for volume in volumes:
            f.write("VolumeFile " + volume + "\n")
            f.write("VolumeFlip " + " ".join([str(x) for x in volumeFlip]) + "\n")
            f.write("VoxelSize " + " ".join([str(x) for x in voxelSize]) + "\n")
        f.write("\n")

        # Render Resolution
        f.write("# Render Resolution\n")
        f.write("RenderResolution " + " ".join([str(x) for x in renderResolution]) + "\n")
        f.write("\n")

        # Optimization Offsets
        f.write("# Optimization Offsets\n")
        f.write("OptimizationOffsets " + " ".join([str(x) for x in optimizationOffsets]) + "\n")
        f.write("\n")

    return os.path.join(mainDirectory, trialName + ".cfg")


def writeVolume(volumeNode: slicer.vtkMRMLVolumeNode, filename: str):
    """
    Writes a volumeNode to a file.

    :param volumeNode: Volume node
    :param filename: Output file name
    """
    slicer.util.exportNode(volumeNode, filename, {"useCompression": False}, world=True)


def castVolumeForTIFF(volumeNode: slicer.vtkMRMLVolumeNode):
    """
    Casts a volume node for writing to a TIFF file. This is necessary because Autoscoper
    only supports unsigned short TIFF stacks.

    :param volumeNode: Volume node
    """
    _castVolume(volumeNode, "Short")

    volumeArray = slicer.util.arrayFromVolume(volumeNode)
    minVal = np.min(volumeArray)
    if minVal < 0:
        minVal = -minVal
    isNotZero = volumeArray != 0  # Since 0 is the background value, we don't want to add minVal to it
    volumeArray[isNotZero] += minVal

    slicer.util.updateVolumeFromArray(volumeNode, volumeArray)

    _castVolume(volumeNode, "UnsignedShort")


def _castVolume(volumeNode: slicer.vtkMRMLVolumeNode, newType: str):
    """
    Internal function to cast a volume node to a new type
    """
    tmpVolNode = _createNewVolumeNode("tmpVolNode")
    castModule = slicer.modules.castscalarvolume
    parameters = {}
    parameters["InputVolume"] = volumeNode.GetID()
    parameters["OutputVolume"] = tmpVolNode.GetID()
    parameters["Type"] = newType  # Short to UnsignedShort
    cliNode = slicer.cli.runSync(castModule, None, parameters)
    slicer.mrmlScene.RemoveNode(cliNode)
    del cliNode, parameters, castModule

    volumeNode.SetAndObserveImageData(tmpVolNode.GetImageData())
    slicer.mrmlScene.RemoveNode(tmpVolNode)


def _createNewVolumeNode(nodeName: str) -> slicer.vtkMRMLVolumeNode:
    """
    Internal function to create a blank volume node
    """
    imageSize = [512, 512, 512]
    voxelType = vtk.VTK_UNSIGNED_CHAR
    imageOrigin = [0.0, 0.0, 0.0]
    imageSpacing = [1.0, 1.0, 1.0]
    imageDirections = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    fillVoxelValue = 0

    # Create an empty image volume, filled with fillVoxelValue
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(imageSize)
    imageData.AllocateScalars(voxelType, 1)
    imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
    # Create volume node
    volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
    volumeNode.SetOrigin(imageOrigin)
    volumeNode.SetSpacing(imageSpacing)
    volumeNode.SetIJKToRASDirections(imageDirections)
    volumeNode.SetAndObserveImageData(imageData)
    volumeNode.CreateDefaultDisplayNodes()
    return volumeNode


def writeTFMFile(filename: str, spacing: list[float], origin: list[float]):
    """
    Writes a TFM file

    :param filename: Output file name
    :param spacing: Spacing
    :param origin: Origin
    """

    tfm = vtk.vtkMatrix4x4()
    tfm.SetElement(0, 0, spacing[0])
    tfm.SetElement(1, 1, spacing[1])
    tfm.SetElement(2, 2, spacing[2])
    tfm.SetElement(0, 3, origin[0])
    tfm.SetElement(1, 3, origin[1])
    tfm.SetElement(2, 3, origin[2])

    transformNode = slicer.vtkMRMLLinearTransformNode()
    transformNode.SetMatrixTransformToParent(tfm)
    slicer.mrmlScene.AddNode(transformNode)

    slicer.util.exportNode(transformNode, filename)

    slicer.mrmlScene.RemoveNode(transformNode)


def writeTemporyFile(filename: str, data: vtk.vtkImageData) -> str:
    """
    Writes a temporary file to the slicer temp directory

    :param filename: Output file name
    :param data: data

    :return: Path to the file
    """

    slicerTempDirectory = slicer.app.temporaryPath

    # write vtk image data as a vtk file
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(os.path.join(slicerTempDirectory, filename))
    writer.SetInputData(data)
    writer.Write()
    return writer.GetFileName()


def removeTemporyFile(filename: str):
    """
    Removes a temporary file from the slicer temp directory

    :param filename: Output file name
    """

    slicerTempDirectory = slicer.app.temporaryPath
    os.remove(os.path.join(slicerTempDirectory, filename))

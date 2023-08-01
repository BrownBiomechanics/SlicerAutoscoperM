import math
from typing import Optional

import slicer
import vtk


class Camera:
    def __init__(self) -> None:
        self.DID = 0
        self.vtkCamera = vtk.vtkCamera()
        self.imageSize = [512, 512]
        self.id = -1


def generateNCameras(N: int, bounds: list[int], offset: int = 100, imageSize: tuple[int] = (512, 512)) -> list[Camera]:
    """
    Generate N cameras

    :param N: Number of cameras to generate
    :type N: int

    :param bounds: Bounds of the volume
    :type bounds: list[int]

    :param offset: Offset from the volume. Defaults to 100.
    :type offset: int

    :param imageSize: Image size. Defaults to [512,512].
    :type imageSize: list[int]

    :return: List of cameras
    :rtype: list[Camera]
    """
    # find the center of the bounds
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]

    # find the largest dimension of the bounds
    largestDimension = max([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

    # find the distance from the center to the bounds
    r = largestDimension / 2 + offset

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(N)
    points.SetDataTypeToDouble()
    points.Allocate(N)

    # use the spherical fibonacci algorithm to generate the points
    goldenRatio = (1 + math.sqrt(5)) / 2
    i = range(0, N)
    theta = [2 * math.pi * i / goldenRatio for i in i]
    phi = [math.acos(1 - 2 * (i + 0.5) / N) for i in i]
    x = [math.cos(theta[i]) * math.sin(phi[i]) for i in i]
    y = [math.sin(theta[i]) * math.sin(phi[i]) for i in i]
    z = [math.cos(phi[i]) for i in i]
    # scale the points to the radius
    x = [r * x[i] for i in i]
    y = [r * y[i] for i in i]
    z = [r * z[i] for i in i]
    for px, py, pz in zip(x, y, z):
        points.InsertNextPoint(px + center[0], py + center[1], pz + center[2])

    # create the cameras
    cameras = []
    for i in range(N):
        camera = Camera()
        camera.vtkCamera.SetPosition(points.GetPoint(i))
        camera.vtkCamera.SetFocalPoint(center)
        camera.vtkCamera.SetViewAngle(25)
        camera.vtkCamera.SetClippingRange(0.1, 1000)
        camera.id = i
        camera.imageSize = imageSize
        cameras.append(camera)

    return cameras


def generateVRG(
    camera: Camera,
    volumeNode: slicer.vtkMRMLVolumeNode,
    outputFileName: str,
    width: int,
    height: int,
) -> None:
    """
    Generate a virtual radiograph from the given camera and volume node

    :param camera: Camera
    :type camera: Camera

    :param volumeNode: Volume node
    :type volumeNode: slicer.vtkMRMLVolumeNode

    :param outputFileName: Output file name
    :type outputFileName: str

    :param width: Width of the output image.
    :type width: int

    :param height: Height of the output image.
    :type height: int
    """

    # create the renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.SetUseDepthPeeling(1)
    renderer.SetMaximumNumberOfPeels(100)
    renderer.SetOcclusionRatio(0.1)

    # create the render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.SetSize(width, height)
    renderWindow.AddRenderer(renderer)

    # create the volume mapper
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(volumeNode.GetImageData())
    volumeMapper.SetBlendModeToComposite()

    # create the volume property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.2)

    # create a piecewise function for scalar opacity
    # first point is X: 300, O: 0.00
    # second point is X: 950, O: 0.20
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(-10000, 0.05)
    opacityTransferFunction.AddPoint(0, 0.00)
    opacityTransferFunction.AddPoint(400, 0.05)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)

    # create the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # add the volume to the renderer
    renderer.AddVolume(volume)
    renderer.SetActiveCamera(camera.vtkCamera)
    renderer.ResetCamera()

    # render the image
    renderWindow.Render()

    # save the image
    writer = vtk.vtkTIFFWriter()
    writer.SetFileName(outputFileName)

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.SetScale(1)
    windowToImageFilter.SetInputBufferTypeToRGB()

    # convert the imag to grayscale
    luminance = vtk.vtkImageLuminance()
    luminance.SetInputConnection(windowToImageFilter.GetOutputPort())

    writer.SetInputConnection(luminance.GetOutputPort())
    writer.Write()


def _calculateDataIntensityDensity(camera: Camera, whiteRadiographFName: str) -> None:
    """
    Calculates the data intensity density of the given camera on its corresponding white radiograph.
    Internal function used by :func:`optimizeCameras`.

    :param camera: Camera
    :type camera: Camera

    :param whiteRadiographFName: White radiograph file name
    :type whiteRadiographFName: str
    """
    import numpy as np
    import SimpleITK as sitk

    MEAN_COMPARISON = 170  # 255 / 3 * 2

    # Read in the white radiograph
    whiteRadiograph = sitk.ReadImage(whiteRadiographFName)

    # Superpixel Segmentation
    slicImageFilter = sitk.SLICImageFilter()
    slicImageFilter.SetSuperGridSize([15, 15, 15])  # smaller grid size = finer grid overall default is [50,50,50]
    labelImage = slicImageFilter.Execute(whiteRadiograph)

    # Get the mean pixel value for each label
    labelStatsFilter = sitk.LabelStatisticsImageFilter()
    labelStatsFilter.Execute(whiteRadiograph, labelImage)
    N = labelStatsFilter.GetNumberOfLabels()
    meanColor = np.zeros((N, 1))
    m, n = labelImage.GetSize()
    labels = list(labelStatsFilter.GetLabels())
    labels.sort()
    for i, label in enumerate(labels):
        meanColor[i, 0] = labelStatsFilter.GetMean(label)

    # Create a binary label from the labelImage where all '1' are labels whose meanColor are < 255/3
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(labelImage)
    binaryLabels = np.zeros((m, n))
    for i, label in enumerate(labels):
        if label == 0:
            continue
        if meanColor[i, 0] < MEAN_COMPARISON:
            pixels = list(labelShapeFilter.GetIndexes(label))
            for j in range(0, len(pixels), 2):
                y = pixels[j]
                x = pixels[j + 1]
                binaryLabels[x, y] = 1

    # Calculate the Data Intensity Density
    # Largest Region based off of https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958/2
    binaryImage = sitk.Cast(sitk.GetImageFromArray(binaryLabels), sitk.sitkUInt8)
    componentImage = sitk.ConnectedComponent(binaryImage)
    sortedComponentImage = sitk.RelabelComponent(componentImage, sortByObjectSize=True)
    largest = sortedComponentImage == 1

    camera.DID = np.sum(sitk.GetArrayFromImage(largest))


def optimizeCameras(
    cameras: list[Camera],
    cameraDir: str,
    nOptimizedCameras: int,
    progressCallback: Optional[callable] = None,
) -> list[Camera]:
    """
    Optimize the cameras by finding the N cameras with the best data intensity density.

    :param cameras: Cameras
    :type cameras: list[Camera]

    :param cameraDir: Camera directory
    :type cameraDir: str

    :param nOptimizedCameras: Number of optimized cameras to find
    :type nOptimizedCameras: int

    :return: Optimized cameras
    :rtype: list[Camera]
    """
    import glob
    import os

    if not progressCallback:

        def progressCallback(_x):
            return None

    for i in range(len(cameras)):
        camera = cameras[i]
        vrgFName = glob.glob(os.path.join(cameraDir, f"cam{camera.id}", "*.tif"))[0]
        _calculateDataIntensityDensity(camera, vrgFName)
        progress = ((i + 1) / len(cameras)) * 29 + 42
        progressCallback(progress)

    cameras.sort(key=lambda x: x.DID, reverse=True)

    return cameras[:nOptimizedCameras]
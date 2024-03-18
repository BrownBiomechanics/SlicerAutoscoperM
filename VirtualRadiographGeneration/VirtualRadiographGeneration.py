#!/usr/bin/env python-real

import concurrent.futures as cf
import glob
import json
import os
import sys

import vtk


def generateVRG(
    camera: vtk.vtkCamera,
    volumeImageData: vtk.vtkImageData,
    outputFileName: str,
    width: int,
    height: int,
) -> None:
    """
    Generate a virtual radiograph from the given camera and volume node

    :param camera: Camera
    :param volumeImageData: Volume image data
    :param outputFileName: Output file name
    :param width: Width of the output image
    :param height: Height of the output image
    """

    # find the min and max scalar values
    hist = vtk.vtkImageHistogramStatistics()
    hist.SetInputData(volumeImageData)
    hist.Update()
    minVal = hist.GetMinimum()
    maxVal = hist.GetMaximum()

    # create the renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # Set background to white
    renderer.SetUseDepthPeeling(True)

    # create the render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.SetSize(width, height)
    renderWindow.AddRenderer(renderer)

    # create the volume mapper
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(volumeImageData)
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetBlendModeToComposite()
    volumeMapper.SetUseJittering(False)

    # Set the transfer functions for opacity, gradient and color
    opacityTransferFunction = vtk.vtkPiecewiseFunction()  # From the Slicer CT XRay preset
    opacityTransferFunction.AddPoint(minVal, 0.0)
    opacityTransferFunction.AddPoint(1500, 0.05)
    opacityTransferFunction.AddPoint(maxVal, 0.05)

    gradTransferFunction = vtk.vtkPiecewiseFunction()  # From the Slicer CT XRay preset
    gradTransferFunction.AddPoint(0, 1)
    gradTransferFunction.AddPoint(255, 1)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(maxVal, 1, 1, 1)
    colorTransferFunction.AddRGBPoint(minVal, 0, 0, 0)

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOff()
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.SetGradientOpacity(gradTransferFunction)
    volumeProperty.SetColor(colorTransferFunction)

    # create the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # add the volume to the renderer
    renderer.AddVolume(volume)
    renderer.SetActiveCamera(camera)

    # render the image
    renderWindow.Render()

    # save the image
    writer = vtk.vtkTIFFWriter()
    writer.SetFileName(outputFileName)

    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.SetScale(1)
    windowToImageFilter.SetInputBufferTypeToRGB()

    # convert the image to grayscale
    luminance = vtk.vtkImageLuminance()
    luminance.SetInputConnection(windowToImageFilter.GetOutputPort())

    writer.SetInputConnection(luminance.GetOutputPort())
    writer.Write()


def _createVTKCameras(cameraDir: str, radiographMainDir: str):
    """
    Generates a vtkCamera object from the given parameters
    """
    cameras = {}
    for camFName in glob.glob(os.path.join(cameraDir, "*.json")):
        with open(camFName) as f:
            camJSON = json.load(f)

        cam = vtk.vtkCamera()
        cam.SetPosition(camJSON["camera-position"])
        cam.SetFocalPoint(camJSON["focal-point"])
        cam.SetViewUp(camJSON["view-up"])
        cam.SetViewAngle(camJSON["view-angle"])
        cam.SetClippingRange(camJSON["clipping-range"])

        cameraSubDirName = os.path.basename(camFName).split(".")[0]
        cameraDirName = os.path.join(radiographMainDir, cameraSubDirName)
        if not os.path.exists(cameraDirName):
            os.mkdir(cameraDirName)

        cameras[cam] = cameraDirName

    return cameras


if __name__ == "__main__":
    expected_args = [
        "inputVolumeFileName",
        "cameraDir",
        "radiographMainOutDir",
        "outputFileName",
        "outputWidth",
        "outputHeight",
    ]
    expected_args = [f"<{arg}>" for arg in expected_args]
    if len(sys.argv[1:]) != len(expected_args):
        print(f"Usage: {sys.argv[0]} {' '.join(expected_args)}")
        sys.exit(1)

    volumeData = sys.argv[1]
    cameraDir = sys.argv[2]
    radiographMainOutDir = sys.argv[3]
    outputFileName = sys.argv[4]
    outputWidth = int(sys.argv[5])
    outputHeight = int(sys.argv[6])

    # create the camera
    cameras = _createVTKCameras(cameraDir, radiographMainOutDir)

    # Read the mhd file
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(volumeData)
    reader.Update()

    # generate the virtual radiograph
    with cf.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                generateVRG, cam, reader.GetOutput(), os.path.join(camDir, outputFileName), outputWidth, outputHeight
            )
            for cam, camDir in cameras.items()
        ]
        for future in cf.as_completed(futures):
            future.result()

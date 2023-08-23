#!/usr/bin/env python-real

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
    opacityTransferFunction.AddPoint(0, 0.0)
    opacityTransferFunction.AddPoint(1500, 0.05)
    opacityTransferFunction.AddPoint(3071, 0.05)

    gradTransferFunction = vtk.vtkPiecewiseFunction()  # From the Slicer CT XRay preset
    gradTransferFunction.AddPoint(0, 1)
    gradTransferFunction.AddPoint(255, 1)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0, 1, 1, 1)  # Low to be white
    colorTransferFunction.AddRGBPoint(3071, 0, 0, 0)  # High to be black

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


def _createVTKCamera(
    position: list[float], focalPoint: list[float], viewUp: list[float], clippingRange: list[float], viewAngle: float
) -> vtk.vtkCamera:
    """
    Generates a vtkCamera object from the given parameters
    """
    camera = vtk.vtkCamera()
    camera.SetPosition(position[0], position[1], position[2])
    camera.SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2])
    camera.SetViewUp(viewUp[0], viewUp[1], viewUp[2])
    camera.SetViewAngle(viewAngle)
    camera.SetClippingRange(clippingRange[0], clippingRange[1])
    return camera


def _strToFloatList(strList: str) -> list[float]:
    """
    Converts a string of floats to a list of floats
    """
    return [float(x) for x in strList.split(",")]


if __name__ == "__main__":
    expected_args = [
        "inputVolumeFileName",
        "cameraPosition",
        "cameraFocalPoint",
        "cameraViewUp",
        "cameraViewAngle",
        "clippingRange",
        "outputFileName",
        "outputWidth",
        "outputHeight",
    ]
    expected_args = [f"<{arg}>" for arg in expected_args]
    if len(sys.argv[1:]) != len(expected_args):
        print(f"Usage: {sys.argv[0]} {' '.join(expected_args)}")
        sys.exit(1)

    volumeData = sys.argv[1]
    cameraPosition = _strToFloatList(sys.argv[2])
    cameraFocalPoint = _strToFloatList(sys.argv[3])
    cameraViewUp = _strToFloatList(sys.argv[4])
    cameraViewAngle = float(sys.argv[5])
    clippingRange = _strToFloatList(sys.argv[6])
    outputFileName = sys.argv[7]
    outputWidth = int(sys.argv[8])
    outputHeight = int(sys.argv[9])

    # create the camera
    camera = _createVTKCamera(cameraPosition, cameraFocalPoint, cameraViewUp, clippingRange, cameraViewAngle)

    # Read the mhd file
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(volumeData)
    reader.Update()

    # generate the virtual radiograph
    generateVRG(camera, reader.GetOutput(), outputFileName, outputWidth, outputHeight)

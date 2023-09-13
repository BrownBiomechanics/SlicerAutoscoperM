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
    :type camera: vtk.vtkCamera

    :param volumeImageData: Volume image data
    :type volumeImageData: vtk.vtkImageData

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

    # create the render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.SetSize(width, height)
    renderWindow.AddRenderer(renderer)

    # create the volume mapper
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(volumeImageData)
    volumeMapper.SetBlendModeToComposite()

    # create the volume property
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.2)

    # create a piecewise function for scalar opacity
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

    # convert the imag to grayscale
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
    EXPECTED_ARGS = 8
    if len(sys.argv) < EXPECTED_ARGS:
        usageString = """
        Usage:
        virtualRadiographGeneration
        <volumeData>
        <cameraPosition>
        <cameraFocalPoint>
        <cameraViewUp>
        <cameraViewAngle>
        <clippingRange>
        <outputFileName>
        <width>
        <height>
        """
        usageString = usageString.replace(" ", "")
        usageString = usageString.replace("\n", " ")
        print(usageString)

        sys.exit(1)
    volumeData = sys.argv[1]
    cameraPosition = _strToFloatList(sys.argv[2])
    cameraFocalPoint = _strToFloatList(sys.argv[3])
    cameraViewUp = _strToFloatList(sys.argv[4])
    cameraViewAngle = float(sys.argv[5])
    clippingRange = _strToFloatList(sys.argv[6])
    outputFileName = sys.argv[7]
    width = int(sys.argv[8])
    height = int(sys.argv[9])

    # create the camera
    camera = _createVTKCamera(cameraPosition, cameraFocalPoint, cameraViewUp, clippingRange, cameraViewAngle)

    # Read the mhd file
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(volumeData)
    reader.Update()

    # generate the virtual radiograph
    generateVRG(camera, reader.GetOutput(), outputFileName, width, height)

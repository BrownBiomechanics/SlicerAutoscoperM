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
        self.FrustumModel = None

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Camera {self.id}",
                f"Position: {self.vtkCamera.GetPosition()}",
                f"Focal Point: {self.vtkCamera.GetFocalPoint()}",
                f"View Angle: {self.vtkCamera.GetViewAngle()}",
                f"Clipping Range: {self.vtkCamera.GetClippingRange()}",
                f"View Up: {self.vtkCamera.GetViewUp()}",
                f"Direction of Projection: {self.vtkCamera.GetDirectionOfProjection()}",
                f"Distance: {self.vtkCamera.GetDistance()}",
                f"Image Size: {self.imageSize}",
                f"DID: {self.DID}",
                "~" * 20,
            ]
        )


def _createFrustumModel(cam: Camera) -> None:
    model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    model.CreateDefaultDisplayNodes()
    model.GetDisplayNode().SetColor(1, 0, 0)  # Red
    model.GetDisplayNode().SetOpacity(0.3)
    model.GetDisplayNode().SetVisibility(True)

    model.SetName(f"cam{cam.id}-frustum")

    cam.FrustumModel = model

    _updateFrustumModel(cam)


def _updateFrustumModel(cam: Camera) -> None:
    if cam.FrustumModel is None:
        _createFrustumModel(cam)
        return
    # The equations of the six planes of the frustum in the order: left, right, bottom, top, far, near
    # Given as A, B, C, D where Ax + By + Cz + D = 0 for each plane
    planesArray = [0] * 24
    aspectRatio = cam.vtkCamera.GetExplicitAspectRatio()

    cam.vtkCamera.GetFrustumPlanes(aspectRatio, planesArray)

    planes = vtk.vtkPlanes()
    planes.SetFrustumPlanes(planesArray)

    hull = vtk.vtkHull()
    hull.SetPlanes(planes)
    pd = vtk.vtkPolyData()
    hull.GenerateHull(pd, [-1000, 1000, -1000, 1000, -1000, 1000])

    cam.FrustumModel.SetAndObservePolyData(pd)


def generateNCameras(
    N: int, bounds: list[int], offset: int = 100, imageSize: tuple[int] = (512, 512), camDebugMode: bool = False
) -> list[Camera]:
    """
    Generate N cameras

    :param N: Number of cameras to generate
    :param bounds: Bounds of the volume
    :param offset: Offset from the volume. Defaults to 100.
    :param imageSize: Image size. Defaults to [512,512].
    :param camDebugMode: Whether or not to show the cameras in the scene. Defaults to False.

    :return: List of cameras
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
        camera.vtkCamera.SetViewAngle(30)
        camera.vtkCamera.SetClippingRange(0.1, r + largestDimension)
        camera.vtkCamera.SetViewUp(0, 1, 0)
        camera.id = i
        camera.imageSize = imageSize
        cameras.append(camera)

    if camDebugMode:
        # Visualize the cameras
        markups = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        for cam in cameras:
            print(cam)
            # add a point to the markups node
            markups.AddControlPoint(
                cam.vtkCamera.GetPosition()[0],
                cam.vtkCamera.GetPosition()[1],
                cam.vtkCamera.GetPosition()[2],
                f"cam{cam.id}",
            )
            # lock the point
            markups.SetNthControlPointLocked(markups.GetNumberOfControlPoints() - 1, True)

            _createFrustumModel(cam)

    return cameras


def generateCamerasFromMarkups(
    fiduaicalNode: slicer.vtkMRMLMarkupsFiducialNode,
    volumeBounds: list[int],
    clippingRange: tuple[int],
    viewAngle: int,
    imageSize: tuple[int] = (512, 512),
    cameraDebug: bool = False,
) -> list[Camera]:
    """
    Generate cameras from a markups fiducial node

    :param fiduaicalNode: Markups fiducial node
    :param volumeBounds: Bounds of the volume
    :param clippingRange: Clipping range
    :param viewAngle: View angle
    :param imageSize: Image size. Defaults to [512,512].
    :param cameraDebug: Whether or not to show the cameras in the scene. Defaults to False.

    :return: List of cameras
    """
    center = [
        (volumeBounds[0] + volumeBounds[1]) / 2,
        (volumeBounds[2] + volumeBounds[3]) / 2,
        (volumeBounds[4] + volumeBounds[5]) / 2,
    ]
    n = fiduaicalNode.GetNumberOfControlPoints()
    cameras = []
    for i in range(n):
        camera = Camera()
        camera.vtkCamera.SetPosition(fiduaicalNode.GetNthControlPointPosition(i))
        camera.vtkCamera.SetFocalPoint(center)
        camera.vtkCamera.SetViewAngle(viewAngle)
        camera.vtkCamera.SetClippingRange(clippingRange[0], clippingRange[1])
        camera.vtkCamera.SetViewUp(0, 1, 0)
        camera.id = fiduaicalNode.GetNthControlPointLabel(i)
        camera.imageSize = imageSize
        if cameraDebug:
            _createFrustumModel(camera)
        cameras.append(camera)
    return cameras


def optimizeCameras(
    cameras: list[Camera],
    cameraDir: str,
    nOptimizedCameras: int,
    progressCallback: Optional[callable] = None,
) -> list[Camera]:
    """
    Optimize the cameras by finding the N cameras with the best data intensity density.

    :param cameras: Cameras
    :param cameraDir: Camera directory
    :param nOptimizedCameras: Number of optimized cameras to find

    :return: Optimized cameras
    """
    import glob
    import os

    if not progressCallback:

        def progressCallback(_x):
            return None

    # Parallel calls to cliModule
    cliModule = slicer.modules.calculatedataintensitydensity
    cliNodes = []
    for i in range(len(cameras)):
        camera = cameras[i]
        vrgFName = glob.glob(os.path.join(cameraDir, f"cam{camera.id}", "*.tif"))[0]
        cliNode = slicer.cli.run(
            cliModule,
            None,
            {"whiteRadiographFileName": vrgFName},
            wait_for_completion=False,
        )
        cliNodes.append(cliNode)

    for i in range(len(cameras)):
        while cliNodes[i].GetStatusString() != "Completed":
            slicer.app.processEvents()
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
        cameras[i].DID = float(cliNodes[i].GetOutputText())  # cliNodes[i].GetParameterAsString("dataIntensityDensity")
        progress = ((i + 1) / len(cameras)) * 50 + 40
        progressCallback(progress)
        slicer.mrmlScene.RemoveNode(cliNodes[i])

    cameras.sort(key=lambda x: x.DID)

    return cameras[:nOptimizedCameras]

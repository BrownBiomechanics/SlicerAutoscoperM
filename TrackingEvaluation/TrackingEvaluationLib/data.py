import glob
import os

import numpy as np
import slicer
import vtk


def loadTraAsSequence(data: np.ndarray) -> list[vtk.vtkMatrix4x4]:
    """
    Converts the tracking data to a list of vtkMatrix4x4.

    :param data: The tracking data.

    :return: The tracking data as a sequence.
    """
    _, cols = data.shape

    EXPECTED_DIMENSION = 16
    if cols != EXPECTED_DIMENSION:
        # Check to see if the data was exported as a 4x4 matrix, probably want to expand this method
        # to support other formats.
        slicer.util.errorDisplay("Loading as sequence currently only supports 4x4 matrices")
        return None

    result = []
    for idx, row in enumerate(data):
        matrix = vtk.vtkMatrix4x4()
        # If there is no data, set the matrix to the previous matrix.
        # If its the first matrix, set it to the identity matrix.
        if np.isnan(row).any():
            if idx == 0:
                matrix.Identity()
            else:
                matrix.DeepCopy(result[idx - 1])
        else:
            for i in range(4):
                for j in range(4):
                    matrix.SetElement(i, j, row[i * 4 + j])
        result.append(matrix)
    return result


def tmpTFMLoader(
    fileName: str,
) -> vtk.vtkMatrix4x4:  # Going to replace this with a dedicated module for loading tfm files.
    """
    Loads a tfm file and returns the matrix.

    :param fileName: The filename of the tfm file.

    :return: The matrix from the tfm file.
    """
    with open(fileName) as f:
        lines = f.readlines()
        matrix = vtk.vtkMatrix4x4()
        matrixLines = lines[:4]  # first 4 lines are the matrix
        for i, line in enumerate(matrixLines):
            values = line.strip()
            values = values.split(" ")
            values = [float(x) for x in values]
            for j in range(4):
                matrix.SetElement(i, j, values[j])
    return matrix


class ModelData:
    """
    Class to store the model data.

    :param modelFileName: The filename of the model.
    :param groundTruthSequenceData: The ground truth sequence data.
    """

    def __init__(self, modelFileName: str, groundTruthSequenceData: np.ndarray):
        self.userModelNode = None
        self.userTransformNode = None
        self.userDisplayNode = None
        self.userSequence = None

        self.groundTruthModelNode = None
        self.groundTruthTransformNode = None
        self.groundTruthDisplayNode = None
        self.groundTruthSequence = None

        self._loadModel(modelFileName)
        self.setColor()  # Set the initial color to white(default value)
        self.groundTruthSequence = loadTraAsSequence(groundTruthSequenceData)

    def _loadModel(self, modelFileName: str):
        """
        Internal function to load the model from the filename.
        """
        if not os.path.exists(modelFileName):
            slicer.util.errorDisplay(f"File not found: {modelFileName}")
            return

        # Load the user model
        # self.userModelNode = slicer.util.loadModel(modelFileName)
        self.userModelNode = slicer.util.loadNodeFromFile(modelFileName, "ModelFile", {"coordinateSystem": "RAS"})
        self.userDisplayNode = self.userModelNode.GetDisplayNode()

        # Load a second model for the groud truth data
        # self.groundTruthModelNode = slicer.util.loadModel(modelFileName)
        self.groundTruthModelNode = slicer.util.loadNodeFromFile(
            modelFileName, "ModelFile", {"coordinateSystem": "RAS"}
        )
        self.groundTruthModelNode.SetName(self.userModelNode.GetName() + "_GroundTruth")
        self.groundTruthDisplayNode = self.groundTruthModelNode.GetDisplayNode()
        self.groundTruthDisplayNode.SetOpacity(0.5)
        self.groundTruthDisplayNode.SetVisibility(False)
        return

    def initializeTransforms(self):
        self.userTransformNode = slicer.vtkMRMLTransformNode()
        slicer.mrmlScene.AddNode(self.userTransformNode)
        self.userModelNode.SetAndObserveTransformNodeID(self.userTransformNode.GetID())

        self.groundTruthTransformNode = slicer.vtkMRMLTransformNode()
        slicer.mrmlScene.AddNode(self.groundTruthTransformNode)
        self.groundTruthModelNode.SetAndObserveTransformNodeID(self.groundTruthTransformNode.GetID())

    def cleanup(self):
        """
        Clean up the model data.
        """
        slicer.mrmlScene.RemoveNode(self.userModelNode)
        slicer.mrmlScene.RemoveNode(self.userTransformNode)
        slicer.mrmlScene.RemoveNode(self.groundTruthModelNode)
        slicer.mrmlScene.RemoveNode(self.groundTruthTransformNode)

    def loadUserTrackingSequence(self, userSequence: list[vtk.vtkMatrix4x4]):
        """
        Load the user tracking sequence.

        :param userSequence: The user tracking sequence.
        """
        self.userSequence = userSequence

    def updateTransform(self, index: int):
        """
        Update the transform of the model node.

        :param index: The index of the transform.
        """
        if self.userSequence is None:
            slicer.util.errorDisplay("No user tracking data!")
            return
        if index >= len(self.userSequence):
            slicer.util.errorDisplay("Index out of bounds!")
            return
        self.userTransformNode.SetMatrixTransformToParent(self.userSequence[index])
        self.groundTruthTransformNode.SetMatrixTransformToParent(self.groundTruthSequence[index])

    def evaluateError(self, index: int, translationTol: float = 1.0, degreeTol: float = 2.0) -> bool:
        """
        Evaluates the current position of the model node and compares it to the ground truth.

        If the model is within a certain threshold of the ground truth, the model is considered to be
        in the correct position.

        :param index: The index of the transform.
        :param translationTol: The translation tolerance in mm.
        :param degreeTol: The rotation tolerance in degrees.
        """
        if self.userSequence is None:
            slicer.util.errorDisplay("No user tracking data!")
            return False

        if index >= len(self.userSequence):
            slicer.util.errorDisplay("Index out of bounds!")
            return False

        groundTruthMatrix = self.groundTruthSequence[index]
        userMatrix = self.userSequence[index]

        groundTruthTranslation = np.array([groundTruthMatrix.GetElement(i, 3) for i in range(3)])
        userTranslation = np.array([userMatrix.GetElement(i, 3) for i in range(3)])

        groundRotation = np.asarray(
            [
                180 * np.arctan2(groundTruthMatrix.GetElement(1, 0), groundTruthMatrix.GetElement(0, 0)) / np.pi,
                180
                * np.arctan2(
                    -groundTruthMatrix.GetElement(2, 0),
                    np.sqrt(groundTruthMatrix.GetElement(2, 1) ** 2 + groundTruthMatrix.GetElement(2, 2) ** 2),
                )
                / np.pi,
                180 * np.arctan2(groundTruthMatrix.GetElement(2, 1), groundTruthMatrix.GetElement(2, 2)) / np.pi,
            ]
        )
        userRotation = np.asarray(
            [
                180 * np.arctan2(userMatrix.GetElement(1, 0), userMatrix.GetElement(0, 0)) / np.pi,
                180
                * np.arctan2(
                    -userMatrix.GetElement(2, 0),
                    np.sqrt(userMatrix.GetElement(2, 1) ** 2 + userMatrix.GetElement(2, 2) ** 2),
                )
                / np.pi,
                180 * np.arctan2(userMatrix.GetElement(2, 1), userMatrix.GetElement(2, 2)) / np.pi,
            ]
        )

        translationDiff = np.linalg.norm(groundTruthTranslation - userTranslation)
        rotationDiff = np.linalg.norm(groundRotation - userRotation)

        if False:  # Debug info
            print(f"groundTruthTranslation: {groundTruthTranslation}")
            print(f"userTranslation: {userTranslation}")
            print(f"groundRotation: {groundRotation}")
            print(f"userRotation: {userRotation}")
            print(f"translationDiff: {translationDiff}")
            print(f"rotationDiff: {rotationDiff}")
            print(f"translationTol: {translationTol}")
            print(f"degreeTol: {degreeTol}")
            print(f"translationDiff > translationTol: {translationDiff > translationTol}")
            print(f"rotationDiff > degreeTol: {rotationDiff > degreeTol}")
            print("groundTruthMatrix:")
            for i in range(4):
                print([groundTruthMatrix.GetElement(i, j) for j in range(4)])
            print("userMatrix:")
            for i in range(4):
                print([userMatrix.GetElement(i, j) for j in range(4)])

        return translationDiff <= translationTol and rotationDiff <= degreeTol

    def setColor(self, rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Update the color of the model node.

        :param rgb: The RGB color to set.
        """
        self.userDisplayNode.SetColor(rgb[0], rgb[1], rgb[2])

    def setGroundTruthVisible(self, visible: bool):
        """
        Hide or show the ground truth model node.

        :param visible: whether the ground truth model node is visible or not.
        """
        self.groundTruthDisplayNode.SetVisibility(visible)


class Scene:
    """
    Class to store the scene data.

    :param sampleDataType: The sample data type.
    :param userSequenceFileName: The filename of the user sequence.
    """

    def __init__(self, sampleDataType: str, userSequenceFileName: str):
        self.sampleDataDir = os.path.join(slicer.mrmlScene.GetCacheManager().GetRemoteCacheDirectory(), sampleDataType)
        self.models = self._loadModels()
        self.inactiveModels = self._parseUserSequence(userSequenceFileName)
        self.currentFrame = 0
        self.maxFrame = len(self.models[0].groundTruthSequence)
        self._validateTransforms()
        self.updateTransforms(0)
        self.modelNames = [model.userModelNode.GetName() for model in self.models]

    def _loadModels(self) -> list[ModelData]:
        """
        Internal function to load the models.
        """
        if not os.path.exists(self.sampleDataDir):
            slicer.util.errorDisplay(
                "Sample data directory not found: {}\n Please download sample data from the Sample Data module.".format(
                    self.sampleDataDir
                )
            )
            return None

        modelFileNames = glob.glob(os.path.join(self.sampleDataDir, "Meshes", "*.stl"))
        groundTruthSequenceFileNames = glob.glob(os.path.join(self.sampleDataDir, "Tracking", "*.tra"))

        if len(modelFileNames) != len(groundTruthSequenceFileNames):
            slicer.util.errorDisplay("Number of models and ground truth sequences do not match!")
            return None

        models = []
        for i in range(len(modelFileNames)):
            models.append(ModelData(modelFileNames[i], np.loadtxt(groundTruthSequenceFileNames[i], delimiter=",")))
            models[i].initializeTransforms()
        return models

    def cleanup(self):
        """
        Cleanup the scene.
        """
        for model in self.models:
            model.cleanup()

    def _parseUserSequence(self, userSequenceFileName: str) -> list[int]:
        """
        Internal function to parse the user sequence.

        :param userSequenceFileName: The filename of the user sequence.

        :return: List of indices that correspond to any models with Nan tracking values.
        """

        if not os.path.exists(userSequenceFileName):
            slicer.util.errorDisplay(f"File not found: {userSequenceFileName}")
            return []

        data = np.loadtxt(userSequenceFileName, delimiter=",")

        _, cols = data.shape

        # Check to see if the data was exported as a 4x4 matrix, probably want to expand this
        # method to support other formats.
        EXPECTED_DIMENSION = 16
        if cols % EXPECTED_DIMENSION != 0:
            slicer.util.errorDisplay(
                "Loading as sequence currently only supports 4x4 matrices", detailedText=f"{cols} columns"
            )
            return []

        # Check to see if the number of models and user sequences match.
        if cols / EXPECTED_DIMENSION != len(self.models):
            slicer.util.errorDisplay(
                "Number of models and user sequences do not match",
                detailedText=f"{len(self.models)} models and {cols / EXPECTED_DIMENSION} user sequences",
            )
            return []

        inactiveModels = []
        for modelIdx, model in enumerate(self.models):
            model.loadUserTrackingSequence(
                loadTraAsSequence(data[:, modelIdx * EXPECTED_DIMENSION : (modelIdx + 1) * EXPECTED_DIMENSION])
            )
        return inactiveModels

    def updateTransforms(self, index: int, translationTol: float = 1.0, degreeTol: float = 2.0):
        """
        Update the transforms of the models and evaluate the error compared to the ground truth.

        :param index: The index of the transform.
        :param translationTol: The translation tolerance.
        :param degreeTol: The degree tolerance.
        """
        self.currentFrame = index
        for i, model in enumerate(self.models):  # Probably want to make this multithreaded instead of sequential.
            if i in self.inactiveModels:
                continue
            model.updateTransform(index)
            withinTol = model.evaluateError(index, translationTol, degreeTol)
            if withinTol:
                model.setColor((0, 1, 0))  # Green
                model.setGroundTruthVisible(False)
            else:
                model.setColor((1, 0, 0))  # Red
                model.setGroundTruthVisible(True)
            slicer.app.processEvents()

    def _validateTransforms(self):
        """
        Validate the transforms of the models.

        Validates that the ground truth and user transforms are the same length.
        """
        for i, model in enumerate(self.models):
            if len(model.groundTruthSequence) != len(model.userSequence):
                # Make sure the user is not longer than the ground truth sequence.
                if len(model.groundTruthSequence) < len(model.userSequence):
                    slicer.util.errorDisplay(f"User sequence is longer than the ground truth sequence for model {i}")
                    return

                lastTransform = model.userSequence[-1]
                for _j in range(len(model.groundTruthSequence) - len(model.userSequence)):
                    matrix = vtk.vtkMatrix4x4()
                    matrix.DeepCopy(lastTransform)
                    model.userSequence.append(matrix)

    def calculateRelativeMovements(self, referenceNode):
        referenceIdx = -1

        for modelIdx, model in enumerate(self.models):
            if model.userModelNode == referenceNode:
                referenceIdx = modelIdx
                break

        if referenceIdx == -1:
            slicer.util.errorDisplay("Reference node not found! Are you sure it's a part of the scene?")
            return None

        results = []

        for modelIdx in range(len(self.models)):
            if modelIdx == referenceIdx:
                matrix = vtk.vtkMatrix4x4()
                matrix.Identity()
                results.append(matrix)
                continue

            userTFM = self.models[modelIdx].userSequence[self.currentFrame]
            referenceTFM = self.models[referenceIdx].userSequence[self.currentFrame]

            userR = vtk.vtkMatrix3x3()
            referenceR = vtk.vtkMatrix3x3()
            for i in range(3):
                for j in range(3):
                    userR.SetElement(i, j, userTFM.GetElement(i, j))
                    referenceR.SetElement(i, j, referenceTFM.GetElement(i, j))

            referenceR_inv = vtk.vtkMatrix3x3()
            vtk.vtkMatrix3x3().Invert(referenceR, referenceR_inv)

            relativeR = vtk.vtkMatrix3x3()
            vtk.vtkMatrix3x3().Multiply3x3(userR, referenceR_inv, relativeR)

            userT = np.array([userTFM.GetElement(i, 3) for i in range(3)])
            referenceT = np.array([referenceTFM.GetElement(i, 3) for i in range(3)])

            relativeT = np.array([0, 0, 0], dtype=np.float64)
            vtk.vtkMatrix3x3().MultiplyPoint(referenceR_inv.GetData(), userT - referenceT, relativeT)

            relativeTFM = vtk.vtkMatrix4x4()
            for i in range(3):
                for j in range(3):
                    relativeTFM.SetElement(i, j, relativeR.GetElement(i, j))
                relativeTFM.SetElement(i, 3, relativeT[i])

            results.append(relativeTFM)

        return results

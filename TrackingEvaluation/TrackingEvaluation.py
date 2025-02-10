from typing import Optional

import slicer
import vtk
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

#
# TrackingEvaluation
#


class TrackingEvaluation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Tracking Evaluation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Tracking",
        ]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Anthony Lombardi (Kitware)",
            "Amy M Morton (Brown University)",
            "Bardiya Akhbari (Brown University)",
            "Beatriz Paniagua (Kitware)",
            "Jean-Christophe Fillion-Robin (Kitware)",
        ]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information on <a href="https://autoscoper.readthedocs.io">Read the Docs</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""


#
# TrackingEvaluationParameterNode
#
@parameterNodeWrapper
class TrackingEvaluationParameterNode:
    """
    The parameters needed by module.
    """

    pass


#
# TrackingEvaluationWidget
#
class TrackingEvaluationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/TrackingEvaluation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TrackingEvaluationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.loadButton.connect("clicked(bool)", self.onLoadButton)
        self.ui.removeDataButton.connect("clicked(bool)", self.onRemoveDataButton)

        # Slider
        self.ui.currentFrameSlider.connect("valueChanged(int)", self.onCurrentFrameSlider)

        self.ui.MRMLNodeComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.initializeTable)

        # Make sure parameter node is initialized (needed for module reload)
        # self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()
        if self.logic.Scene:
            self.logic.Scene.cleanup()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        # self.initializeParameterNode()
        pass

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, _caller, _event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller, _event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[TrackingEvaluationParameterNode]):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def onLoadButton(self):
        """
        Gets the user tracking data and the active radio button and creates the corresponding scene.
        """
        from TrackingEvaluationLib.data import Scene

        wristRadio = self.ui.wristRadioButton.isChecked()
        kneeRadio = self.ui.kneeRadioButton.isChecked()
        ankleRadio = self.ui.ankleRadioButton.isChecked()
        sampleDataType = ""
        if wristRadio:
            sampleDataType = "2025-01-12-Wrist"
        elif kneeRadio:
            sampleDataType = "2025-02-10-Knee"
        elif ankleRadio:
            sampleDataType = "2025-01-12-Ankle"

        # Get the tracking data
        trackingData = self.ui.userTrackingSelector.currentPath

        # Create the scene
        self.logic.Scene = Scene(sampleDataType, trackingData)

        # Update the slider range
        self.ui.currentFrameSlider.maximum = self.logic.Scene.maxFrame - 1
        self.ui.currentFrameSpin.maximum = self.logic.Scene.maxFrame - 1

    def onRemoveDataButton(self):
        """
        Removes the data from the scene.
        """
        if self.logic.Scene:
            self.logic.inCleanUp = True
            self.logic.Scene.cleanup()
            self.logic.Scene = None
            self.ui.currentFrameSlider.maximum = 0
            self.ui.currentFrameSpin.maximum = 0
            self.ui.MRMLTableView.setMRMLTableNode(None)
            slicer.mrmlScene.RemoveNode(self.logic.tableNode)
            self.logic.tableNode = None
            self.logic.referenceNode = None
            self.logic.inCleanUp = False

    def onCurrentFrameSlider(self, value):
        if self.logic.Scene is None:
            return
        transTol = self.ui.tranTolBox.value
        rotTol = self.ui.rotTolBox.value
        self.logic.Scene.updateTransforms(value, transTol, rotTol)
        if self.logic.tableNode is not None:
            results = self.logic.Scene.calculateRelativeMovements(self.logic.referenceNode)
            self.logic.updateTable(results)

    def initializeTable(self, node):
        if self.logic.inCleanUp:
            return
        if node is None:
            return
        if self.logic.tableNode is not None:
            slicer.mrmlScene.RemoveNode(self.logic.tableNode)

        self.logic.referenceNode = node

        self.logic.tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        self.logic.tableNode.SetName("Relative Movements")

        namesArray = vtk.vtkStringArray()
        namesArray.SetName("Model Name")
        modelnames = self.logic.Scene.modelNames
        for i in range(len(modelnames)):
            namesArray.InsertNextValue(modelnames[i])
        self.logic.tableNode.AddColumn(namesArray)

        columnNames = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        for i in range(len(columnNames)):
            array = vtk.vtkDoubleArray()
            array.SetName(columnNames[i])
            self.logic.tableNode.AddColumn(array)

        self.ui.MRMLTableView.setMRMLTableNode(self.logic.tableNode)
        results = self.logic.Scene.calculateRelativeMovements(self.logic.referenceNode)
        self.logic.updateTable(results)


#
# TrackingEvaluationLogic
#
class TrackingEvaluationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self.Scene = None
        self.tableNode = None
        self.referenceNode = None
        self.inCleanUp = False

    def updateTable(self, results):
        """
        Updates the table with the results of the relative movements.
        """
        import math

        for i, matrix in enumerate(results):
            x = matrix.GetElement(0, 3)
            y = matrix.GetElement(1, 3)
            z = matrix.GetElement(2, 3)
            roll = math.atan2(matrix.GetElement(2, 1), matrix.GetElement(2, 2))
            roll = roll * 180 / math.pi
            pitch = math.atan2(
                -matrix.GetElement(2, 0),
                math.sqrt(matrix.GetElement(2, 1) ** 2 + matrix.GetElement(2, 2) ** 2),
            )
            pitch = pitch * 180 / math.pi
            yaw = math.atan2(matrix.GetElement(1, 0), matrix.GetElement(0, 0))
            yaw = yaw * 180 / math.pi
            self.tableNode.SetCellText(i, 1, str(x))
            self.tableNode.SetCellText(i, 2, str(y))
            self.tableNode.SetCellText(i, 3, str(z))
            self.tableNode.SetCellText(i, 4, str(roll))
            self.tableNode.SetCellText(i, 5, str(pitch))
            self.tableNode.SetCellText(i, 6, str(yaw))

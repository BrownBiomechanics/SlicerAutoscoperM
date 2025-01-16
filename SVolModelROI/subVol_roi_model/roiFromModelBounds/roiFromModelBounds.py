"""
For debugging in Slicer, use the following to manipulate the module objects:

mWidget = slicer.modules.roifrommodelbounds.widgetRepresentation().self()
mLogic = mWidget.logic
mNode = mLogic.getParameterNode()
"""


import glob
import os
import pathlib
from typing import Optional

import numpy as np
import slicer
import vtk
from slicer import vtkMRMLMarkupsROINode, vtkMRMLModelNode, vtkMRMLScalarVolumeNode
from slicer.i18n import tr as _
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

#
# roiFromModelBounds
#


class roiFromModelBounds(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("roiFromModelBounds")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = ["Tracking"]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Amy Morton (Brown University)"]

        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
WIP module part of SlicerAutoscoperM
Intended to act as foundation for subvolume 3d-3d volume registration
"""
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
Template for this file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""
        )

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal
    # connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # roiFromModelBounds1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="roiFromModelBounds",
        sampleName="roiFromModelBounds1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images"
        # set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "roiFromModelBounds1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="roiFromModelBounds1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="roiFromModelBounds1",
    )

    # roiFromModelBounds2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="roiFromModelBounds",
        sampleName="roiFromModelBounds2",
        thumbnailFileName=os.path.join(iconsPath, "roiFromModelBounds2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="roiFromModelBounds2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="roiFromModelBounds2",
    )


#
# roiFromModelBoundsParameterNode
#


@parameterNodeWrapper
class roiFromModelBoundsParameterNode:
    """
    The parameters needed by module.

    modelFile_path- load several stl inputModels from file
    inputVolume- Voume for ROI crop
    modelROI - The output volume that will contain the thresholded volume.
    croppedVolume - The output volume that will be .
    """

    modelFile_path: pathlib.Path
    inputVolume: vtkMRMLScalarVolumeNode

    # keep track of the modelss:
    modelIDs = vtk.vtkStringArray()

    # keep track of the rois:
    roiIDs = vtk.vtkStringArray()

    # keep track of the vols:
    volIDs = vtk.vtkStringArray()

    # keep track of the Tforms:
    tf_seed_IDs = vtk.vtkStringArray()
    tf_result_IDs = vtk.vtkStringArray()

    modelROI: vtkMRMLMarkupsROINode
    croppedVolume: vtkMRMLScalarVolumeNode

    # FUTURE DEVELOPMENT
    inputHierarchyRootID: int
    targetVolume: vtkMRMLScalarVolumeNode


#
# roiFromModelBoundsWidget
#


class roiFromModelBoundsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/roiFromModelBounds.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = roiFromModelBoundsLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.showBounds_pb.connect("clicked(bool)", self.onShowModelBoundsButton)
        self.ui.cropInputVol.connect("clicked(bool)", self.onCropVolumeButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Trigger any required UI updates based on the volume node selected by default
        self.onCurrentNodeChanged()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)

    def onSceneStartClose(self, _caller, _event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, _caller, _event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())
        """
        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.modelROI:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsROINode")
            if firstModelNode:
                self._parameterNode.modelROI = firstModelNode

        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
        """

    def setParameterNode(self, inputParameterNode: Optional[roiFromModelBoundsParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanGenModelROI)
            self._checkCanGenModelROI()

    def updateParameterNodeFromGUI(self, _caller=None, _event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # NA

        self._parameterNode.EndModify(wasModified)

    def onCurrentNodeChanged(self):
        """
        Updates and UI components that correspond to the selected input volume node
        """

    def _checkCanGenModelROI(self, _caller=None, _event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.modelFile_path:
            self.ui.showBounds_pb.toolTip = _("Compute output volume")
            self.ui.showBounds_pb.enabled = True
        else:
            self.ui.showBounds_pb.toolTip = _("Select input ModelPath and volume node")
            self.ui.showBounds_pb.enabled = True

    def onShowModelBoundsButton(self) -> None:
        """Load Models from folder, compute bounds, assign to new roi"""
        self.logic.loadModelsComputeBounds()

    def onCropVolumeButton(self) -> None:
        """Load Models from folder, compute bounds, assign to new roi"""
        self.logic.cropFromBounds()


#
# roiFromModelBoundsLogic
#


class roiFromModelBoundsLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return roiFromModelBoundsParameterNode(super().getParameterNode())

    def loadModelsComputeBounds(self):
        parameterNode = self.getParameterNode()
        # TO DO error checking on model path and volume node

        modelFileDir = parameterNode.modelFile_path

        modelFiles = glob.glob(os.path.join(modelFileDir, "*.*"))
        # return modelFiles

        for _indx, file in enumerate(modelFiles):
            modelNode = slicer.util.loadNodeFromFile(file)
            modelNode.CreateDefaultDisplayNodes()

            # roi:
            roi_this = self.modelBounds(modelNode)

            # store this roi ant this index- to ref in vol crop , tfroms
            parameterNode.roiIDs.InsertNextValue(roi_this.GetID())
            # model too (tform on model best visibility)
            parameterNode.modelIDs.InsertNextValue(modelNode.GetID())

    def modelBounds(self, inputModel):

        inputModel: vtkMRMLModelNode

        tB = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        inputModel.GetBounds(tB)
        # print(tB)
        # modelC = [0.0]*3
        # modelSize = [0.0]*3
        # strange that the model nodes have a getBounds- but no center and size..
        # and the roi has bno.. set Bounds- just center and size

        # numpy for array operatots:
        tnp_min = np.array([tB[0], tB[2], tB[4]])
        tnp_max = np.array([tB[1], tB[3], tB[5]])

        tnp_C = (tnp_min + tnp_max) / 2
        tnpS = tnp_min - tnp_max

        # inputModel.GetCenter(modelC)
        # inputModel.GetSize(modelSize)
        mname = inputModel.GetName()

        # use tB lims to size roi
        # Create ROI node
        modelROI = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        modelROI.SetCenter(tnp_C.tolist())
        modelROI.SetSize(tnpS.tolist())
        modelROI.CreateDefaultDisplayNodes()  # only needed for display

        # TODO hide the display

        roi_name = mname + "_roi"
        modelROI.SetName(roi_name)

        return modelROI

    def cropFromBounds(self):
        parameterNode = self.getParameterNode()

        tformIDs = parameterNode.tf_seed_IDs
        roiIDs = parameterNode.roiIDs
        numROIs = roiIDs.GetNumberOfValues()

        modelList = parameterNode.modelIDs

        for _indx in range(numROIs):
            roiID_this = roiIDs.GetValue(_indx)
            modelID_this = modelList.GetValue(_indx)
            modelNode_this = slicer.mrmlScene.GetNodeByID(modelID_this)

            parameterNode.modelROI = slicer.mrmlScene.GetNodeByID(roiID_this)

            # self rn is the logic class- not the widget.. no ui atteributes
            # self.ui.modelROISelector.setCurrentNode(roi_this)

            # crop vol
            cN = self.doCropVolume()

            # rename with model file name
            cropNewName = cN.GetName() + " " + modelNode_this.GetName()
            cN.SetName(cropNewName)

            tform = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode")
            tform.SetName("in_tform_" + cropNewName)

            modelNode_this.SetAndObserveTransformNodeID(tform.GetID())

            # Enable interactive transform
            # tform.SetInteractionMode(slicer.vtkMRMLTransformNode.InteractionModeTranslate)

            tformIDs.InsertNextValue(tform.GetID())

    def doCropVolume(self):

        parameterNode = self.getParameterNode()
        inVolume = parameterNode.inputVolume
        roi = parameterNode.modelROI

        fillValue = 0.0
        interpolate = False
        spacingScalingConst = 1.0
        isotropicResampling = False
        interpolationMode = slicer.vtkMRMLCropVolumeParametersNode().InterpolationLinear
        cropLogic = slicer.modules.cropvolume.logic()
        cvpn = slicer.vtkMRMLCropVolumeParametersNode()

        cvpn.SetROINodeID(roi.GetID())
        cvpn.SetInputVolumeNodeID(inVolume.GetID())
        cvpn.SetFillValue(fillValue)
        cvpn.SetVoxelBased(not interpolate)
        cvpn.SetSpacingScalingConst(spacingScalingConst)
        cvpn.SetIsotropicResampling(isotropicResampling)
        cvpn.SetInterpolationMode(interpolationMode)
        cropLogic.Apply(cvpn)
        roi.SetDisplayVisibility(False)

        outputVolumeNodeID = cvpn.GetOutputVolumeNodeID()
        # format- this is 'cropvol'  needs to be 'scalarVolume'
        outVolNode = slicer.mrmlScene.GetNodeByID(outputVolumeNodeID)

        # display pretty:
        # https://www.slicer.org/wiki/Documentation/4.3/Developers/Python_scripting
        views = slicer.app.layoutManager().sliceViewNames()
        for view in views:
            view_logic = slicer.app.layoutManager().sliceWidget(view).sliceLogic()
            view_cn = view_logic.GetSliceCompositeNode()
            view_cn.SetBackgroundVolumeID(outputVolumeNodeID)
            view_logic.FitSliceToAll()

        return outVolNode


#
# roiFromModelBoundsTest
#


class roiFromModelBoundsTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_roiFromModelBounds1()

    def test_roiFromModelBounds1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("TO DO configure test for this module ")

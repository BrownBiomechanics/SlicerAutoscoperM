import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin


#
# AutoscoperMGenerateTiffStacks
#

class AutoscoperMGenerateTiffStacks(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "AutoscoperMGenerateTiffStacks"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#AutoscoperMGenerateTiffStacks">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # AutoscoperMGenerateTiffStacks1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AutoscoperMGenerateTiffStacks',
        sampleName='AutoscoperMGenerateTiffStacks1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'AutoscoperMGenerateTiffStacks1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='AutoscoperMGenerateTiffStacks1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='AutoscoperMGenerateTiffStacks1'
    )

    # AutoscoperMGenerateTiffStacks2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='AutoscoperMGenerateTiffStacks',
        sampleName='AutoscoperMGenerateTiffStacks2',
        thumbnailFileName=os.path.join(iconsPath, 'AutoscoperMGenerateTiffStacks2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='AutoscoperMGenerateTiffStacks2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='AutoscoperMGenerateTiffStacks2'
    )


#
# AutoscoperMGenerateTiffStacksWidget
#

class AutoscoperMGenerateTiffStacksWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        self._updatingGUIFromParameterNode = False
        self.db = slicer.dicomDatabase

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/AutoscoperMGenerateTiffStacks.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AutoscoperMGenerateTiffStacksLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()


    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
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

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True


        # Update DICOM table and select the first patient, study, and series
        self.db = slicer.dicomDatabase
        self.ui.DICOMTableManager.setDICOMDatabase(self.db)


        self.ui.applyButton.enabled = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):

            patient = self.ui.DICOMTableManager.currentPatientsSelection()
            study = self.ui.DICOMTableManager.currentStudiesSelection()
            series = self.ui.DICOMTableManager.currentSeriesSelection()

            if len(patient) != 1 or len(study) != 1 or len(series) != 1:
                slicer.util.errorDisplay("Please select one patient, one study, and one series.")

            volumeNode = self.logic.getVolumeFromDICOM(self.db,series)
            
            slicer.util.setSliceViewerLayers(background=volumeNode)           

            segmentationNode = self.ui.segmentationSelector.currentNode()
            vrmlFile = self.ui.wrlSelector.currentPath
            if not segmentationNode and not vrmlFile:
                slicer.util.errorDisplay("Please select a segmentation node or a VRML file.")

            
            if vrmlFile:
                # read in VRML
                points, triangles, name = self.logic.readVRML(vrmlFile)
                # map points into volume space
                points = self.logic.transformPoints(points)
                # create segmentation node
                segmentationNode = self.logic.createSegmentationNode(points, triangles, name)

            outputDir = self.ui.PathLineEdit_2.currentPath
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
            # self.logic.generateTiffStacks(segmentationNode,volumeNode,outputDir,self.db,series)


#
# AutoscoperMGenerateTiffStacksLogic
#

class AutoscoperMGenerateTiffStacksLogic(ScriptedLoadableModuleLogic):
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

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        pass

    def readVRML(self,filename):
        """
        Read a WRL file
        """
        MIMICS_10_KEY = '#coordinates written in 1mm / 10000'
        MIMICS_13_KEY = '#coordinates written in 1mm / 0'
        MIMICS_22_KEY = '#resulted coordinates are measured in units, where 1 unit = 1000.000000 mm'
        with open(filename, 'r') as f:
            modelName = os.path.splitext(os.path.basename(filename))[0]
            lines = f.readlines()
            i = 0
            points = []
            coordIndex = []
            scale = [1,1,1]
            mFlag = False
            while i < len(lines):
                curLine = lines[i].strip()
                if curLine.startswith("point"):
                    i+=2 # skip the [ and the #coordinates written in 1mm / 10000
                    while not lines[i].strip().startswith("] #point"):
                        # check for mimics 10, 13, or 22
                        if lines[i].strip().startswith(MIMICS_10_KEY) or lines[i].strip().startswith(MIMICS_13_KEY) or lines[i].strip().startswith(MIMICS_22_KEY):
                            mFlag = True
                            i+=1
                            continue
                        points.append(lines[i].strip())
                        i+=1
                elif curLine.startswith("coordIndex"):
                    i+=2 # skip the [
                    while not lines[i].strip().startswith("] #coordIndex"):
                        coordIndex.append(lines[i].strip())
                        i+=1
                # check for scale
                elif curLine.startswith("scale"):
                    scale = curLine.split()
                    scale.pop(0)
                i+=1
            points = [x.split() for x in points]
            # remove the comma from the 3rd point
            points = [[int(x[0]), int(x[1]), int(x[2][:-1])] for x in points]
            coordIndex = [x.split() for x in coordIndex]
            coordIndex = [[int(x[0]), int(x[1]), int(x[2])] for x in coordIndex] # remove the 4th point since it is -1

            # if the scale is not [1,1,1] then scale the points
            if scale != [1,1,1]:
                points = [[x[0]*float(scale[0]), x[1]*float(scale[1]), x[2]*float(scale[2])] for x in points]
            if mFlag: # if true the points are in m and need to be converted to mm
                points = [[x[0]*1000, x[1]*1000, x[2]*1000] for x in points]

            pointsVtk = vtk.vtkPoints()
            for p in points:
                pointsVtk.InsertNextPoint(p)

            cellArray = vtk.vtkCellArray()
            for c in coordIndex:
                cellArray.InsertNextCell(3)
                cellArray.InsertCellPoint(c[0])
                cellArray.InsertCellPoint(c[1])
                cellArray.InsertCellPoint(c[2])

            return pointsVtk, cellArray, modelName
      
    def createSegmentationNode(self, points, coordIndex, modelName):
        """
        Create a segmentation node from a list of points and a list of triangles

        :param points: list of points, vtkPoints
        :param coordIndex: list of triangles, vtkCellArray
        """

        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        # create the vtkCellArray
        polyData.SetPolys(coordIndex)
        # create a model node and create a segmenation node from it
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", modelName)
        modelNode.SetAndObservePolyData(polyData)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", modelName + "_segmentation")
        segmentationNode.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(modelNode, segmentationNode)
        # delete the model node
        slicer.mrmlScene.RemoveNode(modelNode)
        return segmentationNode
    
    def getVolumeFromDICOM(self,db,series):
        fileList = db.filesForSeries(series[0])
        scene = slicer.mrmlScene
        seriesNumber = db.fileValue(fileList[0], "0020,0011")
        seriesDescription = db.fileValue(fileList[0], "0008,103E")

        volumeNode = None
        for i in range(scene.GetNumberOfNodesByClass("vtkMRMLScalarVolumeNode")):
            node = scene.GetNthNodeByClass(i, "vtkMRMLScalarVolumeNode")
            if node.GetName() == f"{seriesNumber}: {seriesDescription}":
                volumeNode = node
                break
        if volumeNode is None:
            slicer.util.errorDisplay("Could not find the volume node.")
        
        return volumeNode

    def transformPoints(self,points):
        # compute a bounding box around the points
        import math
        bounds = [0,0,0,0,0,0]
        points.GetBounds(bounds)
        # mirror the points accross one edge of the bounding box
        # get the center of the bounding box
        center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
        # get the max distance from the center
        maxDist = 0
        for i in range(points.GetNumberOfPoints()):
            p = points.GetPoint(i)
            dist = math.sqrt((p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2)
            if dist > maxDist:
                maxDist = dist
        # mirror the x coordinate of the points
        for i in range(points.GetNumberOfPoints()):
            p = points.GetPoint(i)
            points.SetPoint(i, -p[0], -p[1], p[2])
        
        # create a transform
        transform = vtk.vtkTransform()
        transform.Translate(center[0], center[1], center[2])
        # transform.Scale(-1,1,1)
        transform.Translate(-center[0], -center[1], -center[2])
        # apply the transform to the points
        transformFilter = vtk.vtkTransformPolyDataFilter()
        # convert the points to a polydata
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)
        transformFilter.SetInputData(polyData)
        transformFilter.SetTransform(transform)
        transformFilter.Update()
        return transformFilter.GetOutput().GetPoints()


    def generateTiffStacks(self,segmentationNode,volumeNode,outputDir,db,series):
        """
        Generate a tiff stack for each segment in the segmentation node
        """
        # get the points from the segmentation node
        # get the segmentation
        segmentation = segmentationNode.GetSegmentation()
        # get all of the segments
        segmentIDs = segmentation.GetSegmentIDs()
        for segmentID in segmentIDs:
            segmentName = segmentation.GetSegment(segmentID).GetName()
            imageData = segmentationNode.GetSegmentation().GetSegment(segmentID).GetRepresentation("Binary labelmap")
            # get the volume from the volume node
            finalSize = self.getTIFFScaleFactor(volumeNode,segmentationNode.GetClosedSurfaceInternalRepresentation(segmentID),db,series)
            # scale the image data to final size
            resample = vtk.vtkImageResample()
            resample.SetInputData(imageData)
            resample.SetAxisMagnificationFactor(0, finalSize[0])
            resample.SetAxisMagnificationFactor(1, finalSize[1])
            resample.SetAxisMagnificationFactor(2, finalSize[2])
            resample.Update()
            imageData = resample.GetOutput()
            # write image data to a tiff stack
            writer = vtk.vtkTIFFWriter()
            # set no compression
            writer.SetCompressionToNoCompression()
            # set the output directory
            writer.SetFilePattern(f"{outputDir}/{segmentName}_%03d.tif")
            # set the input data
            writer.SetInputData(imageData)
            # write the tiff stack
            writer.Write()

    def getTIFFScaleFactor(self,volumeNode,segmentationData,db,series):
        import numpy as np
        # get the rescale slope and intercept from the dicom metadata
        tags = ["ImagePositionPatient","ImageOrientationPatient","RescaleIntercept","RescaleSlope","PixelSpacing","SliceThickness"]
        dicomMetadata = self.getTags(db,series,tags)
        rs = dicomMetadata["RescaleSlope"]
        ri = dicomMetadata["RescaleIntercept"]
        #HU = pixel_value*slope + intercept
        # get the pixel value of the volume
        volume = volumeNode.GetImageData()
        volumeArray = slicer.util.arrayFromVolume(volumeNode) # index is k,j,i
        # move to ijk
        volumeArray = np.moveaxis(volumeArray,0,-1)
        im = volumeArray*rs + ri
        offset = dicomMetadata["ImagePositionPatient"]

        sx,sy,sz = im.shape
        voxelSize = dicomMetadata["PixelSpacing"] + [dicomMetadata["SliceThickness"]]
        M = np.array([[voxelSize[0],0,0,0],[0,voxelSize[1],0,0],[0,0,voxelSize[2],0],[0,0,0,1]])
        N = np.eye(4)
        N[1,1] = -1
        N[2,2] = -1
        N[1,3] = sy
        N[2,3] = sz

        MN = np.matmul(M,N)
        MN[1,3] = sy
        MN[2,3] = sz
        tMN = np.transpose(MN)
        
        xMin = voxelSize[0]/2
        xMax = voxelSize[0]/2 + voxelSize[0]*sx
        yMin = voxelSize[1]/2
        yMax = voxelSize[1]/2 + voxelSize[1]*sy
        zMin = voxelSize[2]/2
        zMax = voxelSize[2]/2 + voxelSize[2]*sz

        im_ref = {
            'ImageSize': [sx,sy,sz],
            'XIntrinsicLimits': [0.5,sx+0.5],
            'YIntrinsicLimits': [0.5,sy+0.5],
            'ZIntrinsicLimits': [0.5,sz+0.5],
            'PixelExtentInWorldX': voxelSize[0],
            'PixelExtentInWorldY': voxelSize[1],
            'PixelExtentInWorldZ': voxelSize[2],
            'ImageExtentInWorldX': voxelSize[0]*sy,
            'ImageExtentInWorldY': voxelSize[1]*sx,
            'ImageExtentInWorldZ': voxelSize[2]*sz,
            'XWorldLimits': [xMin,xMax],
            'YWorldLimits': [yMin,yMax],
            'ZWorldLimits': [zMin,zMax],
        }
        
        new_ref = im_ref.copy()

        new_ref["XWorldLimits"] = [im_ref["XWorldLimits"][0] + offset[0], im_ref["XWorldLimits"][1] + offset[0]]
        new_ref["YWorldLimits"] = [im_ref["YWorldLimits"][0] + offset[1], im_ref["YWorldLimits"][1] + offset[1]]

        wX = [new_ref["XWorldLimits"][0] - new_ref["PixelExtentInWorldX"]/2 , new_ref["XWorldLimits"][1] - new_ref["PixelExtentInWorldX"]/2]
        wXvect = np.arange(wX[0],wX[1]+((wX[1]-wX[0])/sy),(wX[1]-wX[0])/sy)
        dXvect = range(1,sx+1)

        wY = [new_ref["YWorldLimits"][0] - new_ref["PixelExtentInWorldY"]/2 , new_ref["YWorldLimits"][1] - new_ref["PixelExtentInWorldY"]/2]
        wYvect = np.arange(wY[0],wY[1]+((wY[1]-wY[0])/sx),(wY[1]-wY[0])/sx)
        dYvect = range(1,sy+1)

        wZ = [new_ref["ZWorldLimits"][0] - new_ref["PixelExtentInWorldZ"]/2 , new_ref["ZWorldLimits"][1] - new_ref["PixelExtentInWorldZ"]/2]
        wZvect = np.arange(wZ[0],wZ[1]+((wZ[1]-wZ[0])/sz),(wZ[1]-wZ[0])/sz)
        dZvect = range(1,sz+1)

        pointsVtk = segmentationData.GetPoints()

        points = vtk.util.numpy_support.vtk_to_numpy(pointsVtk.GetData())

        bb = np.array([[np.min(points[:,0]),np.max(points[:,0])],[np.min(points[:,1]),np.max(points[:,1])],[np.min(points[:,2]),np.max(points[:,2])]])

        limsPx = np.digitize(bb[1,:],wXvect)
        limsPy = np.digitize(bb[0,:],wYvect)
        limsPz = np.digitize(bb[2,:],wZvect)
        

        # take a subvolume of the image
        im_small = im[limsPx[0]:limsPx[1],limsPy[0]:limsPy[1],limsPz[0]:limsPz[1]]
        return im_small.shape
        

    
    def getTags(self,db,series,taglist):
        """
        Gets the values of the tags in the taglist for the given series
        """
        import pydicom as dicom
        import numpy as np
        fileList = db.filesForSeries(series[0])
        metadata = {}
        for tagName in taglist:
            tagStr = str(dicom.tag.Tag(tagName))[1:-1].replace(" ","")
            metadata[tagName] = db.fileValue(fileList[0], tagStr)
        for key, value in metadata.items():
            if "\\" in value:
                metadata[key] = [float(x) for x in value.split("\\")]
            else:
                metadata[key] = float(value)
        return metadata
        


            
        



#
# AutoscoperMGenerateTiffStacksTest
#

class AutoscoperMGenerateTiffStacksTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_AutoscoperMGenerateTiffStacks1()

    def test_AutoscoperMGenerateTiffStacks1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('AutoscoperMGenerateTiffStacks1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AutoscoperMGenerateTiffStacksLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')

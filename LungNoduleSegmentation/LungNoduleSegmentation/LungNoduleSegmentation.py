import logging
import os

import vtk
from vtk.util import numpy_support

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import SimpleITK as sitk
import sitkUtils

import csv

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

# for later make these try catch or dependencies



class LungNoduleSegmentation(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Deep Learning Lung Nodule Segmentation"  
        self.parent.categories = ["APPIL Tools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jake Kitzmann (Advanced Pulmonary Physiomic Imaging Laboratory -- University of Iowa Roy J. and Lucille H. Carver College of Medicine)"]

        self.parent.helpText = ""
        self.parent.acknowledgementText = ""


class LungNoduleSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.nodeList = []
        self.currentVolume = None
        self.inSlices = False
        self.visualizeROI = False
        self.parameterSetNode = None

    def selectParameterNode(self):
        segmentEditorSingletonTag = "segmentEditorWidget"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            return
        self.parameterSetNode = segmentEditorNode
        self.ui.segmentEditorWidget.setMRMLSegmentEditorNode(self.parameterSetNode)
        


    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/LungNoduleSegmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LungNoduleSegmentationLogic()


        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        # self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
        self.ui.noduleCentroidButton.connect('clicked(bool)', self.onNoduleCentroidButton)
        self.ui.batchCaseApplyButton.connect('clicked(bool)', self.onBatchCaseApplyButton)
        self.ui.segmentationButton.connect('clicked(bool)', self.onSegmentationButton)
        self.ui.singleROIButton.connect('clicked(bool)', self.apply)

        # Sliders
        self.ui.roiSizeSlider.minimum = 4
        self.ui.roiSizeSlider.maximum = 70
        self.ui.roiSizeSlider.value = 40
        self.ui.roiSizeSlider.connect('valueChanged(int)', self.onRoiSliderValueChanged)

        self.ui.roiSizeLabel.connect('textChanged(QString)', self.userChangedRoiSize)
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value}')
        self.ui.centroidManualButton.connect('clicked(bool)', self.onCentroidManualButton)

        # non isotropic size sliders

        nonIsoSliders = [self.ui.sSliderNonIso, self.ui.cSliderNonIso, self.ui.aSliderNonIso]

        nonIsoSliders[0].connect('valueChanged(int)', self.sSliderNonIsoChanged)
        nonIsoSliders[1].connect('valueChanged(int)', self.cSliderNonIsoChanged)
        nonIsoSliders[2].connect('valueChanged(int)', self.aSliderNonIsoChanged)

        for slider in nonIsoSliders:
            slider.minimum = 4
            slider.maximum = 35
            slider.value = 10

        # LineEdits non iso
        self.ui.sLineEditNonIso.text = self.ui.sSliderNonIso.value
        self.ui.cLineEditNonIso.text = self.ui.cSliderNonIso.value
        self.ui.aLineEditNonIso.text = self.ui.aSliderNonIso.value

        self.ui.aLineEditNonIso.connect('textChanged(QString)', self.aLineEditNonIsoChanged)
        self.ui.cLineEditNonIso.connect('textChanged(QString)', self.cLineEditNonIsoChanged)
        self.ui.sLineEditNonIso.connect('textChanged(QString)', self.sLineEditNonIsoChanged)

        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2}')

        # Combobox
        self.ui.volumeComboBox.setMRMLScene(slicer.mrmlScene)
        self.ui.volumeComboBox.connect('currentNodeChanged(vtkMRMLNode*)', self.onVolumeSelected)
        self.ui.volumeComboBox.renameEnabled = True

        # CheckBoxes
        self.ui.roiCheckBox.connect('clicked(bool)', self.onRoiCheckBox)
        self.onRoiCheckBox()
        self.ui.visualizeCheckbox.connect('clicked(bool)', self.onVisualizeCheckbox)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Radio Buttons
        self.ui.singleCaseRadioButton.setChecked(True)
        self.onSingleCaseRadioButton()        
        self.ui.singleCaseRadioButton.connect('clicked(bool)', self.onSingleCaseRadioButton)
        self.ui.batchCaseRadioButton.connect('clicked(bool)', self.onBatchCaseRadioButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # set up export widget
        self.selectParameterNode()
        self.ui.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentFileExportWidget.setMRMLScene(slicer.mrmlScene)

        self.onVolumeSelected()


        # volume, centroid variables
        self.volume = self.ui.volumeComboBox.currentNode()
        self.centroid = None

        self.ui.moreOptionsCollapsible.collapsed = True


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

    def onSingleCaseRadioButton(self):
        self.ui.batchCase.setVisible(False)
        self.ui.singleCase.setVisible(True)

        self.ui.CollapsibleButton_2.setVisible(True)
        self.ui.label_2.setVisible(True)
        self.ui.segmentEditorWidget.setVisible(True)
        self.ui.label_6.setVisible(True)
        self.ui.segmentFileExportWidget.setVisible(True)

    def onBatchCaseRadioButton(self):
        self.ui.singleCase.setVisible(False)
        self.ui.batchCase.setVisible(True)

        self.ui.CollapsibleButton_2.setVisible(False)
        self.ui.label_2.setVisible(False)
        self.ui.segmentEditorWidget.setVisible(False)
        self.ui.label_6.setVisible(False)
        self.ui.segmentFileExportWidget.setVisible(False)

    def aSliderNonIsoChanged(self):
        self.ui.aLineEditNonIso.text = self.ui.aSliderNonIso.value * 2

    def cSliderNonIsoChanged(self):
        self.ui.cLineEditNonIso.text = self.ui.cSliderNonIso.value * 2

    def sSliderNonIsoChanged(self):
        self.ui.sLineEditNonIso.text = self.ui.sSliderNonIso.value * 2

    def aLineEditNonIsoChanged(self):
        self.ui.aSliderNonIso.value = (int(self.ui.aLineEditNonIso.text)) / 2

    def cLineEditNonIsoChanged(self):
        self.ui.cSliderNonIso.value = (int(self.ui.cLineEditNonIso.text)) / 2

    def sLineEditNonIsoChanged(self):
        self.ui.sSliderNonIso.value = (int(self.ui.sLineEditNonIso.text)) / 2
    
    def onRoiCheckBox(self):
        gridLayoutComponents = [self.ui.sLabelNonIso, self.ui.sLineEditNonIso, self.ui.sSliderNonIso, self.ui.cLabelNonIso, self.ui.cLineEditNonIso, self.ui.cSliderNonIso, self.ui.aLabelNonIso, self.ui.aLineEditNonIso, self.ui.aSliderNonIso]
        
        if self.ui.roiCheckBox.isChecked():
           for component in gridLayoutComponents:
                component.setVisible(True)
        else:
            for component in gridLayoutComponents:
                component.setVisible(False)

    # Checkbox to switch view to ROI after it's created
    def onVisualizeCheckbox(self):
        if self.ui.visualizeCheckbox.isChecked():
            self.visualizeROI = True
        else:
            self.visualizeROI = False

    # called before nodule centroid is created to clear any existing centroids
    def clearNoduleCentroids(self):
        nodeList = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
        if len(nodeList) != 0:
            for node in nodeList:
                if node.GetName() == 'nodule_centroid':
                    slicer.mrmlScene.RemoveNode(node)

    # called on mouse click centroid button, allows user to place centroid where they want with the mouse
    def onNoduleCentroidButton(self):
        self.clearNoduleCentroids() # clear prev centroid

        # create new F node for centroid
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.SetName('nodule_centroid')

        # assign it when the user clicks
        slicer.modules.markups.logic().StartPlaceMode(0)        
        self.inSlices = False # boolean used to see if the program needs to switch from xyz to slices

    # called on manual centroid button, allows user to input centroid in slice coordinates (s,c,a)
    def onCentroidManualButton(self):
        self.clearNoduleCentroids() # clear prev centroid

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
        fiducialNode.SetName('nodule_centroid')

        # let the apply button know that the centroid is in slices, slices are taken directly from the textboxes in applyButton
        self.inSlices = True

    # called when the user changes the roi size slider
    def onRoiSliderValueChanged(self):
        self.ui.roiSizeLabel.setText(f'{self.ui.roiSizeSlider.value * 2 }') # set the label to the slider value

    # called when the user changes the roi size label
    def userChangedRoiSize(self):
        self.ui.roiSizeSlider.value = (int(self.ui.roiSizeLabel.text) / 2) # set the slider to what the user input

    # convert the xyz centroid to slices
    def getCentroidFromXYZ(self, centroid, volume):
        difference_slices_int = []

        # Get the origin and spacing of the volume
        origin = volume.GetOrigin()
        spacing = volume.GetSpacing()
        difference_vector = np.subtract(centroid, origin) # location of centroid in xyz
        difference_slices = np.divide(difference_vector, spacing) # location of centroid in sca

        # convert to int and take absolute value because slices cannot be negative
        for i in range(3):
            slice = abs(int(difference_slices[i]))
            difference_slices_int.append(slice)
        
        return difference_slices_int
    
    # create DLS class and segment nodule
    def segment(self, volume_np):
        model = DeepLearningSegmentation()
        return model.run(torch.tensor(volume_np).to(torch.float32).unsqueeze(0).unsqueeze(0)) # two extra dimensions for batch and channel

    # project ROI segmentation back to the full volume
    def project(self, segmentation, centroid, volume):

        # remove uneeded channels
        segmentation = segmentation[0, 0, :, :, :]

        # threshold to binary
        segmentation[segmentation > .5] = 1
        segmentation[segmentation <= .5] = 0

        roi_size_expansion = [20,20,20]

        # create segmentation mapped to full volume
        full_segmentation = np.zeros(slicer.util.arrayFromVolume(volume).shape)

        print(segmentation.shape)
        print(full_segmentation.shape)

        if self.inSlices: # if centroid is already in slices (from manual entering)
            # conversion factor to map the roi to the full volume
            centroid_slices = [int(self.ui.sLineEdit.text),int(self.ui.cLineEdit.text),int(self.ui.aLineEdit.text)]
            conversion = [centroid_slices[0] - roi_size_expansion[0], centroid_slices[1] - roi_size_expansion[1], centroid_slices[2] - roi_size_expansion[2]] 
        else: # if centroid is in xyz coordinates
            node = slicer.mrmlScene.GetFirstNodeByName('nodule_centroid')
            centroid = [0, 0, 0]
            node.GetNthFiducialPosition(0, centroid)
            centroid_slices = self.getCentroidFromXYZ(centroid, volume)
            # conversion factor to map the roi to the full volume
            conversion = [centroid_slices[0] - roi_size_expansion[0], centroid_slices[1] - roi_size_expansion[1], centroid_slices[2] - roi_size_expansion[2]] 

        # assign the segmentation to the full volume
        for i in range(segmentation.shape[0]):
            for j in range(segmentation.shape[1]):
                for k in range(segmentation.shape[2]):
                    # switch a and s axes on seg because numpy is fucking stupid and thinks it's special
                    full_segmentation[i + conversion[2], j + conversion[1], k + conversion[0]] = segmentation[i,j,k] 

        return full_segmentation
    
    def create_segmentation_volume(self, volume, segmentation):
        # Convert the numpy array to a vtkMRMLLabelMapVolumeNode
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(labelmap_node, segmentation)
        
        # find spacing and origin of image so we put the seg in the right spot
        spacing = volume.GetSpacing()
        origin = volume.GetOrigin()
        
        # make a matrix to assign
        direction_matrix = vtk.vtkMatrix4x4()

        # give the matrix the vals it needs
        volume.GetIJKToRASDirectionMatrix(direction_matrix)

        # apply metadata to segmentation
        labelmap_node.SetSpacing(spacing)
        labelmap_node.SetOrigin(origin)
        labelmap_node.SetIJKToRASDirectionMatrix(direction_matrix)

        # Convert the labelmap to a segmentation node
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node)

        # Set segmentation geometry to match the known volume
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume)
   
        # Set the color of the segment to purple
        segment_id = segmentation_node.GetSegmentation().GetNthSegmentID(0)  # Assuming the first segment
        segmentation_node.GetSegmentation().GetSegment(segment_id).SetColor((178./255. , 102./205., 1))
        segmentation_node.SetName(self.ui.volumeComboBox.currentNode().GetName() + '_segmentation')
        
        # Remove the  labelmap node 
        slicer.mrmlScene.RemoveNode(labelmap_node)

        self.allow_editing(segmentation_node=segmentation_node)


    # create a segmentation of nodule on roi generated 
    def onSegmentationButton(self):

        # np array of the roi
        volumeArray = slicer.util.arrayFromVolume(self.apply())

        if volumeArray is None:
            logging.error('No ROI created')
            return

        # normalize the roi for better network performance
        volumeArray = (volumeArray - volumeArray.min()) / (volumeArray.max() - volumeArray.min())
        segment = self.segment(volumeArray)
        
        # project the segmentation back to the full volume
        full_segmentation = self.project(segment, self.centroid, self.volume)

        # create the segmentation volume
        self.create_segmentation_volume(self.volume, full_segmentation)

        print("Segmentation created!")

    def allow_editing(self, segmentation_node):
        self.ui.segmentFileExportWidget.setSegmentationNode(segmentation_node)
        self.ui.segmentEditorWidget.setSegmentationNode(segmentation_node)
        self.ui.segmentEditorWidget.setMasterVolumeNode(self.ui.volumeComboBox.currentNode())

    # preliminary math to be plugged into create_roi
    def apply(self):

        # get centroid
        node = slicer.mrmlScene.GetFirstNodeByName('nodule_centroid')

        # make sure the user isn't dumb and actually has an image to segment
        if self.volume is None:
            logging.error('No volume selected')
            return  

        # make suere the user isn't dumb and actually has a centroid
        if node is None:
            logging.error('No centroid selected')
            return
    
        # if we're working from xyz centroid
        if not self.inSlices:
            # Get the centroid of the nodule
            centroid = [0, 0, 0]
            node.GetNthFiducialPosition(0, centroid)
            #print(f'centroid: {centroid}')

            difference_slices_int = self.getCentroidFromXYZ(centroid, self.volume)

            self.ui.sLineEdit.text = str(difference_slices_int[0])
            self.ui.cLineEdit.text = str(difference_slices_int[1])
            self.ui.aLineEdit.text = str(difference_slices_int[2])

        # otherwise the math is a lot easier
        else:
            difference_slices_int = [int(self.ui.sLineEdit.text), int(self.ui.cLineEdit.text), int(self.ui.aLineEdit.text)]
            self.centroid = difference_slices_int
        
       # size list init
        size = [0,0,0]

        # if the roi isn't isotropic
        if self.ui.roiCheckBox.isChecked():
            size[0] = int(self.ui.sLineEditNonIso.text)
            size[1] = int(self.ui.cLineEditNonIso.text)
            size[2] = int(self.ui.aLineEditNonIso.text)
            #print(f'Non-isotropic size: {size}')
        else:
            size = [self.ui.roiSizeSlider.value * 2, self.ui.roiSizeSlider.value * 2, self.ui.roiSizeSlider.value * 2 ]
            #print(f'Isotropic size: {size}')

        # create roi
        return self.create_roi(self.volume, difference_slices_int, size)

    # create the roi from the centroid and size
    def create_roi(self, volume, centroid, size):

        sitk_img = sitkUtils.PullVolumeFromSlicer(volume.GetID())
        imageROI = ImageROI()

        #print(f'Creating ROI with size {size} and centroid {centroid}')


        roi_img_np = imageROI.create_roi_image(sitk_img, size, centroid)

        roi_img_volume = slicer.util.addVolumeFromArray(roi_img_np)
        roi_img_volume.SetName(self.volume.GetName() + '_roi')
        roi_img_volume.SetSpacing(volume.GetSpacing())
        roi_img_volume.SetOrigin(volume.GetOrigin())

        direction_matrix = vtk.vtkMatrix4x4()

        # Retrieve the direction matrix from the known volume node
        volume.GetIJKToRASDirectionMatrix(direction_matrix)
        roi_img_volume.SetIJKToRASDirectionMatrix(direction_matrix)

        if not self.ui.interpolationCheckBox.isChecked():
            roi_img_display_node = roi_img_volume.GetDisplayNode()
            roi_img_display_node.SetInterpolate(0)

        if self.visualizeROI:
            slicer.util.setSliceViewerLayers(background=roi_img_volume, fit=True)

        
        if not self.ui.visualizeCheckbox.isChecked():
            slicer.mrmlScene.RemoveNode(roi_img_volume)


        return roi_img_volume

    def onVolumeSelected(self):
        self.currentVolume = self.ui.volumeComboBox.currentNode()
        self.volume = self.currentVolume

    # BATCH CASES

    def onBatchCaseApplyButton(self):
        volumeListPath = self.ui.batchVolumeLineEdit.text
        centroidListPath = self.ui.batchCentroidLineEdit.text
        outputDir = self.ui.batchOutputLineEdit.text

        volumePaths = os.listdir(volumeListPath)

        cases = []

        with open(centroidListPath) as file_obj: 
            reader_obj = csv.reader(file_obj)
            rows = []
            for row in reader_obj:
                rows.append(row)

        # create pairs of volumes and centroids for each case
        for volumePath in volumePaths:
            pid = volumePath.split('/')[-1].split('_')[0]
            for row in rows:
                if pid == row[0]:
                    cases.append(self.case(volumeListPath + volumePath, row[0], row[1], row[2], row[3], row[4]))

        for case in cases:
            volume = slicer.util.loadVolume(case.volumePath)
            centroid = [int(case.centroidS), int(case.centroidC), int(case.centroidA)]
            size = [int(case.size), int(case.size), int(case.size)]

            roi = self.create_roi(volume, centroid, size)

            if self.ui.batchROIVolumeCheckBox.isChecked():
                success = slicer.util.saveNode(roi, outputDir + '/' + case.PID + '_roi.nrrd')
                if success:
                    print(f'Volume {case.PID} saved to {outputDir}')
                else:
                    print(f'Failed to save volume {case.PID} to {outputDir}')


            # normalize the roi for better network performance
            roi = (roi - roi.min()) / (roi.max() - roi.min())
            segment = self.segment(roi)

            # project the segmentation back to the full volume
            full_segmentation = self.project(segment, self.centroid, self.volume)

            # create the segmentation volume
            self.create_segmentation_volume(self.volume, full_segmentation)

            success = slicer.util.saveNode(full_segmentation)
            print("Segmentation created!")

                

    # class to store case information in batch processing
    class case:
        def __init__(self,volumePath, PID, centroidS, centroidC, centroidA, size):
            self.volumePath = volumePath
            self.PID = PID
            self.centroidS = centroidS
            self.centroidC = centroidC
            self.centroidA = centroidA
            self.size = size


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
        if self._parameterNode is not None:
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
        
        self.ui.volumeComboBox.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))


        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

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

        nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.volumeComboBox.currentNodeID)


 
        self._parameterNode.EndModify(wasModified)

class LungNoduleSegmentationLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        pass

    def calculate_nodule_ROI(self):
        pass

class LungNoduleROITest(ScriptedLoadableModuleTest):

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_t_ApplyThreshold1()

    def test_t_ApplyThreshold1(self):

        pass

class ImageROI:
    def __init__(self):
        pass
        # print("ImageROI object created")
    # create roi image from sitk image
    # inputs:
    # img -> SimpleITK image
    # centroid -> centroid of lung nodule in coordinates using [1,1,1] spacing NOT SLICER
    # expansion -> amount to expand ROI from centroid in +/- for each direction
    def create_roi_image(self, img, expansion, centroid):
        # expand roi from centroid
        roi = {
            'coronal' : [centroid[1]-int((expansion[1]) / 2), centroid[1]+int((expansion[1]) / 2)],
            'sagittal' : [centroid[0]-int((expansion[0]) / 2), centroid[0]+int((expansion[0]) / 2)],
            'axial' : [centroid[2]-int((expansion[2])/ 2), centroid[2]+int((expansion[2]) / 2)]

        }

        # convert to numpy array and cut down to roi around nodule centroid
        np_img = sitk.GetArrayFromImage(img)
        np_roi = np_img[roi['axial'][0]:roi['axial'][1],
                        roi['coronal'][0]:roi['coronal'][1],
                        roi['sagittal'][0]:roi['sagittal'][1]]

        return np_roi
    
class DeepLearningSegmentation:
    def __init__(self):
        pass

    def run(self, inputVolume):


        model = UNet()
        model.load_state_dict(torch.load('/Users/jacob_kitz/Desktop/SemiAutoLungNoduleSegmentation/LungNoduleSegmentation/LungNoduleSegmentation/model_weights.pth', map_location=torch.device('cpu')))
        # print(model.eval())
        
        output = model(inputVolume).detach().numpy()
        # print(f'shape: {output.shape}')
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=1,   
                 out_channels=1,  # 1 for binary segmentation
                 features=[64, 128, 256, 512]):  # Number of feature maps for each layer
        super(UNet, self).__init__()
        
        self.downs = nn.ModuleList()  # Down convolutions
        self.ups = nn.ModuleList()    # Up convolutions
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # Max pooling for 3D

        # down convolutions
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature  # Update in_channels for next layer
            
        # bottleneck layer
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # up convolutions
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2))  # Transpose convolution
            self.ups.append(DoubleConv(feature * 2, feature))
            
        # Final output layer (segmentation output)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)  # Output layer for segmentation

    def forward(self, x):
        skip_connections = []

        # Downward pass: encoding layers
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck (lowest part of U-Net)
        x = self.bottleneck(x)

        # Reverse the skip connections for upward pass
        skip_connections = skip_connections[::-1]

        # Upward pass: decoding layers
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transpose convolution (upscale)
            skip_connection = skip_connections[idx // 2]

            # Resize if needed (if the sizes mismatch)
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            # Concatenate skip connection with the upsampled feature map
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate along the channel dimension
            x = self.ups[idx + 1](x)  # Apply the second double convolution
        return self.final_conv(x)
        
    
class diceloss(torch.nn.Module):
    def init(self):
        super(diceLoss, self).init()
    def forward(self,pred, target):
        smooth = 1.
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )





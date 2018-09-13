from __main__ import vtk, qt, ctk, slicer
from vtk.util import numpy_support
import numpy as np
from slicer.ScriptedLoadableModule import *
import os
import time
import unittest

#
# ROISegment
#
class ROISegmentation1(ScriptedLoadableModule):
  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self,parent)
    parent.title = "ROISegmentation1"
    parent.categories = ["Examples"]
    parent.dependencies = []
    parent.contributors = [ "Sangni Xu"]
    parent.helpText = """
    Segmentation with ROI annotation tool. Use ROI annotation tool to define 
    the boundary of ROI and give seeding points in ROI.
    """
    parent.acknowledgementText = """
    3D ROI segmentation
    """ 
    
#
# The main widget
#
class ROISegmentation1Widget(ScriptedLoadableModuleWidget):
  def __init__(self, parent):
    ScriptedLoadableModuleWidget.__init__(self,parent)
    self.logic = ROISegmentationLogic()
    self.parameterNode = None


  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    ##########################
    ## for debugging and testing
    ####
    self.reloadCButton = ctk.ctkCollapsibleButton()
    self.reloadCButton.objectName = 'ReloadFrame'
    self.reloadCButton.setLayout(qt.QHBoxLayout())
    self.reloadCButton.setText("Reload and test the Module")
    self.layout.addWidget(self.reloadCButton)
    self.reloadCButton.collapsed = False
 
    # reload button
    reloadButton = qt.QPushButton("Reload")
    reloadButton.toolTip = "Reload this Module"
    reloadButton.name = "ROISegmentation1 Reload"
    reloadButton.connect('clicked()', self.onReload)
    self.reloadButton = reloadButton
    self.reloadCButton.layout().addWidget(self.reloadButton)


    # reload and test button
    # remove it after development
    self.testButton = qt.QPushButton("Test")
    self.testButton.toolTip = "run the self tests."
    self.testButton.connect('clicked()', self.onTest)
    self.reloadCButton.layout().addWidget(self.testButton)
    

    
    ##########################
    ## input volumn
    ####
    self.inputCButton = ctk.ctkCollapsibleButton()
    self.inputCButton.text = "Inputs"
    self.layout.addWidget(self.inputCButton)

    # layout within collapsible button
    # read input volumn node
    self.segmentationFormLayout = qt.QFormLayout(self.inputCButton)
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addEnabled = True
    self.inputSelector.removeEnabled = True
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Input Volume: ", self.inputSelector)
    
    # read seeding ROI inside tumor
    self.seedingSelector = slicer.qMRMLNodeComboBox()
    self.seedingSelector.nodeTypes = ( ("vtkMRMLAnnotationROINode"), "" )
    self.seedingSelector.selectNodeUponCreation = True
    self.seedingSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Input seeding ROI: ", self.seedingSelector)

    # read seeding boundary ROI
    self.seedingBSelector = slicer.qMRMLNodeComboBox()
    self.seedingBSelector.nodeTypes = ( ("vtkMRMLAnnotationROINode"), "" )
    self.seedingBSelector.selectNodeUponCreation = True
    self.seedingBSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Input boundary ROI: ", self.seedingBSelector)

    self.labelmapSelector = slicer.qMRMLNodeComboBox()
    self.labelmapSelector.nodeTypes = ( ("vtkMRMLLabelMapVolumeNode"), "" )
    self.labelmapSelector.addEnabled = True
    self.labelmapSelector.removeEnabled = True
    self.labelmapSelector.setMRMLScene( slicer.mrmlScene )
    self.segmentationFormLayout.addRow("Label Map: ", self.labelmapSelector)
   

    ##########################
    ## apply button
    ####
    applyButton = qt.QPushButton("Apply")
    applyButton.toolTip = "Run the algorithm."
    self.layout.addWidget(applyButton)
    applyButton.connect('clicked()', self.onApply)
    self.applyButton = applyButton

    self.layout.addStretch(1)


  # When the apply button is clicked
  def onApply(self):
    self.applyButton.text = "Working..."
    self.applyButton.repaint()
    slicer.app.processEvents()
    self.logic.roiSegment(self.inputSelector,self.seedingSelector.currentNode(),self.seedingBSelector.currentNode(),self.labelmapSelector)
    self.applyButton.text = "Apply"

  # Reload the Module
  def onReload(self, moduleName = "ROISegmentation1"):
    import imp, sys, os, slicer
    widgetName = moduleName + "Widget"
    fPath = eval('slicer.modules.%s.path' % moduleName.lower())
    p = os.path.dirname(fPath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)
    fp = open(fPath, "r")
    globals()[moduleName] = imp.load_module(
        moduleName, fp, fPath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()
    print "the module name to be reloaded,", moduleName
    # find the Button with a name 'moduleName Reolad', then find its parent (e.g., a collasp button) and grand parent (moduleNameWidget)
    parent = slicer.util.findChildren(name = '%s Reload' % moduleName)[0].parent().parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass
    item = parent.layout().itemAt(0)
    while item:
      parent.layout().removeItem(item)
      item = parent.layout().itemAt(0)
    globals()[widgetName.lower()] = eval('globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()

  # test the module
  def onTest(self,moduleName="ROISegmentation1"):
    try:
      evalString = 'globals()["%s"].%sTest()' % (moduleName, moduleName)
      tester = eval(evalString)
      tester.runTest(self.inputSelector,self.labelmapSelector)
    except Exception, e:
      import traceback
      traceback.print_exc()
      slicer.util.warningDisplay('Exception!\n\n' + str(e) + "\n\nSee Python Console for Stack Trace",
                                 windowTitle="Reload and Test", )



#
# ROISegmentationLogic
#

class ROISegmentation1Logic(ScriptedLoadableModuleLogic):
  def __init__(self,parent = None):
    ScriptedLoadableModuleLogic.__init__(self,parent)

  def roiSegment(self, inputSelector,seedingSelector,seedingBSelector,labelmapSelector):
    ##########################
    ## read input volume node
    ####
    inputVolume     = inputSelector.currentNode()
    # extract array
    inputVolumeData = slicer.util.array(inputVolume.GetID())
    # input label map
    labelMap = labelmapSelector.currentNode()

    
    ##########################
    ## read seeding ROI
    ####
    seedingROI = seedingSelector
    boundROI = seedingBSelector
    
    # center and radius of foreground seeding points
    seedingROIcenter = [0,0,0]
    seedingROIRadius = [0,0,0]
    seedingROI.GetXYZ(seedingROIcenter)
    seedingROI.GetRadiusXYZ(seedingROIRadius)
    roiSBox = vtk.vtkBox()
    roiSBox.SetBounds(seedingROIcenter[0] - seedingROIRadius[0], seedingROIcenter[0] + seedingROIRadius[0], seedingROIcenter[1] - seedingROIRadius[1], seedingROIcenter[1] + seedingROIRadius[1], seedingROIcenter[2] - seedingROIRadius[2], seedingROIcenter[2] + seedingROIRadius[2])

    # center and radius of bounding box
    boundROIcenter = [0,0,0]
    boundROIRadius = [0,0,0]
    boundROI.GetXYZ(boundROIcenter)
    boundROI.GetRadiusXYZ(boundROIRadius)
    roiBBox = vtk.vtkBox()
    roiBBox.SetBounds(boundROIcenter[0] - boundROIRadius[0], boundROIcenter[0] + boundROIRadius[0], boundROIcenter[1] - boundROIRadius[1], boundROIcenter[1] + boundROIRadius[1], boundROIcenter[2] - boundROIRadius[2], boundROIcenter[2] + boundROIRadius[2])

    # transform between ijk and ras
    rasToBox = vtk.vtkMatrix4x4()
    if boundROI.GetTransformNodeID() != None:
      roiBoxTransformNode = slicer.mrmlScene.GetNodeByID(boundROI.GetTransformNodeID())
      boxToRas = vtk.vtkMatrix4x4()
      roiBoxTransformNode.GetMatrixTransformToWorld(boxToRas)
      rasToBox.DeepCopy(boxToRas)
      rasToBox.Invert()
    
    rasToSBox = vtk.vtkMatrix4x4()
    if seedingROI.GetTransformNodeID() != None:
      seedingBoxTransformNode = slicer.mrmlScene.GetNodeByID(seedingROI.GetTransformNodeID())
      sboxToRas = vtk.vtkMatrix4x4()
      seedingBoxTransformNode.GetMatrixTransformToWorld(sboxToRas)
      rasToSBox.DeepCopy(sboxToRas)
      rasToSBox.Invert()


    ijkToRas = vtk.vtkMatrix4x4()
    labelMap.GetIJKToRASMatrix( ijkToRas )

    ijkToBox = vtk.vtkMatrix4x4()
    ijkToSBox = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Multiply4x4(rasToBox,ijkToRas,ijkToBox)
    vtk.vtkMatrix4x4.Multiply4x4(rasToSBox,ijkToRas,ijkToSBox)
    ijkToBoxTransform = vtk.vtkTransform()
    ijkToBoxTransform.SetMatrix(ijkToBox)
    roiBBox.SetTransform(ijkToBoxTransform)
    ijkToBoxTransform.SetMatrix(ijkToSBox)
    roiSBox.SetTransform(ijkToBoxTransform)


    imageData=labelMap.GetImageData()

    # show boudning area and seeding area in label map
    # label the bounding region
    functionToStencil = vtk.vtkImplicitFunctionToImageStencil()
    functionToStencil.SetInput(roiBBox)
    functionToStencil.SetOutputOrigin(imageData.GetOrigin())
    functionToStencil.SetOutputSpacing(imageData.GetSpacing())
    functionToStencil.SetOutputWholeExtent(imageData.GetExtent())
    functionToStencil.Update()

    stencilToImage=vtk.vtkImageStencil()
    stencilToImage.SetInputData(imageData)
    stencilToImage.SetStencilData(functionToStencil.GetOutput())
    stencilToImage.ReverseStencilOn()
    stencilToImage.SetBackgroundValue(2.00) # set background label to 2
    stencilToImage.Update()
    imageData.DeepCopy(stencilToImage.GetOutput())

    # label the seeding region
    functionToStencil = vtk.vtkImplicitFunctionToImageStencil()
    functionToStencil.SetInput(roiSBox)
    functionToStencil.SetOutputOrigin(imageData.GetOrigin())
    functionToStencil.SetOutputSpacing(imageData.GetSpacing())
    functionToStencil.SetOutputWholeExtent(imageData.GetExtent())
    functionToStencil.Update()

    stencilToImage=vtk.vtkImageStencil()
    stencilToImage.SetInputData(imageData)
    stencilToImage.SetStencilData(functionToStencil.GetOutput())
    stencilToImage.ReverseStencilOn()
    stencilToImage.SetBackgroundValue(3.00)
    stencilToImage.Update()
    imageData.DeepCopy(stencilToImage.GetOutput())

    # update label map
    labelMap.Modified()


    ##########################
    ## output label map
    ####
    labelMapData = slicer.util.array(labelMap.GetID())

  
    ##########################
    ## segmentation 
    ####
    #determine threshold from seeding points
    s = np.where(labelMapData==3) # get seeding points location
    threshold = np.median(inputVolumeData[s[0],s[1],s[2]])

    # check voxels within the bounding region
    t = np.where(np.logical_or(labelMapData ==2 , labelMapData==3))
    for w in range(len(t[0])):
      i = t[0][w]
      j = t[1][w]
      k = t[2][w]
      if (inputVolumeData[i,j,k]<threshold+50) and (inputVolumeData[i,j,k]>threshold-30):
        labelMapData[i,j,k] = 1

    #######################################################
    # set background to zero
    labelMapData[labelMapData != 1] = 0

 
 #   
 # test case
 #
class ROISegmentation1Test(ScriptedLoadableModuleTest):
  def setUp(self):
    slicer.mrmlScene.Clear(0)

  def runTest(self,samplevolume,labelmapSelector):
    #self.setUp()
    self.test_ROISegmentation(samplevolume,labelmapSelector)

  def test_ROISegmentation(self,samplevolume,labelmapSelector):

    roiSNode = slicer.vtkMRMLAnnotationROINode()
    roiBNode = slicer.vtkMRMLAnnotationROINode()
    roiSNode.SetXYZ(-5.12,26.72,27.54)
    roiSNode.SetRadiusXYZ(7.87,7.87,9.84)
    roiBNode.SetXYZ(-3.93,26.71,28.72)
    roiBNode.SetRadiusXYZ(20.07,18.88,19.67)

    roiBNode.Initialize(slicer.mrmlScene)
    roiBNode.Initialize(slicer.mrmlScene)

    logic = ROISegmentation1Logic()
    logic.roiSegment(samplevolume,roiSNode,roiBNode,labelmapSelector)

    self.delayDisplay("test passed.")

  





from IPython.display import display
import ipywidgets as widgets
from ipywidgets import Tab, FileUpload
from ipywidgets import Layout, Output, Dropdown, Button, IntText, Checkbox, Tab
from ipywidgets import FloatText, RadioButtons

from torchvision import transforms

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


from ipywidgets import Layout, Output, Dropdown, Button, IntText, Checkbox, Tab, HBox, GridBox
from IPython.display import display, clear_output
import logging
import os
from .lib.__init__ import *


class GUI():
    
    #===========Initialization of Variables==================
    
    def __init__(self, layout, output, sysIn, logger, imageManager, CONFIG_DATA):
        
        # Objekte
        self.CONFIG_DATA = CONFIG_DATA
        self.layout = layout
        self.output = output
        self.sysIn = sysIn
        self.logger = logger
        self.loggerOutput= self.logger.loggerOutput
        self.imgManager = imageManager
        
        #path-variables
        self.notebookDir = os.getcwd()#"/home/shared" #get current working dir //  self.notebookDir needs to be replaced with "/home/shared"
        self.pathData = os.path.join(self.notebookDir, 'models')
        self.locationOfDatasetImages=os.path.join(self.notebookDir, 'Datasets')
        self.pathDataset = ""
        self.pathArchitecture = ""
        self.pathModel = ""
        self.uploadImagesFolder="upload"
        
        self.datasets=self.CONFIG_DATA['DATASETS']
        self.attacks=self.CONFIG_DATA['ATTACKS']
        self.architectures=self.CONFIG_DATA['ARCHITECTURES']
        
        self.NWrapper = Checkbox(description = "NormalizationWrapper")
        #names MUST be identical to parameters in attack-classes -> def constants  
        self.attackParameters={
            'eps':FloatText(description='Epsilon', step=0.0001, value=0.3),
            'iterations':IntText(description='Iterations', value=1),
            'stepsize':FloatText(description='Stepsize', step=0.0001, value=0.0002),
            'momentum':FloatText(description='Momentum', step=0.1, value=0.9),
            'norm':RadioButtons(options=[('Linf','inf'),('L2','l2')],description='Norm'),
            'loss':RadioButtons(options=['CrossEntropy','LogitsDiff', 'ConfDiff'],description='Lossfunction'),
            'normalize_grad':Checkbox(value=True, description='Normalize Gradient'),
            'early_stopping':FloatText(description='Early Stopping', step=0.1),
            'restarts':IntText(description='Restarts'),
            'init_noise_generator':Dropdown(options=self.CONFIG_DATA['NOISE_GENERATORS'], description='Noise Generator'),
            'save_trajectory':Checkbox(description='Save Trajectory'),
            'targeted':Checkbox(description='targeted'),
            'target_label':Dropdown(description='Target Label', disabled=True),
            'batchSize':IntText(description='batchSize', value=1),
            'imageSelection':RadioButtons(description='Images from', options=['Dataset','Uploaded'])
        }        
        
        #define Widgets
        self.uploadDatasetOptions=[key for key in self.datasets]
        self.uploadArchitectureOptions=[key for key in self.architectures]
        self.dropdownDataset=Dropdown()        
        self.dropdownArchitecture=Dropdown()    
        self.dropdownModel=Dropdown()
        self.uploadDatasetDropdown=Dropdown(options=self.uploadDatasetOptions)
        self.uploadArchitecturesDropdown=Dropdown(options=self.uploadArchitectureOptions)
        self.attackSelection=self.createAttackSelectionMenu()
        self.uploaderImage=FileUpload(accept='image/*',  multiple=True, description="Upload Picture")
        self.uploaderModel=FileUpload(accept='.pt, .pth, .txt', multiple=True, description="Upload Model")
        self.accordionMenu=self.createAccordionMenu()
        self.btn_compute=Button(description='Compute', button_style='info')
        self.btn_clearOut=Button(description='Clear Output', button_style='info')
        
        self.epsilonSteps = IntText(description='Epsilon Steps', value=1)
        self.btn_vc = Button(description ='Compute VCs', button_style='info')
        
        
        #other
        self.model=None
        self.imageViewer=None
        self.selectionViewer=None
        self.dataset=None
        self.infoCount=0
        self.error=""
    
    #============Main Method===================================
    
    def run(self):
        self.displayGUI()
        self.initializeDropdown()
        self.observeWidgets()
   
    #==========Other Methods================ 
    
    def displayGUI(self):
        display(self.makeMenu())
            
    def observeWidgets(self):
        self.accordionMenu.children[1].observe(self.eventHandlerAccordionMenu, names='selected_index')
        self.dropdownDataset.observe(self.eventHandlerDropdownDataset, 'value')
        self.dropdownArchitecture.observe(self.eventHandlerDropdownArchitecture, 'value')
        self.dropdownModel.observe(self.eventHandlerDropdownModel, 'value')
        self.uploaderImage.observe(self.eventHandlerUploaderImage, 'value')
        self.uploaderModel.observe(self.eventHandlerUploaderModel, 'value')
        self.btn_compute.on_click(self.eventHandlerBtnCompute)
        self.btn_clearOut.on_click(self.event_handler_btn_clearOut)
        self.btn_vc.on_click(self.event_handler_btn_vc)
        self.attackParameters['imageSelection'].observe(self.eventHandlerImageSelection, 'value')
        self.attackParameters['targeted'].observe(self.eventHandlerTargeted, 'value')

        
    def initializeDropdown(self):
        self.dropdownDataset.options = self.sysIn.getSubdirs(self.pathData)
        self.refreshDropdownDataset()
        #self.updateDropdownImageOptions()
        
    def refreshDropdownDataset(self):
        self.eventHandlerDropdownDataset({'new':self.dropdownDataset.value})
    
    def refreshDropdownArchitecture(self):
        self.eventHandlerDropdownArchitecture({'new':self.dropdownArchitecture.value})
        
    def refreshDropdownModel(self):
        self.eventHandlerDropdownModel({'new':self.dropdownModel.value})
    
    def updateDropdownArchitectureOptions(self):
        options = self.sysIn.getSubdirs(self.pathDataset)
        self.dropdownArchitecture.options=options
        if options == []:
            self.logger.printInfo("Could not find any architectures in "+self.pathDataset)
            raise Exception()
        else:
            self.refreshDropdownArchitecture() #reminder_1
    
    def updateDropdownModelOptions(self):
        options = self.sysIn.getSubdirs(self.pathArchitecture)
        self.dropdownModel.options=options
        if options == []:
            self.logger.printInfo("Could not find any models in "+self.pathArchitecture)
            raise Exception()
        else:      
            self.refreshDropdownModel() #reminder_1
        
    def updatePathModel(self, currentModel):
        self.pathModel=os.path.join(self.pathArchitecture, currentModel)
        modelFiles=self.sysIn.getFiles(self.pathModel)
        modelNames = self.sysIn.filterforEndings(modelFiles, [".pt", ".pth"])
        if modelFiles == []:
            self.logger.printInfo("Could not find any modelFiles in "+self.pathModel)
        else:
            self.pathModel=os.path.join(self.pathModel, modelNames.pop())
    
    def printWrongArchitectureInfo(self):
        infoMessage = "No matching Constructor defined for: {}\n Defined Constuctores are:".format(self.dropdownArchitecture.value)             
        for key in self.architectures:
            infoMessage = infoMessage + "\n"+ key
        self.logger.printInfo(infoMessage)
        
    def printWrongDatasetInfo(self):
        infoMessage = "No matching Constructor defined for: {}\n Defined Constuctores are:".format(self.dropdownDataset.value)             
        for key in self.datasets:
            infoMessage = infoMessage + "\n"+ key
        self.logger.printInfo(infoMessage)
    
    def loadModel(self):
        modelArgs=self.getModelArgs()
        self.model = Model(**modelArgs)
        
    def getModelArgs(self):
        modelArgs={
            'pathModel':self.pathModel,
            'architectureName':self.dropdownArchitecture.value,
            'architecture':self.getChosenArchitecture(),
            'Dataset':self.dataset,
            'NWrapper':self.NWrapper,
            'wrapper':self.CONFIG_DATA['WRAPPERS']
        }
        return modelArgs
        
    def startAttack(self):        
        attackArgs=self.getAttackArgs()
        if attackArgs['allAttackParams']['imageSelection']=='Dataset':
            attack = Attack(**attackArgs)
            attack.executeAttack()
        elif attackArgs['allAttackParams']['imageSelection']=='Uploaded':
            selectedImagesAndLabels=self.selectionViewer.getSelection()
            attackArgs['allAttackParams']['batchSize']=len(selectedImagesAndLabels['images'])
            resizedImages=self.imgManager.resizeImages(selectedImagesAndLabels['images'], self.dataset.img_size)
            selectedImagesAndLabels['images'] = resizedImages
            attackArgs['uploadedImagesAndLabels']=selectedImagesAndLabels
            attack = Attack(**attackArgs)
            attack.executeAttack()
            
    def getAttackArgs(self): 
        attackArgs={
            'logger':self.logger,
            'Dataset':self.dataset,
            'attackType':self.getChosenAttack(),
            'allAttackParams':self.evaluateAttackParameterInput(),
            'imgManager':self.imgManager,
            'output':self.output,
            'uploadedImagesAndLabels':None
        }
        return attackArgs
    
    def startVC(self):
        attackArgs = self.getAttackArgs()
        epsilonSteps = self.epsilonSteps.value
        
        if attackArgs['allAttackParams']['imageSelection']=='Dataset':
            visualCV = VisualCounterfactuals(epsilonSteps = epsilonSteps, **attackArgs)
            visualCV.generateVC()
        elif attackArgs['allAttackParams']['imageSelection']=='Uploaded':
            selectedImagesAndLabels=self.selectionViewer.getSelection()
            resizedImages=self.imgManager.resizeImages(selectedImagesAndLabels['images'], self.dataset.img_size)
            selectedImagesAndLabels['images'] = resizedImages
            attackArgs['uploadedImagesAndLabels']=selectedImagesAndLabels
            visualCV = VisualCounterfactuals(epsilonSteps = epsilonSteps, **attackArgs)
            visualCV.generateVC()

    
    def evaluateAttackParameterInput(self):
        allAttackParams= {}
        for parameter in self.attackParameters:
            allAttackParams[parameter]=self.attackParameters[parameter].value
        allAttackParams['model_name']=self.model.name
        allAttackParams['model']=self.model.model
        allAttackParams['num_classes']=self.dataset.numberOfClasses
        return allAttackParams
        
    def getChosenArchitecture(self):
        return self.architectures[self.dropdownArchitecture.value]
    
    def getChosenAttack(self):
        selectedAttack=self.attackSelection.selected_index
        selectedAttack=self.attackSelection.get_title(selectedAttack)
        selectedAttackCallback=self.attacks[selectedAttack]
        return selectedAttackCallback
    
    def makeMenu(self):
        commandButtons=self.layout.getCommandButtonsLayout(**self.getCommandButtonElements())
        menuLayout=self.layout.getMainMenuLayout(self.accordionMenu, self.attackSelection,
                                                 self.output.selectionOutput, commandButtons)
          
        return menuLayout
    
    def getCommandButtonElements(self):
        args={
            'computeButton':self.btn_compute,
            'vcButton':self.btn_vc,
            'clearOutputButton':self.btn_clearOut,
            'epsilonSteps':self.epsilonSteps,
            'logger':self.loggerOutput
        }
        return args
    
    def createAttackSelectionMenu(self):
            attackChoicesLayout=[
                self.layout.getFGMLayout(self.attackParameters),
                self.layout.getPGDLayout(self.attackParameters),
                self.layout.getPGDLayout(self.attackParameters),
                self.layout.getPGDLayout(self.attackParameters),
                HBox([self.attackParameters['imageSelection'],self.attackParameters['batchSize']])]
            attackSelection=Tab()
            i=0
            for attack in self.attacks:
                attackSelection.set_title(i, attack)
                i+=1
            attackSelection.children = attackChoicesLayout
            return attackSelection
        
    def createAccordionMenu(self):
        modelMenuLayout = self.layout.getModelMenuLayout(
            self.dropdownDataset, self.dropdownArchitecture,
            self.dropdownModel, self.loggerOutput, self.NWrapper, self.output.descriptionOutput)
        imageUploadLayout = self.layout.getImageUploadLayout(
            self.uploaderImage, self.loggerOutput, self.output.previewOutput)
        modelUploadLayout = self.layout.getModelUploadLayout(
            self.uploaderModel, self.uploadDatasetDropdown, self.uploadArchitecturesDropdown, self.loggerOutput)
        menuOptions = [modelMenuLayout, imageUploadLayout, modelUploadLayout]
        titles = ['Model Selection', 'Image Upload', 'Model Upload']
        return self.layout.getAccordionMenu(titles, menuOptions)
        
    def startImageViewer(self, path, viewer):
        allFiles = self.sysIn.getFiles(path)
        allFiles = [os.path.join(path, file) for file in allFiles]
        imageNames = self.sysIn.filterforEndings(allFiles,['.jpg','.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
        if viewer == 'imageViewer':
            self.imageViewer = ImageViewer(imageNames, self.output.previewOutput)
        elif viewer == 'selectionViewer':
            self.selectionViewer = SelectionViewer(imageNames, self.dataset.classes, self.output.selectionOutput)
            
    
    #============Event-Handlers of the Widgets====================

    def eventHandlerAccordionMenu(self, changes):
        imageUploadMenuId = 1
        currentMenuId = changes['new']
        if not currentMenuId == imageUploadMenuId:
            self.imageViewer = None
            self.output.previewOutput.clear_output()
    
    def eventHandlerDropdownDataset(self,changes):
        try:
            self.dropdownArchitecture.disabled=False
            self.dropdownModel.disabled=False
            currentDataset = changes['new']
            self.pathDataset = os.path.join(self.pathData, currentDataset)
            locationOfDatasetImages=os.path.join(self.locationOfDatasetImages, currentDataset)
            if currentDataset not in self.datasets:
                self.printWrongDatasetInfo()
                raise Exception()
            else:
                self.dataset = self.datasets[currentDataset](currentDataset, locationOfDatasetImages)
                self.updateDropdownArchitectureOptions()
        except:
            self.dropdownArchitecture.disabled = True
            self.dropdownModel.disabled = True
            
            
    def eventHandlerDropdownArchitecture(self,changes):
        try:
            self.dropdownModel.disabled = False
            currentArchitecture = changes['new']
            self.pathArchitecture = os.path.join(self.pathDataset,currentArchitecture)
            self.updateDropdownModelOptions()
            if currentArchitecture not in self.architectures:
                self.printWrongArchitectureInfo()
                raise Exception()
        except:
            self.dropdownModel.disabled = True
           
    def eventHandlerDropdownModel(self,changes):
        currentModel = changes['new']
        if not currentModel == None:
            self.updatePathModel(currentModel)
            self.output.printModelDescription(self.pathModel)
            
    def eventHandlerUploaderImage(self, changes):
        uploadedImages = changes.new.values()
        path = os.path.join(self.notebookDir,self.uploadImagesFolder)
        self.sysIn.saveUploadedImages(uploadedImages, path)
        self.startImageViewer(path, 'imageViewer')  

    def eventHandlerUploaderModel(self, changes):
        uploads = changes.new.values()
        dataset= self.uploadDatasetDropdown.value
        path=os.path.join(self.pathData, dataset)
        self.sysIn.makeDir(path)
        architecture=self.uploadArchitecturesDropdown.value
        path=os.path.join(path, architecture)
        self.sysIn.makeDir(path)
        self.logger.printInfo(path)
        self.sysIn.saveUploadedModel(uploads, path)
        self.initializeDropdown()  
        
    def eventHandlerTargeted(self, changes):
        targeted = changes['new']
        imagesFromDataset = self.attackParameters['imageSelection'].value=='Dataset'
        if targeted: #and imagesFromDataset:
            try:
                self.attackParameters['target_label'].disabled = False
                labels = self.dataset.classes
                self.attackParameters['target_label'].options = labels
            except:
                self.attackParameters['target_label'].disabled = True
                self.attackParameters['targeted'].value = False
                self.logger.printInfo("Could not find labels. Is Dataset selected?")
        else:
            self.attackParameters['target_label'].disabled = True     
        
    def eventHandlerImageSelection(self, changes):
        selected=changes['new']
        if selected=='Dataset':
            self.eventHandlerTargeted({'new':self.attackParameters['targeted'].value})
            self.attackParameters['batchSize'].disabled=False
            self.selectionViewer=None
            self.output.selectionOutput.clear_output()
        elif selected=='Uploaded':
           # self.attackParameters['target_label'].disabled=True
            self.attackParameters['batchSize'].disabled=True
            path = os.path.join(self.notebookDir,self.uploadImagesFolder)
            self.startImageViewer(path, 'selectionViewer')
  
    def eventHandlerBtnCompute(self,changes):
        try:
            self.loadModel()
            self.startAttack()
        except:
            self.logger.printError('')
        
    def event_handler_btn_clearOut(self,changes):
        clear_output()
        self.displayGUI()
    
    def event_handler_btn_vc(self,changes):
        try:
            self.loadModel()
            self.startVC() 
        except ValueError:
            
            self.logger.printInfo('No missclassified Images found')
    
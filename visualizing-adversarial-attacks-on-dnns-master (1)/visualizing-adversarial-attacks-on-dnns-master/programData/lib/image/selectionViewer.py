from PIL import Image
from ipywidgets import Button, Checkbox, Text, Select, Box, Layout, HTML, GridBox, Output
from IPython.display import display, clear_output
from .uploadImage import UploadImage

class SelectionViewer():
    def __init__(self, listOfImageNames, datasetClasses, selectionOutput):
        self.listOfImageNames=listOfImageNames
        self.selectionOutput=selectionOutput
        self.imageList=[]
        self.imagePreviewList=[] 
        self.datasetClasses=['']+datasetClasses
        self.labelSelect=Select(options=self.datasetClasses, disabled=True)
        self.currentId=0
        self.makeImageList()
        self.makePreviewList()
        self.display()
        self.observe()
        
    def makeImageList(self):
        for imageName in self.listOfImageNames:
            img=UploadImage(imageName)
            self.imageList.append(img)

    #uses the 'order' attribute of widget-layout to assign an ID (imgId); 
    #the ID is used to identify on which Image an interaction is reported, 
    #thus allowing for only one eventhandler being responsible for all Images
    def makePreviewList(self):
        self.imagePreviewList=[]
        imgId=0
        for image in self.imageList:
            selectedCB=Checkbox(description='Select for attack')
            chooseLabel=Button(description='Choose Label')
            label=Text(placeholder='Type or choose label')
            preview=image.getPreview()
            preview.layout=Layout(grid_area='preview')
            selectedCB.layout=Layout(width='auto', grid_area='selectedCB', order=str(imgId))
            chooseLabel.layout=Layout(width='auto', grid_area='chooseLabel', order=str(imgId))
            label.layout=Layout(width='auto', grid_area='label', order=str(imgId))
    
            elementLayout=GridBox(children=[preview, selectedCB, chooseLabel, label],
                       layout=Layout(
                              min_width='224px',
                              grid_template_rows='auto',
                              grid_template_columns='100%',
                              margin='10px',
                              grid_template_areas='''
                              "preview"
                              "selectedCB"
                              "chooseLabel"
                              "label"
                              '''))

            self.imagePreviewList.append(elementLayout)
            imgId+=1
            
    def observe(self):
        for preview in self.imagePreviewList:
            chooseLabel=preview.children[2]
            chooseLabel.on_click(self.chooseLabelButtonHandler)
        self.labelSelect.observe(self.eventHandlerLabelSelect, 'value')
        
    def eventHandlerLabelSelect(self, changes):
        selected=changes['new']
        if not self.labelSelect.disabled:
            currentLabel=self.imagePreviewList[self.currentId].children[3]
            currentLabel.value=selected
        self.labelSelect.disabled=True
        self.labelSelect.index=0
        
    def chooseLabelButtonHandler(self, changes):
        imgId=int(changes.layout.order)
        self.currentId=imgId
        self.labelSelect.disabled=False
        
    def getSelection(self):
        listOfSelectedImages=[]
        listOfLabels=[]
        for i in range(len(self.imagePreviewList)):
            selectedCB=self.imagePreviewList[i].children[1].value
            if selectedCB:
                image=self.imageList[i].Image
                label=self.imagePreviewList[i].children[3].value
                listOfSelectedImages.append(image)
                listOfLabels.append(label)
        selection={'images':listOfSelectedImages, 'labels':listOfLabels}
        return selection
                
        
    def getLayout(self):
        box_layout = Layout(overflow='scroll hidden',
                        width='100%',
                        height='',
                        flex_flow='row',
                        display='flex',
                        grid_area='carousel')
        carousel = Box(children=self.imagePreviewList, layout=box_layout)
        labelDescription=HTML(value='<h4>Labels</h4>', layout=Layout(width='auto', grid_area='labelDescription'))
        self.labelSelect.layout=Layout(width='auto',height='99%', grid_area='labelSelect')
        
        ViewerLayout=GridBox(children=[carousel, labelDescription, self.labelSelect],
                       layout=Layout(
                              width='100%',
                              grid_template_rows='40px 400px',
                              grid_template_columns='83% 15%',
                              grid_gap='0px 2%',
                              #margin='10px',
                              grid_template_areas='''
                              "carousel labelDescription"
                              "carousel labelSelect"
                              '''))      
        return ViewerLayout
            
    def display(self):
        self.selectionOutput.clear_output()
        layout=self.getLayout()
        with self.selectionOutput:
            display(layout)
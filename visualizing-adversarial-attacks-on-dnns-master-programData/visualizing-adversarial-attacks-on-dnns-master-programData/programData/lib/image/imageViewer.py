from PIL import Image
from ipywidgets import Box, VBox, Layout, Output
from IPython.display import display, clear_output
from .uploadImage import UploadImage

class ImageViewer():
    def __init__(self, listOfImagePaths, previewOutput):
        self.listOfImagePaths=listOfImagePaths
        self.previewOutput=previewOutput
        self.imageList=[]
        self.imagePreviewList=[]
        self.makeImageList()
        self.makePreviewList()
        self.display()
        
    def makeImageList(self):
        for imageName in self.listOfImagePaths:
            img=UploadImage(imageName)
            self.imageList.append(img)

    def makePreviewList(self):
        self.imagePreviewList=[]
        for image in self.imageList:
            self.imagePreviewList.append(VBox([image.getPreview()],
                                              layout=Layout(min_width='224px', margin='10px')))
            
    def getLayout(self):
        box_layout = Layout(overflow='scroll hidden',
                        width='100%',
                        height='',
                        flex_flow='row',
                        display='flex')
        carousel = Box(children=self.imagePreviewList, layout=box_layout)
        return carousel
            
    def display(self):
        self.previewOutput.clear_output()
        layout=self.getLayout()
        with self.previewOutput:
            display(layout)
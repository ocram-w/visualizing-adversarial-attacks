from PIL import Image
from ipywidgets import Output, Layout
from IPython.display import display, clear_output

class UploadImage():
    def __init__(self, imagePath):
        self.imagePath=imagePath
        self.imageName=self.getImageName()
        self.Image=None
        self.imagePreviewOutput=Output(layout=Layout(min_width='224px'))
        self.openImage()
        
    def getImageName(self):
        splitPath=self.imagePath.rsplit('/',1)
        return splitPath.pop()
      
    def getImagePreview(self):
        return self.Image.resize((224,224), resample=Image.BOX)
    
    def getImageSize(self):
        return self.Image.size
    
    def resizeImage(self):
        self.Image=self.Image.resize((32,32), resample=Image.BOX)
        
    def saveImage(self):
        self.Image.save(self.imagePath)
        self.Image.close()
    
    def openImage(self):
        self.Image=Image.open(self.imagePath)
    
    def generatePreview(self):
        self.imagePreviewOutput.clear_output()
        imagePreview=self.getImagePreview()
        imageSize=self.getImageSize()
        with self.imagePreviewOutput:
            display(imagePreview)
            print(self.imageName)
            print('Image Size: '+ str(imageSize))
        
    def getPreview(self):
        self.generatePreview()
        return self.imagePreviewOutput

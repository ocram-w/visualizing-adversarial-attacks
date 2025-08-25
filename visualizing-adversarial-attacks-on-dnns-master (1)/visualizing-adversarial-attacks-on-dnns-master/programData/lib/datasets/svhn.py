import torch
from torchvision import datasets, transforms

class SVHN():
    def __init__(self, name, locationOfDatasetImages):
        self.datasetArgs={
            'root': locationOfDatasetImages,
            'split':'train',
            'transform':transforms.Compose([transforms.ToTensor()]),
            'target_transform':None,
            'download':False
        }
        self.name=name
        self.img_size =(32, 32)
        self.dataset=None
        self.classes=None
        self.loadDataset()
        self.assignClasses()
        self.numberOfClasses=len(self.classes)
        
    def loadDataset(self):
        self.dataset=datasets.SVHN(**self.datasetArgs)
        
    def assignClasses(self):
        self.classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
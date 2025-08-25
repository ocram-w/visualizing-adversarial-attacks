import torch
from torchvision import datasets, transforms

class Cifar10():
    def __init__(self, name, locationOfDatasetImages):
        self.datasetArgs={
            'root':locationOfDatasetImages,
            'train':True,
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
        self.dataset=datasets.CIFAR10(**self.datasetArgs)
        
    def assignClasses(self):
        self.classes=self.dataset.classes
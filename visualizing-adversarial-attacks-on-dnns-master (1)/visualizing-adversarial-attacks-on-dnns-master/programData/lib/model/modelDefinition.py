import torch

class Model():
    def __init__(self, pathModel, architectureName, architecture, Dataset, NWrapper, wrapper):
        self.pathModel = pathModel 
        self.architectureName = architectureName
        self.architecture = architecture
        self.numberOfClasses = Dataset.numberOfClasses
        self.datasetName = Dataset.name
        self.NWrapper = NWrapper
        self.wrapperdict=wrapper
        self.name = None
        self.model = None
        self.modelLoader()       
        
    def modelLoader(self):
            
    # use the constructor specified by the architecure
        model = self.architecture(self.numberOfClasses)
        model.load_state_dict(torch.load(self.pathModel, map_location='cuda:0'))
        
        if(self.NWrapper.value):
            model = self.wrapperdict[self.datasetName](model)
        
        model.eval()
        self.name = self.architectureName + "_" + self.datasetName
        self.model = model
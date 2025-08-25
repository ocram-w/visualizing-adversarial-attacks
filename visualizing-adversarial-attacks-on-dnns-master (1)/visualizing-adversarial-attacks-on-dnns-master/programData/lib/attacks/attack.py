import torch
import torch.nn as nn
from torchvision import transforms
import inspect

import pandas as pd
from IPython.display import display, clear_output

class Attack():

    def __init__(self, logger, Dataset, attackType, allAttackParams, imgManager, output, uploadedImagesAndLabels = None):
        self.logger=logger
        self.Dataset = Dataset
        self.attackType = attackType
        self.allAttackParams = allAttackParams
        self.uploadedImagesAndLabels = uploadedImagesAndLabels
        self.imgManager = imgManager
        self.output = output       
        
    def executeAttack(self):
    #loads images from chosen dataset and executes attack with chosen attack params 
    #then prints out the perturbed images  
        originalImages, labels = self.getImagesAndLabels(self.allAttackParams['batchSize'])
        targetedLabels = None
        outList = None
        
        if (self.allAttackParams['targeted']):
            targetedLabels = self.getTensorforLabel()
        
        Attack = self.createAttack()
        
        perturbedImages = self.attackOnImages(originalImages, labels, targetedLabels, Attack, self.allAttackParams['targeted'])
        if (self.allAttackParams['save_trajectory']):
            
            trajectoryTensors = Attack.get_last_trajectory()
            listOfProbabilities = self.calculateAllProbabilities(trajectoryTensors)
            trajectoryImages = self.getTrajectoryImages(trajectoryTensors)
            
            listOfPlotProbabilities=self.calculateAllProbabilities(trajectoryTensors)
            probabilitiesToString=self.calculateAllProbabilities2(trajectoryTensors)
            
            outList = self.output.makeOutputList(trajectoryImages, probabilitiesToString, listOfPlotProbabilities)
        self.printOutResults(originalImages, labels, perturbedImages, outList)
        
    def attackOnImages(self, images, labels, targetedLabels, Attack, booltargeted=False):
        if (targetedLabels is None):
            perturbedImages = Attack.perturb(images, labels, targeted=booltargeted)
        else:
            perturbedImages = Attack.perturb(images, targetedLabels, targeted=booltargeted)
        return perturbedImages
    
    def calculateAllProbabilities(self, trajectoryTensor):
        model=self.allAttackParams['model']
        batchList=[]
        for batch in trajectoryTensor:
            probList=[]
            for iteration in batch:
                probabilities=self.returnLogitsAndProbabilities(iteration)
                probList.append(probabilities)
            batchList.append(probList)
        return batchList

    def probabilitiesToString(self, listOfProbabilities):
        for batch in listOfProbabilities:
            batch=[str(self.probabilitiesToDataFrame(prob)) for prob in batch]
            print(type(batch[0]))
        return listOfProbabilities
    
    def calculateAllProbabilities2(self, trajectoryTensor):
        model=self.allAttackParams['model']
        batchList=[]
        for batch in trajectoryTensor:
            probList=[]
            for iteration in batch:
                probabilities=self.returnLogitsAndProbabilities(iteration)
                probabilities=self.probabilitiesToDataFrame(probabilities)
                probList.append(str(probabilities))
            batchList.append(probList)
        return batchList

    def printOutResults(self, originalImages, labels, perturbedImages, outList = None):
    #prints out perturbed images in comparison to original images
        for i in range(len(perturbedImages)):
            orgProb = self.probabilitiesToDataFrame(self.returnLogitsAndProbabilities(originalImages[i]))
            altProb = self.probabilitiesToDataFrame(self.returnLogitsAndProbabilities(perturbedImages[i]))
            orgImg = self.imgManager.transformToPIL(originalImages[i])
            altImg = self.imgManager.transformToPIL(perturbedImages[i])
            imgDiff = self.imgManager.imageDifference(orgImg, altImg)
            imgMonDiff = self.imgManager.inverseMonochrome(imgDiff)
            newImg = self.imgManager.mergeHorizontally(orgImg, altImg)
            newImg = self.imgManager.mergeHorizontally(newImg, imgDiff)
            newImg = self.imgManager.mergeHorizontally(newImg, imgMonDiff)
            
            print("\n")
            print("Image " + str(i + 1) + " = " + self.Dataset.classes[labels[i]])
            display(newImg)
            print(self.allAttackParams['model_name'])
            print('probabilities before attack:    probabilities after attack:')
            gap = [" "] * len(orgProb.index)
            dfgap = pd.DataFrame({' ': pd.Series(gap)})
            print(pd.concat([orgProb, dfgap, altProb], axis=1))
            
            if(self.allAttackParams["save_trajectory"]):
                #with output:
                #print(outList)
                #for i in outList:
                    #i.display()
                outList[i].display()    

    def returnLogitsAndProbabilities(self, inputBatch):
        # takes an image as input, feeds it to the model
        # and returns the probabilities and logits for given input batch in a list "out"
        inputBatch = inputBatch.unsqueeze(0)
        model=self.allAttackParams['model']
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            inputBatch = inputBatch.to('cuda')
            model.to('cuda')
        with torch.no_grad():
            output = model(inputBatch)
        output_logits = output[0]
        output_prob = nn.functional.softmax(output[0], dim=0)  # logits to probabilities
        out = [None] * len(output[0])

        # the probability, logit and name are saved for each class
        for i in range(len(output[0])):
            out[i] = [output_logits[i].item(), output_prob[i].item(), self.Dataset.classes[i]]
        # sort "out", beginning with the highest score
        out.sort(reverse=True)
        return out


    def getTrajectoryImages(self, trajectoryTensors):
        batchSize=self.allAttackParams['batchSize']
        iterations=self.allAttackParams['iterations']
        trajectoryImages = self.imgManager.transformTrajectoryTensorsToImages(trajectoryTensors, batchSize, iterations)
        return trajectoryImages
    
    def getImagesAndLabels(self, batchSize):
        if self.uploadedImagesAndLabels is None:
            dataLoader = self.getDataLoader(batchSize)
            dataiter = iter(dataLoader)
            originalImages, labels = dataiter.next()
        else:
            originalImagesList = self.uploadedImagesAndLabels['images']
            originalImages = self.getTensorsforImageList(originalImagesList)
            labels = self.getTensorforLabelList(self.uploadedImagesAndLabels['labels'])
        return originalImages, labels

    def getDataLoader(self, batchSize):
    #loads chosen number of images from chosen dataset
        dataLoader = torch.utils.data.DataLoader(self.Dataset.dataset,  batchSize,
                                                 shuffle=True, num_workers=6)
        return dataLoader


# ====================== Tensor =====================================

    
    
    def getTensorforLabel(self):
    # creates tensor with label index for a label(string) of the chosen dataset
        dic = self.getLabelTensorDic() #dic = self.getLabelTensorDic(self.Dataset)
        a = []
        for i in range(self.allAttackParams['batchSize']):
            a.append(dic[self.allAttackParams['target_label']])
        t = torch.tensor(a)
        # t = t.to('cuda')
        return t

  
    def getTensorforLabelList(self, labels):
        dic = self.getLabelTensorDic() #dic = getLabelTensorDic(self.Dataset)
        # labels = self.uploadedImagesAndLabels['labels']
        a = []
        for i in range(len(labels)):
            a.append(dic[labels[i]])
        t = torch.tensor(a)
        return t

    def getLabelTensorDic(self):
        dic = {}
        for i in range(len(self.Dataset.classes)):
            dic.update({self.Dataset.classes[i]: i})
        return dic

    def getTensorsforImageList(self, originalImagelist):
    #creates tensor with image tensosr out of image list
        transform = transforms.Compose([transforms.ToTensor()])
        a = []
        for i in range(len(originalImagelist)):
            a.append(transform(originalImagelist[i]))
        a[0] = torch.unsqueeze(a[0], 0)
        for j in range(1, len(a), 1):
            us = torch.unsqueeze(a[j], 0)
            a[0] = torch.cat((a[0], us))
        return a[0]

    

   
 # ========================= attack creation =================

    def createAttack(self):
        necessaryAttackParams = self.getNecessaryAttackParams()
        attackParams = self.pickNecessaryAttackParamsFromAllParams(necessaryAttackParams)
        Attack = self.attackType(**attackParams) 
        return Attack

    def getNecessaryAttackParams(self):
        return inspect.signature(self.attackType).parameters

    def pickNecessaryAttackParamsFromAllParams(self,necessaryAttackParams):
        return {param: self.allAttackParams[param] for param in self.allAttackParams
                if param in necessaryAttackParams}
    

# ============================= calculate Probabolities ======================

    #turns output of returnLogitsAndProbabilities method into dataframe  
    def probabilitiesToDataFrame(self, out):     
        probabilites, logits, classes=self.getHighestThreeStats(out)
        dataFrame = pd.DataFrame({
        'Probability':pd.Series(probabilites),
        'Logit' : pd.Series(logits),
        'Class' : pd.Series(classes)
        })
        return dataFrame
    
    def getHighestThreeStats(self, out):
        probabilites =[]
        logits=[]
        classes=[]
        for i in range(3):
            probabilites.append(str( "%.4f" % round(out[i][1] * 100, 4)))
            logits.append(str(round(out[i][0], 4)))
            classes.append(str(out[i][2]))
        return probabilites, logits, classes
        

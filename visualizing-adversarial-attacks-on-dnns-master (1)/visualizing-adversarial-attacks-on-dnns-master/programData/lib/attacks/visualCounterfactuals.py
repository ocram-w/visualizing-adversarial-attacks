from .attack import Attack
import torch
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

class VisualCounterfactuals(Attack):
    def __init__(self, logger, Dataset, attackType, allAttackParams, imgManager, output, epsilonSteps, uploadedImagesAndLabels = None):
        super().__init__(logger, Dataset, attackType, allAttackParams, imgManager, output,
                         uploadedImagesAndLabels = uploadedImagesAndLabels)
        self.epsilonSteps = epsilonSteps
        self.radii = self.createRadii()
        self.getMisclassifiedImages = self.getMisclassifiedImages()     
        self.modelFailureExamples = self.getMisclassifiedImages[0]
        self.modelFailureTargets = self.getMisclassifiedImages[1]
        
    def generateVC(self):
    #generates visual counterfactuals and prints them out
        numExamples = self.modelFailureExamples.shape[0]
        #container for images 
        #confidence of images gets maximized for the ground truth class and for the false class
        
        gtConfImgs = torch.zeros((len(self.radii),) + self.modelFailureExamples.shape)
        falseConfImgs = torch.zeros_like(gtConfImgs)
   

        batchStartIdx =  0
        batchEndIdx = numExamples

        batchData = self.modelFailureExamples[batchStartIdx:batchEndIdx, :].to('cuda')
        batchGTTargets = self.modelFailureTargets[batchStartIdx:batchEndIdx].to('cuda')


        #saves probabilities for each image
        batchFailureProbs =[None for i in range(len(batchData))]
        #saves the most probable class for each image before the attack is executed
        batchFailurePredictions = [None for i in range(len(batchData))]
        for i in range(numExamples):
            
            predictions=self.returnLogitsAndProbabilities(batchData[i])
            batchFailurePredictions[i] = predictions[0][2]
            batchFailureProbs[i]=predictions

        batchFailurePredictions = self.getTensorforLabelList(batchFailurePredictions)
        batchFailurePredictions = batchFailurePredictions.to('cuda')

        #saves propabilites for ground truth class for each attack radius
        radiProbsGT=[[None for r in range(len(batchData))] for  i in range(len(self.radii))] 
        #saves propabilites for false class for each attack radius
        radiProbsFalse=[[None for r in range(len(batchData))] for  i in range(len(self.radii))] 

        for radiusIdx in range(len(self.radii)):
            eps = self.radii[radiusIdx]
            self.allAttackParams['eps'] = eps

            Attack = self.createAttack()

            gtExs = Attack.perturb(batchData, batchGTTargets, targeted=True).detach()
            falseExs = Attack.perturb(batchData, batchFailurePredictions, targeted=True).detach()

            for k in range(len(batchData)):
                probsGT=self.returnLogitsAndProbabilities(gtExs[k])
                probsFalse=self.returnLogitsAndProbabilities(falseExs[k])
                radiProbsGT[radiusIdx][k]=probsGT
                radiProbsFalse[radiusIdx][k]=probsFalse

            gtConfImgs[radiusIdx, batchStartIdx:batchEndIdx, :] = gtExs.cpu().detach()
            falseConfImgs[radiusIdx, batchStartIdx:batchEndIdx, :] = falseExs.cpu().detach()

        #visual counterfactuals are printed 101 011 122  0=nur in 1 1=beide 2=nur2
        self.printVC(gtConfImgs, batchFailureProbs, radiProbsGT, batchFailurePredictions, falseConfImgs, radiProbsFalse)
   
    def createRadii(self):
        try:
            if self.epsilonSteps < 1:
                raise Exception()
            else:
                radii = [None for i in range(self.epsilonSteps)]
                step = self.allAttackParams['eps']/self.epsilonSteps
                for i in range(self.epsilonSteps):
                    radii[i] = (i+1) * step          
                return radii 
        except:
            self.logger.printInfo('please chose stepsize > 0')
        
    def getMisclassifiedImages(self):
    # searches for chosen number of batches different misclassified images in chosen Dataset
    #returns list with a tensor of misclassified images and a tensor of the matching image labels
        batchSize=self.allAttackParams['batchSize']
        ImagesLabels=[]
        counter=0
        originalImages, labels = self.getImagesAndLabels(len(self.Dataset.dataset))
        originalImages, labels = originalImages.to('cuda'), labels.to('cuda')
        misclassifiedImages = torch.zeros_like(originalImages[0:batchSize])

        for i in range(len(originalImages)):
            pic_prediction = self.returnLogitsAndProbabilities(originalImages[i])[0][2]
            if pic_prediction != self.Dataset.classes[labels[i]]:
                misclassifiedImages[counter]=originalImages[i]
                ImagesLabels.append(self.Dataset.classes[labels[i]])
                counter = counter +1
            if counter == batchSize:
                break
        labelTensor = self.getTensorforLabelList(ImagesLabels)  
        return [misclassifiedImages,labelTensor]

   
    def getProbForLabel(self, label, ProbsList):
    #gets probability for label(string)
        for list in ProbsList:
            if label in list:
                return list[1]
        return 0      


    def printVC(self, gtConfImgs, batchFailureProbs, radiProbsGT, batchFailurePredictions, falseConfImgs, radiProbsFalse):
       #loop iterates over different images
        for i in range(gtConfImgs.shape[1]):
            newImgGT = self.imgManager.transformToPIL(self.modelFailureExamples[i].cpu())
            newImgBlank =  self.imgManager.transformToPIL(torch.ones(3, 32, 32))
            ImageDraw.Draw(newImgBlank).text((15, 160),'ground truth = ' + 
                                             str(self.Dataset.classes[self.modelFailureTargets[i]])+'\n'+
                                             str(self.Dataset.classes[self.modelFailureTargets[i]])+ ' = '
                                             + str("%.2f" % round(self.getProbForLabel(self.Dataset.classes[
                                                 self.modelFailureTargets[i]],batchFailureProbs[i])*100,2))+ '\n'+
                                             batchFailureProbs[i][0][2]+ ' = '
                                             + str("%.2f" % round(batchFailureProbs[i][0][1]*100,2))
                                             ,(0, 0, 0),font = ImageFont.truetype("arial.ttf",20))
            
            newImgGT = self.imgManager.mergeVertically(newImgGT, newImgBlank)
            
            newImgBlank =  self.imgManager.transformToPIL(torch.ones(3, 32, 32))
            newImgFalse = self.imgManager.mergeVertically(newImgBlank, newImgBlank)
            
            #loop iterates over different attack radii
            for j in range(gtConfImgs.shape[0]):
                tempImgGT = self.imgManager.transformToPIL(gtConfImgs[j][i])
                tempImgGTtext =  self.imgManager.transformToPIL(torch.ones(3, 32, 32))
                ImageDraw.Draw(tempImgGTtext).text((15, 160), 
                                             'eps = ' + str(round(self.radii[j],3)) +'\n' +
                                             str(self.Dataset.classes[self.modelFailureTargets[i]])+ ' = '
                                             + str("%.2f" % round(self.getProbForLabel(
                                                 self.Dataset.classes[self.modelFailureTargets[i]], radiProbsGT[j][i])*100,2))+ '\n'+
                                             self.Dataset.classes[batchFailurePredictions[i]] + ' = '
                                             + str("%.2f" % round(self.getProbForLabel(
                                                 self.Dataset.classes[batchFailurePredictions[i]], radiProbsGT[j][i])*100,2)) ,(0, 0, 0),
                                                   font = ImageFont.truetype("arial.ttf",20))
                tempImgGTmerged = self.imgManager.mergeVertically(tempImgGT, tempImgGTtext)
                newImgGT = self.imgManager.mergeHorizontally(newImgGT, tempImgGTmerged)
                
                tempImgFalse = self.imgManager.transformToPIL(falseConfImgs[j][i])
                tempImgFalseText =  self.imgManager.transformToPIL(torch.ones(3, 32, 32))
                ImageDraw.Draw(tempImgFalseText).text((15, 25),  str(self.Dataset.classes[self.modelFailureTargets[i]])+ ' = '+
                                             str("%.2f" % round(self.getProbForLabel(
                                                 self.Dataset.classes[self.modelFailureTargets[i]], radiProbsFalse[j][i])*100,2))+ '\n'+
                                             self.Dataset.classes[batchFailurePredictions[i]] + ' = '
                                             + str("%.2f" % round(self.getProbForLabel(
                                                 self.Dataset.classes[batchFailurePredictions[i]], radiProbsFalse[j][i])*100,2)) ,(0, 0, 0),
                                                   font = ImageFont.truetype("arial.ttf",20))
                tempImgFalseMerged = self.imgManager.mergeVertically(tempImgFalseText,tempImgFalse)
                newImgFalse = self.imgManager.mergeHorizontally(newImgFalse, tempImgFalseMerged)
            display(newImgGT)
            display(newImgFalse)
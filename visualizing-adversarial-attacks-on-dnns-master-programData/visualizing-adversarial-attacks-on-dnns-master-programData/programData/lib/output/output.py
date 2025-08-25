import ipywidgets as widgets
from IPython.display import display, clear_output
#import matplotlib
import matplotlib.pyplot as plt

from .trajectoryOutput.__init__ import *

class Output():

    def __init__(self, systemInterface):
        self.systemInterface=systemInterface
        self.descriptionOutput = widgets.Output(layout={'border': '1px solid black'})
        self.previewOutput = widgets.Output()
        self.selectionOutput = widgets.Output()
        #self.calculatedElement = calculatedElement

    # Function takes a path to a model and prints the content of a .txt-file
    # with the same name as the model in the descriptionOutput
    def printModelDescription(self, path):                         
        path = self.systemInterface.changeFileEnding(path, ".txt")       
        self.descriptionOutput.clear_output()
        try:
            description = self.systemInterface.getDescriptionFromFile(path)
            with self.descriptionOutput:
                print(description)
        except FileNotFoundError:
            with self.descriptionOutput:
                print("No description-file for this model found")


    def printUploadStatus(self, message):  # hier wurde Parameter self hinzugefügt
        with self.uploadStatusOutput:
            print(message)
            

    def makeOutputList(self, listOfImages, probabilitiesToString, listOfPlotProbabilities):
        outputList=[]
        for i in range(len(listOfImages)):
            probabilities=probabilitiesToString[i]
            p_probabilities=listOfPlotProbabilities[i]
            frames=listOfImages[i]
            numberOfFrames=len(frames)
            plot=self.makePlot(p_probabilities)
            OutputElement=OutputAnimation(numberOfFrames, frames, probabilities, plot)
            outputList.append(OutputElement)
        return outputList

    def makePlot(self, probabilities): 
                #makePlot gets a list of Triples containing (prob, logit, classname) per iteration for one image. makePlot then plots the probabities of the classes with the highest probabilities in the first and last iteration against the iteration.
        nrOfIterations= len(probabilities)-1
        #getting the probabilities of the classes with highest probabilities in the first and last iteration
        firstTriple= probabilities[0][0]
        lastTriple= probabilities[nrOfIterations][0]
        
        firstString = firstTriple[2]
        lastString = lastTriple[2]
        
        probabilitiesFirst = [None] * (nrOfIterations +1)
        probabilitiesLast = [None] * (nrOfIterations +1)
        counterFirst = 0 
        counterLast = 0
        
        for iteration in probabilities:
            for triple in iteration:

                if(triple[2]== firstString) :
                    probabilitiesFirst[counterFirst]= triple[1]
                    counterFirst+=1

                elif(triple[2]== lastString) :
                    probabilitiesLast[counterLast]= triple[1]
                    counterLast+=1

        
        plt.plot(list(range(0,len(probabilitiesFirst))), probabilitiesFirst, label=firstString)
        plt.plot(list(range(0,len(probabilitiesLast))), probabilitiesLast, label=lastString)
        plt.ylabel('p(x)')        
        plt.xlabel('iteration step')
        plt.legend()
        
        return plt

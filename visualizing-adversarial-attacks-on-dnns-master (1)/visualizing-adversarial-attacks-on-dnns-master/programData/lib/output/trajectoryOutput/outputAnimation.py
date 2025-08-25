import matplotlib
from ipywidgets import Output, Textarea, IntSlider, Play, jslink, VBox, HBox, Layout, Image
from IPython.display import display, clear_output

class OutputAnimation:
    def __init__(self, numberOfFrames, frames, probabilities, plot):
        self.numberOfFrames = numberOfFrames
        self.maxSteps=self.numberOfFrames-1
        self.frames=frames
        self.probabilities=probabilities
        self.plot=plot
        self.plotOutput=Output()
        self.textArea=Textarea(placeholder='empty Data', layout=Layout(width='250px', height='300px', border='1px solid black'))
        self.img=Image(
            value=frames[0],
            format='jpg',
            width=300,
            height=300,
        )
        self.IntSlider=IntSlider(min=0, max=self.maxSteps)
        self.Play = Play(min=0,max=self.maxSteps,step=1,interval=400)
        self.jslink=jslink((self.Play,'value'),(self.IntSlider, 'value'))
        
        self.initializeAnimation()
        self.observeIntSlider()
        self.showPlot()
            
    def showPlot(self):
        with self.plotOutput:
            self.plot.show()
        
    def animationHandler(self, changes):
        index=changes['new']
        self.img.value=self.frames[index]
        self.textArea.value=self.probabilities[index]
        
    def observeIntSlider(self):
        self.IntSlider.observe(self.animationHandler, 'value')
        
    def display(self):
        display(VBox([HBox([self.img, self.textArea, self.plotOutput]), self.IntSlider, self.Play]))
        
    def initializeAnimation(self):
        self.animationHandler({'new':self.IntSlider.value})
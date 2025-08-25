from .configData import CONFIG_DATA
from .__init__ import *

#import GUI as GUI

def main():
    
    logger = Logger(CONFIG_DATA)
    sysIn = SystemInterface(logger)
    output = Output(sysIn)    
    layout = GUILayout()
    imgManager=ImageManager()
    gui = GUI(layout, output, sysIn, logger, imgManager, CONFIG_DATA)
    gui.run()

    

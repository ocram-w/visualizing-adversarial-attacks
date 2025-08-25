import logging
import ipywidgets as widgets

class Logger():
    def __init__(self, CONFIG_DATA):
        self.loggerOutput=None
        self.logger=None
        self.initializeLogger()
        self.infoCount=0
        self.maxInfoBeforeCearout=CONFIG_DATA['MAX_INFO_BEFORE_LOGGER_CLEAROUT']
     
    def initializeLogger(self):
        logger = logging.getLogger(__name__)
        handler = OutputWidgetHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] \n %(message)s \n'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        self.logger=logger
        self.loggerOutput=handler.out
        
    def printInfo(self, infoMessage):
        self.clearLoggerIfInfoCountHigh()
        self.logger.info(infoMessage)

    def printError(self, errorMessage):
        self.clearLoggerIfInfoCountHigh()
        self.logger.exception(errorMessage)
        
    def clearLoggerIfInfoCountHigh(self):
        if self.infoCount==self.maxInfoBeforeCearout:
            self.loggerOutput.clear_output()
            self.infoCount=0
        self.infoCount+=1
        
class OutputWidgetHandler(logging.Handler):
    #Custom logging handler sending logs to an output widget 
    #(from https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html)

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': 'auto',
            'height': 'auto',
            'border': '1px solid black',
            'overflow_y':'scroll',
            'grid_area': 'loggerOP'
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        #Overload of logging.Handler method
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        display(self.out)

    def clear_logs(self):
        self.out.clear_output()
        
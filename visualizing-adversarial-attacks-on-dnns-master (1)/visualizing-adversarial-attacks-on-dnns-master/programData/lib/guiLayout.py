from ipywidgets import GridBox, Output, HTML, Accordion, VBox, Layout


class GUILayout():
    def __init__(self):
        pass
       # self.output = output
        
    def getMainMenuLayout(self, accordionMenu, attackSelection, selectionOutput, commandButtons):
        
        accordionMenu.layout.grid_area='accordionMenu'
        attackSelection.layout.grid_area='attackSelection'
        selectionOutput.layout=Layout(grid_area='selectionOP')
        commandButtons.layout.grid_area='commandB'
        
        menuLayout=GridBox(children=[accordionMenu, attackSelection, commandButtons, selectionOutput],
                           layout=Layout(
                                  width='100%',
                                  grid_template_rows='auto',
                                  grid_gap='0px 2%',
                                  grid_template_columns='100%',
                                  grid_template_areas='''
                                  "accordionMenu"
                                  "attackSelection"
                                  "selectionOP"
                                  "commandB"
                                  '''))
        return menuLayout

    def getModelMenuLayout(self, dropdownDataset, dropdownArchitecture, dropdownModel, loggerOutput, NWrapper, descriptionOutput):
        # header=HTML(value='<h3>Model Selection</h3>', layout=Layout(width='auto', grid_area='header'))
        datasetLabel = HTML(value='<h4>Dataset</h4>', layout=Layout(width='auto', grid_area='labelDataset'))
        architectureLabel = HTML(value='<h4>Architecture</h4>',layout=Layout(width='auto', grid_area='labelArchitecture'))
        modelLabel = HTML(value='<h4>Model</h4>', layout=Layout(width='auto', grid_area='labelModel'))
        dropdownDataset.layout = Layout(width='auto', height='40px', grid_area='ddDataset')
        dropdownArchitecture.layout = Layout(width='auto', height='40px', grid_area='ddArchitecture') 
        dropdownModel.layout = Layout(width='auto', height='40px', grid_area='ddModel')
        #loggerOutput.layout = Layout(width='auto', border='1px solid black', overflow_y='scroll', grid_area='loggerOP')
        descriptionOutput.layout = Layout(width='auto', height='auto', border='1px solid black', overflow_y='scroll',
                                          grid_area='descriptionOP')
        NWrapper.layout=Layout(width="auto", height ="auto",grid_area = "NormalizationWrapper")
        menuLayout=GridBox(children=[
                            datasetLabel, architectureLabel, modelLabel,
                            dropdownDataset, dropdownArchitecture, dropdownModel,
                            loggerOutput, descriptionOutput, NWrapper],                       
                        layout=Layout(
                            width='100%',
                            grid_template_rows='50px 50px 50px 40px 300px',
                            grid_template_columns='10% 33% 53%',
                            grid_gap='5px 2%',
                            grid_template_areas='''
                            "labelDataset ddDataset loggerOP"
                            "labelArchitecture ddArchitecture loggerOP "
                            "labelModel ddModel loggerOP"
                            "NormalizationWrapper NormalizationWrapper loggerOP "
                            "descriptionOP descriptionOP loggerOP"
                            '''))
        return menuLayout

    def getPGDLayout(self, attackParameters):    
        #header=HTML(value='<h3>Model Selection</h3>', layout=Layout(width='auto', grid_area='header'))
        #normLabel=HTML(value='<h4>Norm</h4>', layout=Layout(width='auto', grid_area='labelNorm'))
        #epsLabel=HTML(value='<h4>Epsilon</h4>', layout=Layout(width='auto', grid_area='labelEps'))
        child=[]
        for param in attackParameters:
            attackParameters[param].layout=Layout(width='200px', grid_area=param)
            child.append(attackParameters[param])
        #attackParameters['imageSelection'].layout.width='auto'
        #attackParameters['imageSelection'].layout.object_position='right'
        #attackParameters['imageSelection'].style={'description_width': 'initial'}

        menuLayout=GridBox(children=child,
                           layout=Layout(
                                  width='100%',
                                  overflow='hidden',
                                  grid_template_rows='50px 50px 70px 50px 50px',
                                  grid_template_columns='7% 31% 31% 31%',
                                  #grid_gap='5px 2%',
                                  grid_template_areas='''
                                  ". eps early_stopping  restarts"
                                  ". iterations momentum stepsize"
                                  ". norm  loss targeted "
                                  ". init_noise_generator normalize_grad target_label"
                                  ". save_trajectory imageSelection batchSize"
                                  '''))
        return menuLayout
    
    def getFGMLayout(self, attackParameters):    
        for param in attackParameters:
            attackParameters[param].layout=Layout(width='200px', grid_area=param)

        menuLayout=GridBox(children=[
                                attackParameters['norm'],
                                attackParameters['eps'], attackParameters['loss'], attackParameters['restarts'],
                                attackParameters['init_noise_generator'], attackParameters['batchSize'],
                                attackParameters['imageSelection'], attackParameters['targeted'],
                                attackParameters['target_label']],
                           layout=Layout(
                                  width='100%',
                                  overflow='hidden',
                                  grid_template_rows='50px 50px 70px 50px 50px',
                                  grid_template_columns='7% 31% 31% 31%',
                                  #grid_gap='5px 2%',
                                  grid_template_areas='''
                                  ". eps . restarts"
                                  ". . . ."
                                  ". norm  loss targeted "
                                  ". init_noise_generator . target_label"
                                  ". . imageSelection batchSize"
                                  '''))
        return menuLayout

    def getImageUploadLayout(self, uploaderImage, loggerOutput, previewOutput):
        previewOutput.layout = Layout(width='auto', min_height='50px', grid_area='previewOP')
        uploaderImage.layout = Layout(width='200px', grid_area='uploaderImage')

        menuLayout = GridBox(children=[loggerOutput, previewOutput, uploaderImage],
                             layout=Layout(
                                 width='100%',
                                 min_height='520px',
                                 overflow='hidden',
                                 grid_template_rows='80px 80px auto',
                                 grid_template_columns='25% 73%',
                                 grid_gap='2%',
                                 grid_template_areas='''
                                  "uploaderImage loggerOP"
                                  ". loggerOP"
                                  "previewOP previewOP"
                                  '''))
        return menuLayout

    def getModelUploadLayout(self, uploaderModel, dropdownDatasetUpload, dropdownArchitectureUpload, loggerOutput):
        
        header = HTML(value='<h4>Select Destination</h4>', layout=Layout(width='auto', grid_area='header'))
        datasetLabel = HTML(value='<h4>Dataset</h4>', layout=Layout(width='auto', grid_area='labelDataset'))
        architectureLabel = HTML(value='<h4>Architecture</h4>',
                                 layout=Layout(width='auto', grid_area='labelArchitecture'))
        uploaderModel.layout = Layout(width='auto', grid_area='uploaderModel')
        dropdownDatasetUpload.layout = Layout(width='auto', grid_area='uploadeDataset')
        dropdownArchitectureUpload.layout = Layout(width='auto', grid_area='uploadeArchitecture')

        menuLayout = GridBox(children=[
            header, datasetLabel, architectureLabel,
            dropdownDatasetUpload, dropdownArchitectureUpload, uploaderModel, loggerOutput],
            layout=Layout(
                width='100%',
                grid_template_rows='auto 50px 50px 50px',
                grid_template_columns='10% 33% 53%',
                grid_gap='5px 2%',
                grid_template_areas='''
                                "header header header"
                                "labelDataset uploadeDataset loggerOP"
                                "labelArchitecture uploadeArchitecture loggerOP "
                                ". uploaderModel loggerOP"
                                '''))
        return menuLayout

    def getAccordionMenu(self, titles, options):
        header = HTML(value='<h3>Main Menu</h3>', layout=Layout(width='auto'))
        footer = HTML(value='<h3>Attack Selection</h3>', layout=Layout(width='auto'))
        accordion = Accordion(children=options)
        for i in range(len(titles)):
            accordion.set_title(i, titles[i])
        return VBox([header, accordion, footer])
    
    def getCommandButtonsLayout(self, computeButton, vcButton, clearOutputButton, epsilonSteps, logger):
        computeButton.layout=Layout(width='auto', grid_area='computeB')
        vcButton.layout=Layout(width='auto', grid_area='vcB')
        epsilonSteps.layout=Layout(width='auto', grid_area='epsSteps')
        clearOutputButton.layout=Layout(width='auto', grid_area='clearOutB')

        menuLayout = GridBox(children=[computeButton, vcButton, epsilonSteps, clearOutputButton, logger],
            layout=Layout(
                width='100%',
                grid_template_rows='50px 50px 50px 50px',
                grid_template_columns='20% 78%',
                grid_gap='5px 2%',
                grid_template_areas='''
                                "computeB loggerOP"
                                "vcB  loggerOP"
                                "epsSteps loggerOP "
                                "clearOutB loggerOP"
                                '''))
        return menuLayout
        

import sys, os

class SystemInterface():
    def __init__(self, logger):
        self.logger=logger

    # Function takes a directory-name/path as a string (e.g. "./Data" or ".")
    # and returns an alphabetical sorted list of all (non-hidden) subdirs
    def getSubdirs(self, directory):
        dirList = next(os.walk(directory))[1]
        dirList = self.sortOutHiddenFiles(dirList)
        return sorted(dirList, key=str.lower)

    # Function takes a directory-name/path as a string (e.g. "./Data" or ".")
    # and a fileEnding (e.g. ".txt") and returns an alphabetical sorted list
    # of all (non-hidden) files with this ending in the specified directory
    def getFiles(self, directory, fileEnding=''):
        fileList = [file for file in os.listdir(directory) if file.endswith(fileEnding)]
        fileList = self.sortOutHiddenFiles(fileList)
        return sorted(fileList, key=str.lower)

    def sortOutHiddenFiles(self, fileList):
        fileList = [file for file in fileList if not file.startswith('.')]
        return fileList

    def filterForImages(self, fileList):
        validExtensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        fileList = [file for file in fileList if any(file.endswith(ext) for ext in validExtensions)]
        return fileList
    
    def filterforEndings(self, fileList, ExtensionList):
        fileList = [file for file in fileList if any (file.endswith(ext) for ext in ExtensionList)]
        return fileList


    def changeFileEnding(self, path, newFileEnding):
        currentChar = len(path) - 1
        while (path[currentChar] != ".") and (currentChar > 0):
            currentChar -= 1
        path = path[0:currentChar] + newFileEnding
        return path

    def getDescriptionFromFile(self, path):
        with open(path, "r") as descriptionFile:
            description = descriptionFile.read()
        return description

    def saveUploadedImages(self, uploadedImages, folder):
        for elem in uploadedImages:
            filename = elem['metadata']['name']
            self.logger.printInfo(filename)
            path = os.path.join(folder, filename)
            self.writeToFile(path, elem['content'])
            self.logger.printInfo('Done')

    def saveUploadedModel(self, uploads, folder):
        for elem in uploads:
            filename = elem['metadata']['name']
            self.logger.printInfo(filename)
            path = os.path.join(folder, filename)
            path = self.changeFileEnding(path, '')
            
            self.makeDir(path)
            path = os.path.join(path, filename)

            self.writeToFile(path, elem['content'])
            self.logger.printInfo("Done")
            
    def makeDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def writeToFile(self, path, content):
        with open(path, 'wb') as file:
            file.write(content)
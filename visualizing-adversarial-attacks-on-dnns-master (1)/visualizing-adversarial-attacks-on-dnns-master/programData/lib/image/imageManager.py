import io
from PIL import Image, ImageChops, ImageDraw, ImageFont
from torchvision import transforms

class ImageManager():
    def __init__(self):
        pass
        
    #size is tupel e.g. (32,32) 
    def resizeImages(self,listOfPILImages, size):
        resizedImages=[image.resize(size, resample=Image.BOX) for image in listOfPILImages]
        return resizedImages
    
    def transformTrajectoryTensorsToImages(self, trajImg, batchSize, iterations):
        trans = transforms.ToPILImage()
        outerList = []
        for k in range(batchSize):
            innerList = []
            for m in range(iterations):
                transImg = trans(trajImg[k][m])
                transImg = transImg.resize((320, 320), resample=Image.BOX)
                buf = io.BytesIO()
                transImg.save(buf, format='JPEG')
                byteIm = buf.getvalue()
                innerList.append(byteIm)
            outerList.append(innerList)
        return outerList
    
    def transformToPIL(self, img):
        trans = transforms.ToPILImage()
        img = trans(img)
        return img.resize((256,256), resample=Image.BOX)

    def mergeHorizontally(self, img1, img2):
        newImg=Image.new('RGB', (img1.width+img2.width, img1.height))
        newImg.paste(img1,(0,0))
        newImg.paste(img2,(img1.width,0))
        return newImg

    def mergeVertically(self, img2, img1):
        newImg=Image.new('RGB', (img1.width, img1.height + img1.height))
        newImg.paste(img1,(0,0))
        newImg.paste(img2,(0,img2.height))
        return newImg

    #sets all black pixels(0) to white and all other to black
    def inverseMonochrome(self, img):
        monImg=img.convert(mode='L')
        monImg=monImg.point(lambda x: 255 if x==0 else 0, '1')
        return monImg

    def imageDifference(self, img1, img2):
        imgDiff=ImageChops.difference(img1, img2)
        return imgDiff
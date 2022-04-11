import matplotlib
import pandas as pd
import numpy as np
import os 
# from matplotlib.image import imread
# from matplotlib.image import imsave
import matplotlib.pyplot as plt



##### TODO: make a dataframe that has image path and label
# TODO: make class with label attribute?




def createImagePath(rootPath):

    label = {'buildings': 0,
    'forest' :1,
    'glacier' : 2,
    'mountain' :3,
    'sea' :4,
    'street':5 }



    subPath = "/archive/seg_train/seg_train"

    filesWithLabel = []
    for key,val in label.items():
        files = os.listdir(rootPath+subPath+"/"+key) ## file names
        d = []
        for f in files:
            d.append((f,val))
        filesWithLabel = filesWithLabel+d
    df = pd.DataFrame(filesWithLabel,columns=["filename","label"])
    return df

def loadImage(imagePath):
    # TODO: load images into 
    pass


df = createImagePath(os.getcwd())
# class loadImage:
#     def __init__(self, image) -> None:
#         self.image = image


#     def imageToFeatureVec(self,image):
#         pass
    






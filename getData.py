import matplotlib
import pandas as pd
import numpy as np
import os 
# from matplotlib.image import imread
# from matplotlib.image import imsave
import matplotlib.pyplot as plt
from PIL import Image



##### TODO: make a dataframe that has image path and label
# TODO: make class with label attribute?



class getData:
    label  =  {"buildings": 0,"forest":1,"glacier":2,"mountain":3,"sea":4,"street":5}
    subPath = "/archive/seg_train/seg_train"

    def __init__(self) -> None:
        pass

    def createImageDf(rootPath = os.getcwd()):
        """
        This fuction creates a dataframe with filename, label (0,1,2,3,4,5), landscape type,
        and a flatened image as a numpy array... probably more useful in SVM or other Stat learning methods. 

        """

        # assign labels based on the image path
        print("loading...it takes a sec")
        filesWithLabel = []
        for key,val in getData.label.items():
            files = os.listdir(rootPath+getData.subPath+"/"+key) ## file names
            d = []
            for f in files:
                d.append((f,val,key))
            filesWithLabel = filesWithLabel+d
        df = pd.DataFrame(filesWithLabel,columns=["filename","label","landscape"])

        # load image, turn into numpy array, flatten then slap on dataframe
        flatImg = []
        for i in df.index:  
            img = Image.open(rootPath + getData.subPath + "/"+df["landscape"][i]+"/"+ df['filename'][i])
            a = np.asarray(img)
            flatImg.append(a.flatten())
            
        df["flat_arrays"]  = flatImg
        print("done")
        return df

    def loadImage(df,rootPath=os.getcwd()):
        """
        A function to load an image, dispaly image with type of landscape. could be useful later when evaluating model performance.
        """
        for i in df.index:  
            img = Image.open(rootPath + getData.subPath + "/"+df["landscape"][i]+"/"+ df['filename'][i])
            a = np.asarray(img)
            plt.imshow(a)
            plt.title(f"{df.landscape[i]}")
            plt.show()
            


if __name__ == '__main__':
    df = getData.createImageDf()






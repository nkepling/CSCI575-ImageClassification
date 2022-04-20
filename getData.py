import matplotlib
import pandas as pd
import numpy as np
import os 
# from matplotlib.image import imread
# from matplotlib.image import imsave
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path




# TODO: I pretty much hard coded to load training data...need to change that so we can use this for testing.
# TODO: Fix how file paths are handeled -- so that it works with Windows 





class getData:
    label  =  {"buildings": 0,"forest":1,"glacier":2,"mountain":3,"sea":4,"street":5}
    #subPath = "/archive/seg_train/seg_train"
    #subPath  = Path("/archive/seg_train/seg_train")


    def __init__(self) -> None:
        pass

    def createImageDf(rootPath = Path.cwd()):
        """
        This fuction creates a dataframe with filename, label (0,1,2,3,4,5), landscape type,
        and a flatened image as a numpy array... probably more useful in SVM or other Stat learning methods. 

        """

        # assign labels based on the image path
        print("loading...it takes a sec")
        filesWithLabel = []
        for key,val in getData.label.items():
            files = os.listdir(rootPath/ "archive" /"seg_train" /"seg_train"/ key) ## file names
            d = []
            for f in files:
                d.append((f,val,key))
            filesWithLabel = filesWithLabel+d
        df = pd.DataFrame(filesWithLabel,columns=["filename","label","landscape"])

        # load image, turn into numpy array, flatten then slap on dataframe
        flatImg = []
        for i in df.index:  
            img = Image.open(rootPath/ "archive" /"seg_train" /"seg_train"/ df["landscape"][i] / df['filename'][i])
            a = np.asarray(img)
            flatImg.append(a.ravel())
            
        df["flat_arrays"]  = flatImg
        print("done")
        return df

    def dispImage(df,rootPath=Path.cwd()):
        """
        A function to load an image, dispaly image with type of landscape. could be useful later when evaluating model performance.
        """
        for i in df.index:  
            img = Image.open(rootPath/ "archive" /"seg_train" /"seg_train"/ df["landscape"][i] / df['filename'][i])
            a = np.asarray(img)
            plt.imshow(a)
            plt.title(f"{df.landscape[i]}")
            plt.show()
    
    # def loadFeatureMat(path = Path.cwd(),dataset = "train"):
    #     """
    #     Returns NxP feature matrix where N is the number of images and P is the length of the flattened images.
    #     (250 * 250) = 62500 = P
    #     """
    #     d = "seg_"+dataset 

    #     flatImg = []
    #     for key in getData.label.keys():  
    #         files = os.listdir(path/ "archive" / d / d / key)
    #         for file in files:
    #             img = Image.open(path/ "archive" / d / d / key / file)
    #             a = np.asarray(img)
    #             flatImg.append(a.flatten())
    #     return flatImg

        
            


if __name__ == '__main__':
    df = getData.createImageDf()
    # mat = getData.loadFeatureMat()

from sklearn.decomposition import PCA
from getData import getData
import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os 



def featureMat(label = getData.label,path = Path.cwd(),dataset = "train"):
    """
    
    """
    flatImg = []
    for key in label.key():  
        img = Image.open(path/ "archive" /"seg_"+dataset /"seg_"+dataset/ key / df['filename'][i])ls
        a = np.asarray(img)
        flatImg.append(a.flatten())
    pass


def landscapePCA():
    pass


if __name__ == "main":
    pass

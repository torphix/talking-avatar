# from pipeline import DiffusedAvatarPipeline


# DiffusedAvatarPipeline()
import json
path = '/home/j/Desktop/Programming/DeepLearning/multilingual/avatar/DiffHeads/data/vox/processed/video/id06209/6oI-FJQS9V0/00027/00027_1.npy'
import numpy as np

from PIL import Image
Image.fromarray(np.load(path).transpose(1,2,0)).save('out.png')



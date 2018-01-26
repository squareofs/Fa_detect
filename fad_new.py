import cv2
import os
import numpy as np

subjects=["","sagar", "Shashank"]

print("Preparing image data ")

faces, label= training("images_data")




#training function
def traning(folder_path):
    faces=[]
    label=[]


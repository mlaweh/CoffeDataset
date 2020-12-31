import glob
import cv2
import os
from shutil import copyfile

mother_folder = "LeavesDiseases/*.png"
paths = glob.glob(mother_folder)
annot_file = "dataset/annotations/test/"
test_file = "dataset/images/test/"

for root, dirs, files in os.walk(annot_file, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
for root, dirs, files in os.walk(test_file, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
             
os.makedirs(test_file, exist_ok=True)
os.makedirs(annot_file, exist_ok=True)
for i, path in enumerate(paths) :
    copyfile(path,test_file+str(i)+'.jpg')
    copyfile(path, annot_file+str(i) +"_mask.png" )

import numpy as np
import cv2
import dlib
import copy
import matplotlib.pyplot as plt
import pickle
import PresetTriangulation as pre

with open('PresetTriangulation.pickle', 'rb') as handle:
    TriangulationIndexes = pickle.load(handle)
    
nof_triangles=TriangulationIndexes.shape[0]


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""Load Source Image"""
directory='sundayfunday5.png'

source_image_rgb=cv2.imread(directory)
source_image_gray=cv2.imread(directory,0)

"""Find and Extract Source Image Landmarks"""
faces=face_detector(source_image_gray)
source_landmarks = pre.ExtractLandmarks(source_image_gray,faces,landmark_predictor,False)

SourceTriangles,SourceTransform=pre.GetList_CroppedTriangulation(source_image_rgb,\
                                                         source_landmarks,\
                                                         TriangulationIndexes)
    
Framebool=False
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)

while True:
    
    _,dest_image_rgb=cap.read()
    dest_image_gray=cv2.cvtColor(dest_image_rgb, cv2.COLOR_BGR2GRAY)


    frame=copy.copy(dest_image_rgb)
    transfer_face=np.zeros_like(frame)

    """Detect Faces and Estimate Landmarks"""
    faces=face_detector(dest_image_gray)

    """Extract Landmarks"""
    dest_landmarks = pre.ExtractLandmarks(dest_image_gray,faces,landmark_predictor,False)
    
    
    """Triangulate with preset triangles and the extracted landmarks"""
    
    
    DestTriangles,DestTransform=pre.GetList_CroppedTriangulation(dest_image_rgb,\
                                                           dest_landmarks,\
                                                        TriangulationIndexes)
        
    
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:57:26 2021

@author: OKYANUS
"""
import numpy as np
import cv2
import dlib
import copy
import matplotlib.pyplot as plt
import pickle

import PresetTriangulation as pre


"""pre.Indexed_Triangulation(name_of_pickle_file="PresetTriangulation",\
                          source_image_destination="DirectPortrait.jpg",\
                              visual_bool=False)
"""
"""Load Preset Triangulation"""
with open('PresetTriangulation.pickle', 'rb') as handle:
    TriangulationIndexes = pickle.load(handle)

nof_triangles=TriangulationIndexes.shape[0]


"""Relevant NNs"""
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


"""Load Source Image"""
directory='directportrait.jpg'
directory='sundayfunday5.png'
directory='orc.jpg'



source_image_rgb=cv2.imread(directory)
source_image_gray=cv2.imread(directory,0)

"""Find and Extract Source Image Landmarks"""
faces=face_detector(source_image_gray)
source_landmarks = pre.ExtractLandmarks(source_image_gray,faces,landmark_predictor,False)

SourceTriangles,SourceTransform=pre.GetList_CroppedTriangulation(source_image_rgb,\
                                                         source_landmarks,\
                                                         TriangulationIndexes)
        

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
    
    """Find Transform and Aplly it"""
    for t in range(nof_triangles):
        SourcePoints_xy=np.float32(SourceTransform[t][0])
        DestPoints_xy=np.float32(DestTransform[t][0])
        
        SourceRect_xy=np.float32(SourceTransform[t][1])
        DestRect_xy=np.float32(DestTransform[t][1])
        
        DestRect_wh=(DestTransform[t][2][0], DestTransform[t][2][1])
        SourceRect_wh=(SourceTransform[t][2][0], SourceTransform[t][2][1])
        T=cv2.getAffineTransform(SourcePoints_xy-SourceRect_xy,\
                                 DestPoints_xy-DestRect_xy)
        
            
        TransferedTriangle = cv2.warpAffine(SourceTriangles[t], T, (DestRect_wh[0],DestRect_wh[1]))

        
        y_low=np.int32(DestRect_xy[1])
        y_high=np.int32(DestRect_xy[1]+DestRect_wh[1])
        
        x_low=np.int32(DestRect_xy[0])
        x_high=np.int32(DestRect_xy[0]+DestRect_wh[0])
        
        #transfer_face[y_low:y_high,x_low:x_high,:]=transfer_face[y_low:y_high,x_low:x_high,:]+\
        #    TransferedTriangle
        
        ROI_index=np.where(TransferedTriangle != 0)
        
        transfer_face_ROI=transfer_face[y_low:y_high,x_low:x_high,:]
        transfer_face_ROI[ROI_index]=TransferedTriangle[ROI_index]
        transfer_face[y_low:y_high,x_low:x_high,:]=transfer_face_ROI
        
        
        
        
    #transfer_face =cv2.medianBlur( transfer_face , 5) 

    frame=frame*(transfer_face==0)
    frame=frame+transfer_face
    frame=np.uint8(frame)

    scale_percent = 10 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


    cv2.imshow("a",resized_frame)

    
    key=cv2.waitKey(1)

    if(key==65):
        cv2.waitKey(0)
        break

cap.release()
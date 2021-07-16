# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 19:34:54 2021

@author: OKYANUS
"""
import numpy as np
import cv2
import dlib
import copy
import matplotlib.pyplot as plt
import pickle


def Indexed_Triangulation(name_of_pickle_file,source_image_destination,visual_bool):
    
    gray = cv2.imread(source_image_destination,0)
    
    
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    subdiv = cv2.Subdiv2D((0,0,gray.shape[0],gray.shape[1]))
    
    faces=face_detector(gray)
    indexed_landmark_points=np.zeros(shape=(68,2))
    
    
    for face in faces:
        landmarks = landmark_predictor(gray,face)
        for i in range(68):
            
            p = (landmarks.part(i).x,landmarks.part(i).y)
    
            indexed_landmark_points[i,0]=p[0]
            indexed_landmark_points[i,1]=p[1]
            
            cv2.circle(gray,p,3,(0,0,0),-1)
            subdiv.insert(p)
            
    triangles=np.array(subdiv.getTriangleList().astype('int32')).reshape(-1,3,2)
    indexed_triangles=np.zeros(shape=(triangles.shape[0],3)).astype('int32')
    
    
    for t in range(triangles.shape[0]):
        for i in range(3):
            p=triangles[t,i,:]
            point_index=np.where(np.all(indexed_landmark_points == p, axis=1))[0][0]
            indexed_triangles[t,i]=point_index
    
    with open(name_of_pickle_file+".pickle", 'wb') as handle:
        pickle.dump(indexed_triangles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if(visual_bool):
        cv2.imshow("Triangulation",gray)
        cv2.waitKey(0)
    
def ExtractLandmarks(gray,faces,landmark_predictor,plot_bool):
   
    landmarks=np.zeros(shape=(68,2),dtype='int32')
    
    """Extract Landmarks"""
    for face in faces:
        pts = landmark_predictor(gray,face).parts()
        for i in range(68):
            landmarks[i,0]=pts[i].x
            landmarks[i,1]=pts[i].y    
        
            if(plot_bool):
                cv2.circle(gray,(landmarks[i,0],landmarks[i,1]),3,(0,0,255) ,-1)
        if(plot_bool):
            cv2.imshow('a',gray)
            cv2.waitKey(1)
    return landmarks

def GetList_CroppedTriangulation(source_image,source_landmarks,TriangulationIndexes):
    
    nof_triangles=TriangulationIndexes.shape[0]
    
    cropped_triangles=[]
    transform=[]
    
    for t in range(nof_triangles):
        triangle_xy=np.zeros(shape=(3,2),dtype='int32')
        for p in range(3):
            triangle_xy[p,:]=source_landmarks[TriangulationIndexes[t,p],:]
        
        
        x,y,w,h = cv2.boundingRect(triangle_xy)
        
        
        Region_of_Interest=source_image[y:(y+h),x:(x+w),:]
        
        mask=np.zeros(shape=(h,w,3),dtype='int32')
        mask=cv2.fillConvexPoly(mask, triangle_xy-np.array([x,y]), (1,1,1))
     
        cropped_triangle=np.uint8(Region_of_Interest*mask)
        cropped_triangles.append(cropped_triangle)
        
        
        
        transform.append( (triangle_xy, np.array([x,y]), np.array([w,h])) )
    
    return cropped_triangles,transform
    
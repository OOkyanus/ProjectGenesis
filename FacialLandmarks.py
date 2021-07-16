# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:53:39 2021

@author: OKYANUS
"""
import numpy as np
import cv2
import dlib

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_delaunay(img, subdiv, delaunay_color ) :
    
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
 
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)



if __name__ == '__main__':


    
    cap = cv2.VideoCapture(0)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    

    while True:
        _,frame=cap.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces=detector(gray)
        
        
        delaunay_color = (255,255,255)
        
        subdiv = cv2.Subdiv2D((0,0,1000,1000))
    

        for face in faces:
            x1=face.left()
            x2=face.right()
            y1=face.top()
            y2=face.bottom()
            
            landmarks = predictor(gray,face)
            
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),3)    
            for i in range(68):
                
                p = (landmarks.part(i).x,landmarks.part(i).y)
                
                cv2.circle(frame,p,3,(255,0,0),-1)
                subdiv.insert(p)
        
        draw_delaunay(frame, subdiv, delaunay_color )
        
        cv2.imshow("Frame", frame)
        
        key=cv2.waitKey(1)
        
        if key==65:
            key=cv2.waitKey(0)
    
            break
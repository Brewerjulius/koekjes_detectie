# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:32:50 2022

@author: Julius Klein
"""

import cv2
import glob
# import copy
import numpy as np
# from matplotlib import pyplot as plt
import time


def Scaling(image_for_scaling):
    # Scaling setup
    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image_for_scaling = cv2.resize(image_for_scaling, dim, interpolation = cv2.INTER_AREA)
    return image_for_scaling


def nothing(x):
    pass


def static_windows():
    # Show Static images
    # RGB - Blue
    cv2.imshow('B-RGB', image_RGB[:, :, 0])
    # RGB - Green
    cv2.imshow('G-RGB', image_RGB[:, :, 1])
    # RGB Red
    cv2.imshow('R-RGB', image_RGB[:, :, 2])
    # HSV - H
    cv2.imshow('H-HSV', image_HSV[:, :, 0])
    # HSV - S
    cv2.imshow('S-HSV', image_HSV[:, :, 1]) # Good
    # HSV - V
    cv2.imshow('V-HSV', image_HSV[:, :, 2])
    # HSV
    cv2.imshow("HSV", image_HSV)    # Good
    # RGB
    cv2.imshow("RGB", image_RGB)    # Good
    # Original image
    cv2.imshow("original_image", original_image)
    # Grayscaled image
    cv2.imshow("Gray", gray_image)


def binary_conversion(image_to_binary, name):
    image_to_binary = cv2.cvtColor(image_to_binary, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(image_to_binary, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow(name, thresh)

    return thresh


def image_contour_create(binary_image, original_image, name):
    # detect the image_contour_create on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=binary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw image_contour_create on the original image
    image_copy = original_image.copy()

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # see the results
    cv2.imshow(name, image_copy)


def image_contour_show():
    # Using Canny to make edges
    edges_gray = cv2.Canny(gray_image, Min, Max)
    edge_image_HSV_S = cv2.Canny(image_HSV_S, Min, Max)
    edge_image_HSV = cv2.Canny(image_HSV, Min, Max)
    edge_image_RGB = cv2.Canny(image_RGB, Min, Max)

    cv2.imshow("Edges", edges_gray)
    cv2.imshow("edge_HSV_S", edge_image_HSV_S)
    cv2.imshow("edge_HSV", edge_image_HSV)
    cv2.imshow("edge_RGB", edge_image_RGB)

    # image_contour_create(binary_gray, gray_image, "gray_image")
    # image_contour_create(binary_HSV_S, image_HSV_S, "image_HSV_S")
    # image_contour_create(binary_HSV, image_HSV, "image_HSV")

    image_contour_create(edges_gray, gray_image, "Contours Gray Image")
    image_contour_create(edge_image_HSV_S, image_HSV_S, "Contours HSV S Image")
    image_contour_create(edge_image_HSV, image_HSV, "Contours HSV Image")
    image_contour_create(edge_image_RGB, image_RGB, "Contours RGB Image")


def trackbars():
    # Creating a window with black image
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # creating trackbars for Min value
    cv2.createTrackbar('Min', 'image', 0, 400, nothing)

    # creating trackbars for Max value
    cv2.createTrackbar('Max', 'image', 0, 400, nothing)

    # creating trackbar for closing program.
    cv2.createTrackbar('Close Program', 'image', 0, 1, nothing)

    # creating trackbar for going to the next picture.
    cv2.createTrackbar('Next Photo', 'image', 0, 1, nothing)
    # end trackbars

trackbars()

# reading all pictures
for filename in glob.glob('L:\Onedrive Julius\OneDrive - Stichting Hogeschool Utrecht\School\Derde jaar\Beeldherkenning\Images\Test Samples\*\*.png'):
    # read pic into variable
    image = cv2.imread(filename)
    
    image = Scaling(image)
    
    # Make backup of pic
    original_image=image.copy() 
    
    # Copying image to make it gray
    gray_image = image.copy()
    
    # making gray_image gray
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) - OUD, zet in verslag
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # making new image where Blue and Red channels are swapped. Eigenlijk zie je BGR
    #image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # converting image to HSV 0-179 from RGB 0-255. De parameters worden 1 op 1 overgezet, en alles boven de 179 wordt verandered naar: 0 + de orginele waarde - 179
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Making a copy of an image for later processing.
    image_HSV_S = image_HSV[:, :, 1].copy()
    
    static_windows()
    
    while(True):
        
        # for button pressing and changing
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        #Trackbar Sliders
        Min = cv2.getTrackbarPos('Min', 'image')
        Max = cv2.getTrackbarPos('Max', 'image')
        Close_Program = cv2.getTrackbarPos('Close Program', 'image')
        Next_Picture = cv2.getTrackbarPos('Next Photo', 'image')

        if Close_Program == 1:
            break
        if Next_Picture == 1:
            cv2.setTrackbarPos('Next Photo', 'image', 0)
            break

        ###################################################################
        # apply binary thresholding
        #binary_gray = Binary_conversion(gray_image, "binary_gray")
        #binary_HSV_S = Binary_conversion(image_HSV_S, "binary_HSV_S")
        #binary_HSV = Binary_conversion(image_HSV, "binary_HSV")
        #############################################################
        image_contour_show()





cv2.destroyAllWindows()


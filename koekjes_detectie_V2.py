# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:32:50 2022

@author: Julius Klein
"""

import cv2
import glob
import copy
import numpy as np
# from matplotlib import pyplot as plt
import time


def Scaling(image_for_scaling):
    # Scaling setup
    scale_percent = 30  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image_for_scaling = cv2.resize(image_for_scaling, dim, interpolation=cv2.INTER_AREA)
    return image_for_scaling


def nothing(x):
    pass


def static_windows():
    # Show Static images
    # RGB - Blue
    if True == False:
        cv2.imshow('B-RGB', image_RGB[:, :, 0])
        # RGB - Green
        cv2.imshow('G-RGB', image_RGB[:, :, 1])
        # RGB Red
        cv2.imshow('R-RGB', image_RGB[:, :, 2])
        # HSV - H
        cv2.imshow('H-HSV', image_HSV[:, :, 0])
        # HSV - S
        cv2.imshow('S-HSV', image_HSV[:, :, 1])  # Good
        # HSV - V
        cv2.imshow('V-HSV', image_HSV[:, :, 2])
        # HSV
        cv2.imshow("RGB 0-255 > HSV 0-179", image_HSV)  # Good
        # RGB
        cv2.imshow("RGB - Blue Red channel swap", image_RGB)  # Good
        # Original image
    cv2.imshow("original_image", original_image)
    # Grayscaled image
    cv2.imshow("Gray", gray_image)


def binary_conversion(image_to_binary, name):
    # image_to_binary = cv2.cvtColor(image_to_binary, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image_to_binary, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow(name, thresh)

    return thresh


def erosion_dilation(img, kernel_1, kernel_2, iterations_trackbar):
    kernel = np.ones((kernel_1, kernel_2), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=iterations_trackbar)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=iterations_trackbar)

    edges_gray_erosion_dialation = np.hstack((img_erosion, img_dilation))

    cv2.imshow("edges_gray_erosion_dialation", edges_gray_erosion_dialation)

    return img_erosion, img_dilation


def trackbars():
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 700, 400)

    # creating trackbars for Min value
    cv2.createTrackbar('Min', 'image', 0, 400, nothing)

    # creating trackbars for Max value
    cv2.createTrackbar('Max', 'image', 0, 400, nothing)

    # creating trackbar for closing program.
    cv2.createTrackbar('Close Program', 'image', 0, 1, nothing)

    # creating trackbar for going to the next picture.
    cv2.createTrackbar('Next Photo', 'image', 0, 1, nothing)

    # creating trackbar for kernel_1
    cv2.createTrackbar('kernel_1', 'image', 0, 10, nothing)

    # creating trackbar for kernel_2
    cv2.createTrackbar('kernel_2', 'image', 0, 10, nothing)

    # creating trackbar for iterations_trackbar
    cv2.createTrackbar('iterations_trackbar', 'image', 0, 10, nothing)



trackbars()

# reading all pictures
for filename in glob.glob(
        './Test Samples/top-Test-Set/*.png'):
    # read pic into variable
    image = cv2.imread(filename)

    # scale image
    image = Scaling(image)

    # Make backup of pic
    original_image = image.copy()

    # making gray_image gray
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) - OUD, zet in verslag
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # making new image where Blue and Red channels are swapped. Eigenlijk zie je BGR
    # image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # converting image to HSV 0-179 from RGB 0-255. De parameters worden 1 op 1 overgezet, en alles boven de 179 wordt verandered naar: 0 + de orginele waarde - 179
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Making a copy of an image for later processing.
    image_HSV_S = image_HSV[:, :, 1].copy()

    # create static windows for visualization purpose.
    static_windows()

    # activate main program loop.
    while (True):

        # for button pressing and changing
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # Trackbar Sliders
        Close_Program = cv2.getTrackbarPos('Close Program', 'image')
        Next_Picture = cv2.getTrackbarPos('Next Photo', 'image')
        kernel_1 = cv2.getTrackbarPos('kernel_1', 'image')
        kernel_2 = cv2.getTrackbarPos('kernel_2', 'image')
        iterations_trackbar = cv2.getTrackbarPos('iterations_trackbar', 'image')

        if Close_Program == 1:
            break
        if Next_Picture == 1:
            cv2.setTrackbarPos('Next Photo', 'image', 0)
            break

        ###################################################################
        # apply binary thresholding
        binary_HSV_S = binary_conversion(image_HSV_S, "binary_HSV_S")
        #############################################################

        erosion_dilation(binary_HSV_S, kernel_1, kernel_2, iterations_trackbar)

cv2.destroyAllWindows()

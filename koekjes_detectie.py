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


def image_contour_create(binary_image, original_image, name):
    # if name = None, no picture window will be opened.

    # detect the image_contour_create on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=binary_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw image_contour_create on the original image
    image_copy = original_image.copy()

    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)

    if name != None:
        # see the results
        cv2.imshow(name, image_copy)

    return image_copy


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


def contour_drawer(image_Bounding_Box, original_image_draw, name):
    contours, hierarchy = cv2.findContours(image=image_Bounding_Box, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    cnt = contours[4]

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(original_image_draw, [box], 0, (0, 0, 255), 2)

    cv2.imshow(name, original_image_draw)


def box_circle_drawer(image_Bounding_Box, original_image_draw, name):
    contours, hierarchy = cv2.findContours(image=image_Bounding_Box, mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)
    areaArray = []
    count = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    # find the nth largest contour [n-1][1], in this case 2
    secondlargestcontour = sorteddata[1][1]

    # draw it
    x, y, w, h = cv2.boundingRect(secondlargestcontour)
    cv2.drawContours(original_image_draw2, secondlargestcontour, -1, (255, 0, 0), 2)
    cv2.rectangle(original_image_draw2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    Circle_x = int((x + x + w) / 2)
    Circle_y = int((y + y + h) / 2)

    if Circle_x > x:
        Radius = Circle_x - x
    elif x > Circle_x:
        Radius = x - Circle_x

    cv2.circle(original_image_draw2, (Circle_x, Circle_y), Radius, (0, 0, 255), -1)

    cv2.imshow('Photos/output3.jpg', original_image_draw2)


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

    # Copying image to make it gray
    gray_image = image.copy()

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
        Min = cv2.getTrackbarPos('Min', 'image')
        Max = cv2.getTrackbarPos('Max', 'image')
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
        # binary_gray = Binary_conversion(gray_image, "binary_gray")
        # binary_HSV_S = Binary_conversion(image_HSV_S, "binary_HSV_S")
        # binary_HSV = Binary_conversion(image_HSV, "binary_HSV")
        #############################################################

        # Using Canny to make edges
        edges_gray = cv2.Canny(gray_image, Min, Max)
        #edge_image_HSV_S = cv2.Canny(image_HSV_S, Min, Max)
        #edge_image_HSV = cv2.Canny(image_HSV, Min, Max)
        #edge_image_RGB = cv2.Canny(image_RGB, Min, Max)

        # Show the edge images
        cv2.imshow("Edges", edges_gray)
        #cv2.imshow("edge_HSV_S", edge_image_HSV_S)
        #cv2.imshow("edge_HSV", edge_image_HSV)
        #cv2.imshow("edge_RGB", edge_image_RGB)

        # Erode and Dialte the edge image.
        edges_gray_erosion, edges_gray_dialation = erosion_dilation(edges_gray, kernel_1, kernel_2, iterations_trackbar)

        # Use Eroded image as mask.
        mask = edges_gray_erosion

        # make a copy of the original image to put the mask on
        original_image_masked = image.copy()

        # put mask on image
        masked = cv2.bitwise_and(original_image_masked, original_image_masked, mask=mask)

        # show masked image
        cv2.imshow("masked", masked)

        # zet in verslag: Contouren getest op pre processed images, resultaat was niet bruikbaar.
        # image_contour_create(binary_gray, gray_image, "gray_image")
        # image_contour_create(binary_HSV_S, image_HSV_S, "image_HSV_S")
        # image_contour_create(binary_HSV, image_HSV, "image_HSV")

        image_contour_create(edges_gray, gray_image, "Contours Gray Image")

        # copy gray image
        gray_copy = gray_image.copy()

        # blur gray image
        blur = cv2.GaussianBlur(gray_copy, (5, 5), 0)
        blur2 = cv2.GaussianBlur(gray_copy, (11, 11), 0)
        blur3 = cv2.GaussianBlur(gray_copy, (21, 21), 0)

        cv2.imshow("blur", blur)
        cv2.imshow("blur2", blur2)
        cv2.imshow("blur3", blur3)


        image_contour_create(blur, gray_image, "Blur contour Image")
        image_contour_create(blur2, gray_image, "Blur contour Image2")
        image_contour_create(blur3, gray_image, "Blur contour Image3")

        #image_contour_create(edge_image_HSV_S, image_HSV_S, "Contours HSV S Image")
        #image_contour_create(edge_image_HSV, image_HSV, "Contours HSV Image")
        #image_contour_create(edge_image_RGB, image_RGB, "Contours RGB Image")

        # ###############################################################
        # image_Bounding_Box = image_contour_create(edges_gray, gray_image, None)
        # original_image_draw = original_image.copy()
        # contour_drawer(image_Bounding_Box, original_image_draw, "contour_original_image")
        #
        # ###############################################################
        # original_image_draw2 = original_image.copy()
        # box_circle_drawer(image_Bounding_Box, original_image_draw, 'Photos/output3.jpg')
        # ###############################################################


        # maak een circle binnen het vierkant wat getekend wordt.
        # maak het vierkant kunnen draaien.
        # houghcricles()

        # pixels van de contour optellen om ruigheid van surface te detecteren.

        # LBP

        # difference in gausian

        # roteer fotos om het kleinst mogelijke vierkant te krijgen - voor de prins koek vooral

        # hough lines voor de prins en de stroopwafel koekjes

cv2.destroyAllWindows()

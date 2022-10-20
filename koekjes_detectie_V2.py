# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:32:50 2022

@author: Julius Klein
"""

import cv2
import glob
import copy
import numpy as np
from matplotlib import pyplot as plt


def Scaling(image_for_scaling, scale_percent):

    # Scaling setup
    if scale_percent == None:
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


def pixel_counter(image):
    number_of_white_pix = np.sum(image == 255)
    number_of_black_pix = np.sum(image == 0)

    #print('Number of white pixels:', number_of_white_pix)
    #print('Number of black pixels:', number_of_black_pix)

    return number_of_white_pix


def histogram_rgb_plot(image, mask):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], mask, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


def histogram_rgb_max_value(image, mask):
    color = ('b')
    for i, col in enumerate(color):
        histr_b = cv2.calcHist([image], [0], mask, [256], [0, 256])
    histr_b_max = max(histr_b)

    color = ('g')
    for i, col in enumerate(color):
        histr_g = cv2.calcHist([image], [1], mask, [256], [0, 256])
    histr_g_max = max(histr_g)

    color = ('r')
    for i, col in enumerate(color):
        histr_r = cv2.calcHist([image], [2], mask, [256], [0, 256])
    histr_r_max = max(histr_r)

    print(histr_b_max, histr_g_max, histr_r_max)

    return histr_b_max, histr_g_max, histr_r_max


def color_identifier(blue_value, green_value, red_value):

    #print(blue_value, green_value, red_value)

    if blue_value >= 900 and green_value <= 200 and red_value <= 200:
        koekje = "Kokosmacroon"

        # Kokosmacroon
        # Blue => Y 900 tussen x 0 en x 25
        # Red Green < 200

    elif (200 <= blue_value <= 260) and (200 <= green_value <= 260) and (200 <= red_value <= 260):
        koekje = "Pennywafel Choco kant"
        # Pennywafel Choco kant
        # Red Green Blue => 200 && =< 260

    elif (35 < blue_value < 90): #and (35 < green_value < 70) and (35 < red_value < 70):
        koekje = "Pennywafel NIET choco"
        # Pennywafel NIET choco kant
        # Red Green Blue => 35 && =< 70

    elif (325 < blue_value < 500) and (0 < green_value < 100) and (0 < red_value < 100):
        koekje = "Vlinder koekje"

        # Vlinder koekje
        # Blue => 325 && <= 500
        # Red Green <= 100
    elif (500 < blue_value < 800) and (100 < green_value < 300) and (100 < red_value < 300):
        koekje = "Choco chip"
        # Choco chip
        # BLue => 500 && <= 800
        # Red Green => 100 && <= 300
    elif (500 < blue_value < 850) and (100 < green_value < 1000) and (100 < red_value < 1000):
        koekje = "stroopwafel"
    else:
        koekje = 404

    if koekje == "Choco chip" or koekje == "stroopwafel":
        if pixel_counter(chocolate_detector) >= 20:
            koekje = "Choco chip"
        else:
            koekje = "stroopwafel"

    return koekje


# #def lines(input_image, lower_canny, upper_canny):
#     dst = cv2.Canny(input_image, lower_canny, upper_canny, None, 3)
#
#     # Copy edges to the images that will display the results in BGR
#     cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
#     cdstP = np.copy(cdst)
#
#     lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
#
#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
#             pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
#             cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
#
#     linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
#
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
#
#         input_image = Scaling(input_image, 50)
#         cdst = Scaling(cdst, 50)
#         cdstP = Scaling(cdstP, 50)
#
#     cv2.imshow("Source", input_image)
#     cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#     cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)


def trackbars():
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 700, 400)

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

    # creating trackbar for iterations_trackbar
    cv2.createTrackbar('lower_canny', 'image', 0, 300, nothing)

    cv2.setTrackbarPos('lower_canny', 'image', 50)

    # creating trackbar for iterations_trackbar
    cv2.createTrackbar('upper_canny', 'image', 0, 300, nothing)

    cv2.setTrackbarPos('upper_canny', 'image', 200)

trackbars()

# reading all pictures
for filename in glob.glob(
        './Test Samples/top-Test-Set/*.*'):
    # read pic into variable
    image = cv2.imread(filename)

    # scale image
    image = Scaling(image, None)

    # Make backup of pic
    original_image = image.copy()

    # cycle counter reset
    cycle_counter = 0

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
        lower_canny = cv2.getTrackbarPos('lower_canny', 'image')
        upper_canny = cv2.getTrackbarPos('upper_canny', 'image')

        if Close_Program == 1:
            break
        if Next_Picture == 1:
            cv2.setTrackbarPos('Next Photo', 'image', 0)
            break

        # apply binary thresholding
        binary_HSV_S = binary_conversion(image_HSV_S, "binary_HSV_S")

        # making final mask version
        final_mask, _ = erosion_dilation(binary_HSV_S, kernel_1, kernel_2, iterations_trackbar)

        # copying original image, on this copy the final mask will be applied.
        original_masked = original_image.copy()
        # placing mask on image
        original_masked = cv2.bitwise_and(original_masked, original_masked, mask=final_mask)

        cv2.imshow("original_masked", original_masked)

#########################################################################
        # Cookie identification from here on!
#########################################################################

        lower_red = np.array([10, 10, 10], dtype="uint8")

        upper_red = np.array([69, 69, 69], dtype="uint8")

        chocolate_detector = cv2.inRange(original_masked, lower_red, upper_red)

        cv2.imshow("chocolate_detector", chocolate_detector)

        #lines(original_masked, lower_canny, upper_canny)

        if cycle_counter == 0:
            histogram_rgb_plot(original_image, final_mask)
            blue_value, green_value, red_value = histogram_rgb_max_value(original_image, final_mask)

            koekje = color_identifier(blue_value, green_value, red_value)

            ##//todo make it work

            print(koekje)
            cycle_counter = 1



cv2.destroyAllWindows()
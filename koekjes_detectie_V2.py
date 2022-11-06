# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:32:50 2022
@author: Julius Klein
"""

import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pandas as pd
import os

def scaling(image_for_scaling, scale_percent):

    # Scaling setup
    if scale_percent == None:
        scale_percent = 30  # percent of original size

    # setting scaling for width and height
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # set dimensions
    dim = (width, height)
    # resize image
    image_for_scaling = cv2.resize(image_for_scaling, dim, interpolation=cv2.INTER_AREA)
    return image_for_scaling


def binary_conversion(image_to_binary, name):
    # set image to binary
    ret, thresh = cv2.threshold(image_to_binary, 150, 255, cv2.THRESH_BINARY)
    return thresh


def erosion_dilation(img, kernel_1, kernel_2, iterations_trackbar):
    # make kernel based on sliders
    kernel = np.ones((kernel_1, kernel_2), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=iterations_trackbar)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=iterations_trackbar)

    return img_erosion, img_dilation


def pixel_counter(image):
    # count the number of white pixels
    number_of_white_pix = np.sum(image == 255)

    return number_of_white_pix


def histogram_rgb_max_value(image, mask):
    # get the max value of 'b' from the histogram
    color = ('b')
    for i, col in enumerate(color):
        histr_b = cv2.calcHist([image], [0], mask, [256], [0, 256])
    histr_b_max = max(histr_b)

    # get the max value of 'g' from the histogram
    color = ('g')
    for i, col in enumerate(color):
        histr_g = cv2.calcHist([image], [1], mask, [256], [0, 256])
    histr_g_max = max(histr_g)

    # get the max value of 'r' from the histogram
    color = ('r')
    for i, col in enumerate(color):
        histr_r = cv2.calcHist([image], [2], mask, [256], [0, 256])
    histr_r_max = max(histr_r)

    return histr_b_max, histr_g_max, histr_r_max


def color_identifier(blue_value, green_value, red_value):

    # kokosmacroon = 1
    # Pennywafel Choco = 2
    # Penny wafel Niet choco = 3
    # Vlinder koekje = 4
    # choco chip = 5
    # stroopwafel = 6

    # setting basic values for the array and 'koekje' variable to avoid errors
    output_number_color = ["color"]
    koekje = "opgegeten"

    if (425 <= blue_value <= 1200) and (75 <= green_value <= 145) and (55 <= red_value <= 130):
        # set koekje to Kokosmacroon
        koekje = "Kokosmacroon"
        # append number 1 to the output_number_color
        output_number_color.append(1)

    if (90 <= blue_value <= 480) and (55 <= green_value <= 450) and (40 <= red_value <= 440):
        # set koekje to Pennywafel Choco kant
        koekje = "Pennywafel Choco kant"
        # append number 2 to the output_number_color
        output_number_color.append(2)

    if (20 <= blue_value <= 95) and (15 <= green_value <= 95) and (10 <= red_value <= 95): #and (35 < green_value < 70) and (35 < red_value < 70):
        # set koekje to Pennywafel NIET choco
        koekje = "Pennywafel NIET choco"
        # append number 3 to the output_number_color
        output_number_color.append(3)

    if (90 < blue_value < 730) and (45 < green_value < 90) and (45 < red_value < 120):
        # set koekje to Vlinder koekje
        koekje = "Vlinder koekje"
        # append number 4 to the output_number_color
        output_number_color.append(4)

    if (180 < blue_value < 760) and (70 < green_value < 220) and (70 < red_value < 220):
        # set koekje to Choco chip
        koekje = "Choco chip"

    if (500 < blue_value < 1200) and (140 < green_value < 220) and (115 < red_value < 170):
        # set koekje to stroopwafel
        koekje = "stroopwafel"

    if koekje == "Choco chip" or koekje == "stroopwafel":

        # set threshold for chocolate identification
        lower_red = np.array([10, 10, 10], dtype="uint8")
        upper_red = np.array([69, 69, 69], dtype="uint8")
        # detect chocolate
        chocolate_detector = cv2.inRange(original_masked, lower_red, upper_red)

        if pixel_counter(chocolate_detector) >= 20:
            # set koekje to Choco chip
            koekje = "Choco chip"
            # append number 5 to the output_number_color
            output_number_color.append(5)
        else:
            # set koekje to stroopwafel
            koekje = "stroopwafel"
            # append number 6 to the output_number_color
            output_number_color.append(6)

    if len(output_number_color) == 1:
        # if the output_number_color doesn't have any added values, set/add 404 to koekje and output_number_color
        koekje = 404
        output_number_color.append(404)

    return output_number_color


def rotator(image, rotation_angle):
    # rotate image by rotation_angle
    rotate_image = imutils.rotate(image, rotation_angle)
    return rotate_image


def box_circle_drawer(input_gray_image, original_image):
    # make loop counter and set it to 0
    counter = 0
    # make variable for storing max difference and set it to 0
    max_difference = 0
    output_number_shape = []

    # loop 360 times (rotating the image 360 degrees)
    while counter <= 360:

        # rotate gray image
        input_gray_image_rotated = rotator(input_gray_image, counter)
        # make edge image of input image
        edges_gray = cv2.Canny(input_gray_image_rotated, 110, 0)

        input_2 = input_gray_image_rotated.copy()
        input_3 = input_gray_image_rotated.copy()

        # detect the image_contour_create on the binary/edge image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=edges_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # copy the gray image so we dont overwrite it
        image_countours_V1 = input_gray_image_rotated.copy()

        # draw image_contour_create on the copy of original image
        cv2.drawContours(image=image_countours_V1, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

        # find countours of edge image
        contours, hierarchy = cv2.findContours(image=image_countours_V1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        areaArray = []

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            areaArray.append(area)

        # first sort the array by area
        sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

        # find the nth largest contour [n-1][1], in this case 2
        secondlargestcontour = sorteddata[1][1]

        # get the X and Y cordinate, and the width and hight data from the second largest contour
        x, y, w, h = cv2.boundingRect(secondlargestcontour)
        # draw countours
        cv2.drawContours(input_2, secondlargestcontour, -1, (255, 0, 0), 2)
        # draw rectangle
        cv2.rectangle(input_3, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # combine images into one final output
        combined_image = cv2.bitwise_and(input_2, input_3, mask=None)

        # Loop counter + 1
        counter = counter + 1

        # find the bigger variable, then subtract big from small to calculate the difference
        if w > h:
            difference = (w-h)
        elif h > w:
            difference = (h-w)
        else:
            difference = 0

        # compare current difference with max found difference
        if difference >= max_difference:
            # set new max difference if current difference is higher
            max_difference = difference

    if max_difference > 50:
        # if max_difference is greater than 50, cookie is rectangle
        output_number_shape = ["shape", 2, 3]
        output_shape = 1
    else:
        # cookie is round or square
        output_number_shape = ["shape", 1, 4, 5, 6]
        output_shape = 0

    return output_number_shape, output_shape


def contrast_functie(image_contrast_input):
    # convert to LAB color space
    contrast_image = cv2.cvtColor(image_contrast_input, cv2.COLOR_BGR2LAB)

    # separate channels
    L, A, B = cv2.split(contrast_image)

    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel_contrast = np.ones((5, 5), np.uint8)
    l_min = cv2.erode(L, kernel_contrast, iterations=1)
    l_max = cv2.dilate(L, kernel_contrast, iterations=1)

    # convert min and max to floats
    l_min = l_min.astype(np.float64)
    l_max = l_max.astype(np.float64)

    # compute local contrast
    contrast_variable = (l_max - l_min) / (l_max + l_min)

    # get average across whole image
    average_contrast = 100 * np.mean(contrast_variable)

    # setup array
    check_number_contrast = ["contrast"]

    if 5 <= average_contrast <= 7.2:
        # Kokosmacroon
        # append 1 to array
        check_number_contrast.append(1)

    if 2 <= average_contrast < 4.5:
        # Pennywafel Choco kant
        # append 2 to array
        check_number_contrast.append(2)

    if 3 <= average_contrast <= 4.5:
        # Pennywafel NIET choco
        # append 3 to array
        check_number_contrast.append(3)

    if 2 <= average_contrast <= 4.5:
        # Vlinder koekje
        # append 4 to array
        check_number_contrast.append(4)

    if 3 <= average_contrast <= 6:
        # Choco chip
        # append 5 to array
        check_number_contrast.append(5)

    if 5 <= average_contrast <= 6.5:
        # stroopwafel
        # append 6 to array
        check_number_contrast.append(6)

    if len(check_number_contrast) == 1:
        # append 404 to array
        check_number_contrast.append(404)

    return check_number_contrast, average_contrast


def cookie_identifier(color, shape, contrast):

    # make 1 array from all the inputs
    combined_array = color + shape + contrast

    # set biggest_number to the amount of 1s in the array
    biggest_number = combined_array.count(1)
    # set koekje to 1
    koekje_nummer = 1

    # check if the amount of 2s, if it's bigger than the current biggest_number, replace biggest_number with the 2 count
    if combined_array.count(2) > biggest_number:
        biggest_number = combined_array.count(2)
        koekje_nummer = 2

    # check if the amount of 3s, if it's bigger than the current biggest_number, replace biggest_number with the 3 count
    if combined_array.count(3) > biggest_number:
        biggest_number = combined_array.count(3)
        koekje_nummer = 3

    # check if the amount of 4s, if it's bigger than the current biggest_number, replace biggest_number with the 4 count
    if combined_array.count(4) > biggest_number:
        biggest_number = combined_array.count(4)
        koekje_nummer = 4

    # check if the amount of 5s, if it's bigger than the current biggest_number, replace biggest_number with the 5 count
    if combined_array.count(5) > biggest_number:
        biggest_number = combined_array.count(5)
        koekje_nummer = 5

    # check if the amount of 6s, if it's bigger than the current biggest_number, replace biggest_number with the 6 count
    if combined_array.count(6) > biggest_number:
        biggest_number = combined_array.count(6)
        koekje_nummer = 6

    return koekje_nummer, biggest_number


def koekje_name_finder(filename_input):

    # remove first part of string (path) based on path_cutter
    koekje_name_long = os.path.split(filename_input)[1]
    koekje_split = koekje_name_long.find("_")
    koekje_name = koekje_name_long[:koekje_split]

    # match filename to assign the matching number to koekje_verification_number
    if koekje_name == "kokosmacroon":
        koekje_verification_number = 1

    elif koekje_name == "pennywafel-choco":
        koekje_verification_number = 2

    elif koekje_name == "pennywafel-niet-choco":
        koekje_verification_number = 3

    elif koekje_name == "vlinderkoekje":
        koekje_verification_number = 4

    elif koekje_name == "chocolate-chip-b" or koekje_name == "chocolate-chip-t":
        koekje_verification_number = 5

    elif koekje_name == "stroopwafel":
        koekje_verification_number = 6

    # set koekje_verification_number to 405 when no match is found
    else:
        koekje_verification_number = 405

    return koekje_verification_number


with open('log.txt', 'w') as f:
    f.write('Succes_or_fail;Koekjes_Filename;Blue_value;Green_value;Red_value;Shape_value;Contrast_value'
            ';output_number_color;output_number_shape;output_number_contrast;most_common_number_amount;koekje_nummer'
            '\n')
    loop_counter = 0

    # reading all pictures
    for filename in glob.glob(
            './fotos/*.*'):

        # count the amount of loops that have happened
        loop_counter = loop_counter + 1

        # read pic into variable
        image = cv2.imread(filename)

        koekje_verification = koekje_name_finder(filename)

        # scale image
        image = scaling(image, None)

        # Make backup of pic
        original_image = image.copy()

        # cycle counter reset
        cycle_counter = 0

        # making gray_image gray
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # making new image where Blue and Red channels are swapped. Eigenlijk zie je BGR
        # image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # converting image to HSV 0-179 from RGB 0-255. De parameters worden 1 op 1 overgezet, en alles boven de 179 wordt verandered naar: 0 + de orginele waarde - 179
        image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Making a copy of an image for later processing.
        image_HSV_S = image_HSV[:, :, 1].copy()

        # activate main program loop.
        while (True):

            # for button pressing and changing
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # apply binary thresholding
            binary_HSV_S = binary_conversion(image_HSV_S, "binary_HSV_S")

            # making final mask version
            final_mask, _ = erosion_dilation(binary_HSV_S, 0, 0, 0)

            # copying original image, on this copy the final mask will be applied.
            original_masked = original_image.copy()
            # placing mask on image
            original_masked = cv2.bitwise_and(original_masked, original_masked, mask=final_mask)
            cv2.imshow("original_masked", original_masked)
    #########################################################################
            # Cookie identification from here on!
    #########################################################################

            blue_value, green_value, red_value = histogram_rgb_max_value(original_image, final_mask)

            cookie_color = color_identifier(blue_value, green_value, red_value)

            box_circle_output, shape_number = box_circle_drawer(gray_image, image)

            image_copy_contrast = image.copy()
            contrast_value, contrast_average = contrast_functie(image_copy_contrast)

            identifier_check, biggest_number = cookie_identifier(cookie_color, box_circle_output, contrast_value)

            #########################################################################
            # Start log file.
            #########################################################################

            # match outcome with verification number (for development only).
            if identifier_check == koekje_verification:
                f.write('success;')
            elif koekje_verification == 405:
                f.write('error;')
            elif identifier_check == 404:
                f.write('error;')
            else:
                f.write('fail;')

            # Koekjes filename added to log
            f.write(str(koekje_verification) + ';')
            # remove the .0 from blue, green, red values
            blue_value_str = str(blue_value[0])[:-2]
            green_value_str = str(green_value[0])[:-2]
            red_value_str = str(red_value[0])[:-2]
            # Blue, Green, Red value added to log
            f.write(blue_value_str + ";" + green_value_str + ";" + red_value_str + ";")
            # Shape Number added to log
            f.write(str(shape_number) + ";")
            # Contrast value added to log
            contrast_average_str = str(contrast_average).replace(".", ",")
            f.write(contrast_average_str + ";")
            # output_number_color added to log
            f.write(str(cookie_color) + ";")
            # output_number_shape added to log
            f.write(str(box_circle_output) + ";")
            # output_number_contrast added to log
            f.write(str(contrast_value) + ";")
            # most common number amount added to log
            f.write(str(biggest_number) + ";")
            # Koekje nummer added to log
            f.write(str(identifier_check) + "\n")

            print("still working! Loop:" + str(loop_counter))
            break

    cv2.destroyAllWindows()

# load data into excel
df = pd.read_csv('log.txt', sep=';')
df.to_excel('koekjes_detectie_output.xlsx', 'Data')
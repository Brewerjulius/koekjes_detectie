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
    if mydebug == 1:
        cv2.imshow("original_image", original_image)
        # Grayscaled image
        cv2.imshow("Gray", gray_image)


def binary_conversion(image_to_binary, name):
    # image_to_binary = cv2.cvtColor(image_to_binary, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(image_to_binary, 150, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    if mydebug == 1:
        cv2.imshow(name, thresh)

    return thresh


def erosion_dilation(img, kernel_1, kernel_2, iterations_trackbar):
    kernel = np.ones((kernel_1, kernel_2), np.uint8)

    img_dilation = cv2.dilate(img, kernel, iterations=iterations_trackbar)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=iterations_trackbar)

    edges_gray_erosion_dialation = np.hstack((img_erosion, img_dilation))
    if mydebug == 1:
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

    print("Blue", histr_b_max, "Green", histr_g_max, "Red", histr_r_max)

    return histr_b_max, histr_g_max, histr_r_max


def color_identifier(blue_value, green_value, red_value):

    # kokosmacroon = 1
    # Pennywafel Choco = 2
    # Penny wafel Niet choco = 3
    # Vlinder koekje = 4
    # choco chip = 5
    # stroopwafel = 6

    output_number_color = ["color"]
    koekje = "opgegeten"

    if (425 <= blue_value <= 1200) and (75 <= green_value <= 145) and (55 <= red_value <= 130):
        koekje = "Kokosmacroon"
        output_number_color.append(1)

        # Kokosmacroon
        # Blue => Y 900 tussen x 0 en x 25
        # Red Green < 200

    if (90 <= blue_value <= 480) and (55 <= green_value <= 450) and (40 <= red_value <= 440):
        koekje = "Pennywafel Choco kant"
        output_number_color.append(2)
        # Pennywafel Choco kant
        # Red Green Blue => 200 && =< 260

    if (20 <= blue_value <= 95) and (15 <= green_value <= 95) and (10 <= red_value <= 95): #and (35 < green_value < 70) and (35 < red_value < 70):
        koekje = "Pennywafel NIET choco"
        output_number_color.append(3)
        # Pennywafel NIET choco kant
        # Red Green Blue => 35 && =< 70

    if (90 < blue_value < 730) and (45 < green_value < 90) and (45 < red_value < 120):
        koekje = "Vlinder koekje"
        output_number_color.append(4)

        # Vlinder koekje
        # Blue => 325 && <= 500
        # Red Green <= 100
    if (180 < blue_value < 760) and (70 < green_value < 220) and (70 < red_value < 220):
        koekje = "Choco chip"
        #output_number_color.append(5)
        # Choco chip
        # BLue => 500 && <= 800
        # Red Green => 100 && <= 300
    if (500 < blue_value < 1200) and (140 < green_value < 220) and (115 < red_value < 170):
        koekje = "stroopwafel"
        #output_number_color.append(6)

    if koekje == "Choco chip" or koekje == "stroopwafel":
        if pixel_counter(chocolate_detector) >= 20:
            koekje = "Choco chip"
            output_number_color.append(5)
        else:
            koekje = "stroopwafel"
            output_number_color.append(6)

    if len(output_number_color) == 1:
        koekje = 404
        output_number_color.append(404)

    print("output_number_color:", output_number_color, koekje)
    return output_number_color


def rotator(image, rotation_angle):
    rotate_image = imutils.rotate(image, rotation_angle)
    # cv2.imshow("rotate_image", rotate_image)
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
        # rotate original image
        #original_image_rotated = rotator(original_image, counter)

        # make edge image of input image
        edges_gray = cv2.Canny(input_gray_image_rotated, 110, 0)

        # copy input image for later use
        # input_2 = original_image_rotated.copy()
        # input_3 = original_image_rotated.copy()
        # input_4 = original_image_rotated.copy()

        input_2 = input_gray_image_rotated.copy()
        input_3 = input_gray_image_rotated.copy()
        input_4 = input_gray_image_rotated.copy()


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
        count = 1

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

        # #find middle point of X axis. (X bottom + (X bottom + width) divided by 2 to find the middel
        # Circle_x = int((x + x + w) / 2)
        # # find middle point of Y axis (Y bottom + (Y bottom + hight) divided by 2 to find the middel
        # Circle_y = int((y + y + h) / 2)
        #
        #
        # if Circle_x > x:
        #     Radius = Circle_x - x
        # elif x > Circle_x:
        #     Radius = x - Circle_x
        #
        # cv2.circle(input_4, (Circle_x, Circle_y), Radius, (0, 0, 255), -1)

        combined_image = cv2.bitwise_and(input_2, input_3, mask=None)
        # combined_image = cv2.bitwise_and(combined_image, input_4, mask=None)

        # Loop counter + 1
        counter = counter + 1

        # find the bigger variable
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

    # show combined image
    cv2.imshow('Photos/output3.jpg', combined_image)
    #print(max_difference)

    if max_difference > 50:
        # if cookie is rectangle
        output_number_shape = ["shape", 2, 3]
        output_shape = 1
    else:
        # cookie is square
        output_number_shape = ["shape", 1, 4, 5, 6]
        output_shape = 0

    print("output_number_shape:", output_number_shape)
    return output_number_shape, output_shape


def contrast(image_contrast_input):
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

    #print("average_contrast:", str(average_contrast) + "%")


    check_number_contrast = ["contrast"]

    if 5 <= average_contrast <= 7.2:
        # Kokosmacroon
        check_number_contrast.append(1)


    if 2 <= average_contrast < 4.5:
        # Pennywafel Choco kant
        check_number_contrast.append(2)

    if 3 <= average_contrast <= 4.5:
        # Pennywafel NIET choco
        check_number_contrast.append(3)

    if 2 <= average_contrast <= 4.5:
        # Vlinder koekje
        check_number_contrast.append(4)

    if 3 <= average_contrast <= 6:
        # Choco chip
        check_number_contrast.append(5)

    if 5 <= average_contrast <= 6.5:
        # stroopwafel
        check_number_contrast.append(6)

    if check_number_contrast == 1:
        check_number_contrast.append(404)

    print("output_number_contrast:", check_number_contrast, average_contrast)
    return check_number_contrast, average_contrast


def cookie_identifier(color, shape, contrast):

    combined_array = color + shape + contrast

    biggest_number = combined_array.count(1)
    koekje_nummer = 1
    if combined_array.count(2) > biggest_number:
        biggest_number = combined_array.count(2)
        koekje_nummer = 2

    if combined_array.count(3) > biggest_number:
        biggest_number = combined_array.count(3)
        koekje_nummer = 3

    if combined_array.count(4) > biggest_number:
        biggest_number = combined_array.count(4)
        koekje_nummer = 4

    if combined_array.count(5) > biggest_number:
        biggest_number = combined_array.count(5)
        koekje_nummer = 5

    if combined_array.count(6) > biggest_number:
        biggest_number = combined_array.count(6)
        koekje_nummer = 6

    print("combined_array", combined_array)
    print("biggest number amount:", biggest_number, "koekje nummer:", koekje_nummer)

    return koekje_nummer, biggest_number

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


def koekje_name_finder(koekje_name):

    koekje_name = filename.partition("_")

    koekje_name = koekje_name[0]

    koekje_name = koekje_name[6:]

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

    else:
        koekje_verification_number = 405

    print("Koekjes Filename:", koekje_name)

    return koekje_verification_number


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

mydebug = 0

with open('log.txt', 'w') as f:
    f.write('Succes_or_fail;Koekjes_Filename;Blue_value;Green_value;Red_value;Shape_value;Contrast_value;output_number_color;output_number_shape;output_number_contrast;most_common_number_amount;koekje_nummer\n')

    # reading all pictures
    for filename in glob.glob(
            './top/*.*'):
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
            if mydebug == 1:
                cv2.imshow("original_masked", original_masked)

    #########################################################################
            # Cookie identification from here on!
    #########################################################################

            lower_red = np.array([10, 10, 10], dtype="uint8")

            upper_red = np.array([69, 69, 69], dtype="uint8")

            chocolate_detector = cv2.inRange(original_masked, lower_red, upper_red)

            if mydebug == 1:
                cv2.imshow("chocolate_detector", chocolate_detector)

            #lines(original_masked, lower_canny, upper_canny)

            if cycle_counter == 0:
                cycle_counter = 1

                histogram_rgb_plot(original_image, final_mask)

                blue_value, green_value, red_value = histogram_rgb_max_value(original_image, final_mask)

                cookie_color = color_identifier(blue_value, green_value, red_value)

                box_circle_output, shape_number = box_circle_drawer(gray_image, image)

                image_copy_contrast = image.copy()
                contrast_value, contrast_average = contrast(image_copy_contrast)

                identifier_check, biggest_number = cookie_identifier(cookie_color, box_circle_output, contrast_value)

                if identifier_check == koekje_verification:
                    f.write('success;')
                elif identifier_check == 404:
                    f.write('error;')
                else:
                    f.write('fail;')

                # Koekjes filename added to log
                f.write(str(koekje_verification) + ';')

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

            cv2.setTrackbarPos('Next Photo', 'image', 1)

    cv2.destroyAllWindows()
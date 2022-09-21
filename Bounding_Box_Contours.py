import numpy as np
import cv2
import glob

for filename in glob.glob('D:\\OneDrive - Stichting Hogeschool Utrecht\\School\\Derde jaar\\Beeldherkenning\\Images\\Test Samples\\top-Test-Set\\*.png'):
    # read pic into variable
    img = cv2.imread(filename, 0)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    print( M )

    radius = 3
    filtered_image_kernel_3 = cv2.GaussianBlur(img, (radius, radius), 0)
    radius = 15
    filtered_image_kernel_15 = cv2.GaussianBlur(img, (radius, radius), 0)
    cv2.imshow("filtered 3", filtered_image_kernel_3)
    cv2.imshow("filtered 15", filtered_image_kernel_15)
    cv2.waitKey(0)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(filtered_image_kernel_15, [box], 0, (0, 0, 255), 2)



    cv2.imshow("image", img)
    cv2.waitKey()
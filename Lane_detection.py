import cv2
import numpy as np
import matplotlib.pyplot as plt


#==definisson une fonction canny==#
def canny(image):
    # converting an image to gray scale
    gray_img = cv2.cvtColor(image, 0)
    # Reducing Noise
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    canny = cv2.Canny(blur, 40, 100)
    return canny


#=========Definition of a function allowing the Cropping the Region of Interest=====#
def Interest_regions(image):
    h= image.shape[0]
    T = np.array([
            [(200, h),(1100, h), (550,250)]
            ])
    Mask = np.zeros_like(image)
    cv2.fillPoly(Mask, T, [255, 255, 255])
    Masked_image = cv2.bitwise_and(canny, Mask)
    return  Masked_image


#=========Definition of a function allowing for the desplay of a the image lines=====#
def show_lines(image, lines):
    Line_Image=np.zeros_like(image)
    if lines is not None:
        for L in lines:
            #print(L)
            LN=L.reshape(4)
            A1, B1, A2, B2 = LN
            cv2.line(Line_Image, (A1, B1), (A2, B2), (255, 0, 0), 10) #LaneImage, start point, end point, color(BGR) ,thickness
    return Line_Image
    
    
#=====Définition de la fontion  HOUGHLINESP for drawing detected  lane marking on the image====#   
def HOUGHLINESP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
    return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap) #Edges, rho, theta, thresh , minLength ,max Gap
    


#=======================Implementation====================#

image = cv2.imread('Capture2.jpg')

lane_img_C = np.copy(image)

canny = canny(lane_img_C)

C_Image = Interest_regions(canny) #Appel de la fonction Interest_regions

Lines = HOUGHLINESP(C_Image, 2, 1*np.pi/180, 200, np.array([]), 50, 5)

L_Image=show_lines(lane_img_C, Lines)

cb_images=cv2.addWeighted(lane_img_C, 0.8, L_Image, 1, 1) # Allows to merge the image with the lines onto the original.

 #cv2.imshow('result', Lines)

#cv2.imshow('result', canny)

#cv2.imshow('result', C_Image)

#cv2.imshow('', L_Image)

cv2.imshow('', cb_images)

plt.show()

cv2.waitKey(0)

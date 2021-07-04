import cv2
import numpy as np

import Utility


class BinaryImage:

    def __init__(self, image, debug=False):
        self.debug = debug

        gray_image =image = self.convert2grayImage(image)
        remove_shadow_image = self.removeShadow(gray_image)

        self.binary_image = self.convert2binaryImage(remove_shadow_image)


    def convert2grayImage(self, rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        if (self.debug):
            Utility.showImage(gray,'gray image')
    
        return gray

    def removeShadow(self, gray_img):
        dilated_img = cv2.dilate(gray_img, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 99)
        diff_img = 255 - cv2.absdiff(gray_img, bg_img)
        norm_img = diff_img.copy()
        cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        if(self.debug):
            Utility.showImage(diff_img, 'remove shadow')

        return diff_img

    def convert2binaryImage(self, gray_img):

        bilateral = cv2.bilateralFilter(gray_img, 11, 40, 40)
        blur = cv2.GaussianBlur(bilateral,(5,5),0)
        thresh_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10)

        if(self.debug):
            # Utility.showImage(bilateral, 'bilateral image')
            # Utility.showImage(blur, 'blur')
            Utility.showImage(thresh_img, 'thresholding (binary image)')
            
        return thresh_img


class CellSheets:

    def __init__(self, binaryImage, debug=False, max_cols=20):
        self.debug = debug

        contour_sheet = self.contours(binaryImage, 'RETR_LIST')
        sheet_width = binaryImage.shape[1]
        cell_minWidth = (sheet_width/max_cols)
        cell_bounding = []

        contour_sheet_sorted, _ = self.sortContours(contour_sheet)

        i = 0
        amount_of_row = 0
        first_row_width = 0
        for cnt in contour_sheet_sorted:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if (len(approx) == 4):
                x, y, w, h = cv2.boundingRect(approx)

                if(w > cell_minWidth):
                    print("(width,height) : ({0},{1})".format(w,h))
                    cell_bounding.append(binaryImage[y:y+h, x:x+w])

                    if(i == 1):
                        first_row_width = w
                    
                    if(w >= first_row_width-100 and w <= first_row_width+100 and first_row_width > 0):
                        amount_of_row = amount_of_row + 1
        
                    if(self.debug):
                        Utility.showImage(binaryImage[y:y+h, x:x+w],"segment")
                    
                    i = i+1
                    
        self.scoreSheet = cell_bounding
        self.amountOfRow = amount_of_row

    def contours(self, binary_image, mode="RETR_EXTERNAL"):
        if mode == "RETR_EXTERNAL":
            contours, _ = cv2.findContours(binary_image , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elif mode == "RETR_LIST":
            contours, _ = cv2.findContours(binary_image , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def sortContours(self,contours, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
            
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (cnts, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
            
        return (cnts, boundingBoxes)

        
# class DigitBox:
#     def __init__(self, cellSheet, debug=False):



    
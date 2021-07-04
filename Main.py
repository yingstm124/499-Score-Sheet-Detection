from ImageProcessing import *
from Segmentation import *
from Prediction import *


import cv2




if __name__ == '__main__':
    
    img = cv2.imread('./asset/sample01.png')
    # Utility.showImage(img, 'original image')

    binary_image = BinaryImage(img).binary_image
    cell_sheets = CellSheets(binary_image, True)
    cellBoxs = cell_sheets.scoreSheet
    amount_of_row = cell_sheets.amountOfRow
    print(amount_of_row) 
    
    

    
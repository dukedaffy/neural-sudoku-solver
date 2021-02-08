
import cv2
import numpy as np
def extract(sudoku_loc):
    

    img_path = sudoku_loc
    img = cv2.imread(img_path)
    he, wi, c = img.shape
    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, kernel, iterations=1)

    # (2) threshold-inv and morph-open 
    th, threshed = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, np.ones((2,2)))

    # (3) find and filter contours, then draw on src 
    cnts = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]


    digitCnts=[]
    counter=0
    for cnt in cnts:
        x,y,w,h  = cv2.boundingRect(cnt)
        if ((w >= wi/11 and w<=wi/7.8) and (h >= he/11 and h <= he/7.8)):
            digitCnts.append([x,y,w,h])
            counter+=1
            cv2.rectangle(morphed, (x,y), (x+w, y+h), (0, 0, 255), 2, cv2.LINE_AA)
    print(counter)  
    print(np.shape(digitCnts))
    cv2.imwrite("dst.png", img)
    cv2.imwrite("morphed.png", morphed)

    img=morphed
    for i in range (81):
        pts = digitCnts[i]
        x1=pts[0]
        y1=pts[1]
        w1=pts[2]
        h1=pts[3]
        crop_img = img[x1:x1+h1, y1:y1+w1]
        cv2.imwrite("D:\\duke\\project\\sudoku\\cropped\\"+str(i)+".png", crop_img)
    


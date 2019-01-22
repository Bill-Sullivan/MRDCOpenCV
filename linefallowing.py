import cv2

import numpy as np

USE_GUI = False
leftRightThreshold = .98

def convBGRToHSV(pixel):
    #doesnt work
    print(pixel)
    print(type(pixel))
    tupplePixel = [[list(pixel),[0,0,0]]]
    print(tupplePixel)
    print(type(tupplePixel))
    hsv = cv2.cvtColor(cv2.UMat(tupplePixel), cv2.COLOR_BGR2HSV)
    print(hsv[0])
    

def selectColorRead(image, lowColorHSV, highColorHSV):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert image to Hue Shade Value
    
    mask = cv2.inRange(hsv, lowColorHSV, highColorHSV)
        
    imask = mask>0
    res = np.zeros_like(img, np.uint8)
    res[imask] = img[imask]
    if (USE_GUI == True):
        cv2.imshow('frame',image) 
        cv2.imshow('mask',mask) 
        cv2.imshow('res',res) 
        
        # This displays the frame, mask  
        # and res which we created in 3 separate windows. 
        cv2.waitKey(0)
        
        # Destroys all of the HighGUI windows. 
        cv2.destroyAllWindows()

    return res

def threshToBinary(image):    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert image to Hue Shade Value
    
    ret,thresh = cv2.threshold(gray, 80, 250, cv2.THRESH_BINARY)
    
    return thresh
    
def detectContores(binImage, image):
    contours, im2= cv2.findContours(binImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print("len contours")
    #print(len(contours))
    #print(contours[len(contours)-1])
    
    largestArea = np.array([0,0,0])
    if (len(contours) != 0):
        largestArea = max(contours, key = cv2.contourArea)
    #print(largestArea == contours[len(contours)-1])
    
    for contourGroup in contours:
        if (np.array_equal(largestArea, contourGroup)):
            for contour in contourGroup:
                contour = tuple(contour[0])
                
                cv2.circle(img, contour, 4, [255,255,0], thickness=5, lineType=8, shift=0)
        else: 
            for contour in contourGroup:
                contour = tuple(contour[0])
                
                cv2.circle(img, contour, 4, [255,0,0], thickness=5, lineType=8, shift=0)                
       
    if (USE_GUI == True):
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        
        cv2.imshow('contours',img) 
        
        # This displays the frame, mask  
        # and res which we created in 3 separate windows. 
        cv2.waitKey(0)
        
        # Destroys all of the HighGUI windows. 
        cv2.destroyAllWindows()
        
    return largestArea
    
def fallowLine(contures, img):
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(contures, cv2.DIST_L2,0,0.01,0.01)
    
    print(vx)
    print(vy)
    print(x)
    print(y)
    
    if (abs(vy) > leftRightThreshold):
        print("Go Strait")
    elif(vy>0):
        print("Turn Left")
    else: 
        print("Turn Right")
    
    if (USE_GUI == True):
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        print(lefty)
        print(righty)        
        
        cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)
        cv2.circle(img, (x,y), 4, [255,0,0], thickness=5, lineType=8, shift=0)                
    
        cv2.imshow('contours',img) 
            
        # This displays the frame, mask  
        # and res which we created in 3 separate windows. 
        cv2.waitKey(0)
        
        # Destroys all of the HighGUI windows. 
        cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    USE_GUI = False
    #img = cv2.imread('RedLine.bmp', cv2.IMREAD_COLOR)
    img = cv2.imread('RedLine2.bmp', cv2.IMREAD_COLOR)
    #img = cv2.imread('RedLine3.bmp', cv2.IMREAD_COLOR)
    #img = cv2.imread('RedLine4.bmp', cv2.IMREAD_COLOR)
    #img = cv2.imread('RedLine5.bmp', cv2.IMREAD_COLOR)
    #img = cv2.imread('RedLine6.bmp', cv2.IMREAD_COLOR)
    
    #img = cv2.imread('thomas.bmp', cv2.IMREAD_COLOR)
    
    
    intermediate = selectColorRead(img, (170,200,230), (180,240,250))
    binImage = threshToBinary(intermediate)
    largestContouredArea = detectContores(binImage, img)
    fallowLine(largestContouredArea, img)
import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('RedLine2.bmp', cv2.IMREAD_COLOR)
    cv2.circle(img, (400,400), 10, [255,0,0], thickness=1, lineType=8, shift=0)
    cv2.imshow('img',img) 
    
    # This displays the frame, mask  
    # and res which we created in 3 separate windows. 
    cv2.waitKey(0)
    
    # Destroys all of the HighGUI windows. 
    cv2.destroyAllWindows()    
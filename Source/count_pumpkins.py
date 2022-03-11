import cv2
import numpy as np
import os


class CountPumpkins:
    def __init__(self, img):
        
        # Inversed covariance matrix and average from "colour_variance/get_threshold.py"
        # Used to determine Mahalanobis distance for colour segmentation
        self.cov_inv = np.array([[ 0.1425055,   0.00316633, -0.00380055],
 [ 0.00316633,  0.00282081,  0.00163333],
 [-0.00380055,  0.00163333,  0.00575833]])

        self.avg = np.array([ 17.49497487, 155.94974874, 247.45477387])
        
        self.img = img
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.pixels = np.reshape(self.img_hsv, (-1, 3))
        
        self.detect()
        
        
    def detect(self):
        
        img_blur = cv2.GaussianBlur(self.img_hsv, (5, 5), 0)
        
        segmented_image = self.mahalanobis(img_blur)
        
        # Morphological filtering the image
        kernel = np.ones((5, 5), np.uint8)
        closed_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)        

        contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.show_output(closed_image, contours)
        
        
    def mahalanobis(self, img):
        # Mahalanobis based segmentation
        shape = self.pixels.shape
        diff = self.pixels - np.repeat([self.avg], shape[0], axis=0)

        mahalanobis_dist = np.sum(diff * (diff @ self.cov_inv), axis=1)
        mahalanobis_distance_image = np.reshape(mahalanobis_dist, (img.shape[0], img.shape[1]))

        _, mahalanobis_segmented = cv2.threshold(mahalanobis_distance_image, 40, 255, cv2.THRESH_BINARY_INV)
        mahalanobis_segmented = mahalanobis_segmented.astype(np.uint8)

        return mahalanobis_segmented
        
        
    def show_output(self, closed_image, contours):
        
        # Visual display
        # cv2.drawContours(image=self.img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        
        for contour in contours:
            M = cv2.moments(contour)
            
            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(self.img, (cx, cy), 8, (0, 0, 255), 2)

        
        # height, width, channels = self.img.shape 
        # dim = (600, round(600 * height/width))
        # img_resized = cv2.resize(self.img, dim, interpolation=cv2.INTER_LINEAR)
        # mah_resized = cv2.resize(closed_image, dim, interpolation=cv2.INTER_LINEAR)

        # cv2.namedWindow('Mahalanobis')
        # cv2.moveWindow('Mahalanobis', 40,30)
        # cv2.namedWindow('Contours')
        # cv2.moveWindow('Contours', 40,500)  

        # cv2.imshow('Mahalanobis', mah_resized)
        # cv2.imshow('Contours', img_resized)
        # cv2.waitKey(0)

        print("Number of detected pumpkins: %d" % len(contours))
        cv2.imwrite("output/located-objects.jpg", self.img)



if __name__ == '__main__':
    
    path = os.getcwd() + "/../2019-03-19 Images for third miniproject/"
    filename = path + "EB-02-660_0595_0007.JPG"
    # filename = "../Pumpkin_Field_Ortho.tif"

    image = cv2.imread(filename)

    pumpkins = CountPumpkins(image)


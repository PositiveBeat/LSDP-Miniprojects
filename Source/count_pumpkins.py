import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CountPumpkins:
    def __init__(self, img):
        
        # Inversed covariance matrix and average from "colour_variance/get_threshold.py"
        # Used to determine Mahalanobis distance for colour segmentation
        self.cov_inv = np.array([[0.1425055, 0.00316633, -0.00380055],
                                 [0.00316633, 0.00282081, 0.00163333],
                                 [-0.00380055, 0.00163333, 0.00575833]])

        self.avg = np.array([17.49497487, 155.94974874, 247.45477387])
        
        self.img = img
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.pixels = np.reshape(self.img_hsv, (-1, 3))
        
        self.detect()
        
        
    def detect(self):
        
        img_blur = cv2.GaussianBlur(self.img_hsv, (7, 7), 0)
        
        segmented_image = self.mahalanobis(img_blur)
        cv2.imwrite("output/segmented.jpg", segmented_image)
        
        # Morphological filtering the image
        kernel_cls = np.ones((11, 11), np.uint8)
        kernel_opn = np.ones((7, 7), np.uint8)
        morp_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel_cls)
        morp_image = cv2.morphologyEx(morp_image, cv2.MORPH_OPEN, kernel_opn)
        cv2.imwrite("output/morp.jpg", morp_image)     

        contours, hierarchy = cv2.findContours(morp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # contours = self.circlepass_filter(contours)
        contours = self.remove_outliers(contours)
        
        self.show_output(morp_image, contours)


    def mahalanobis(self, img):
        # Mahalanobis based segmentation
        shape = self.pixels.shape
        diff = self.pixels - np.repeat([self.avg], shape[0], axis=0)

        mahalanobis_dist = np.sum(diff * (diff @ self.cov_inv), axis=1)
        mahalanobis_distance_image = np.reshape(mahalanobis_dist, (img.shape[0], img.shape[1]))

        _, mahalanobis_segmented = cv2.threshold(mahalanobis_distance_image, 40, 255, cv2.THRESH_BINARY_INV)
        mahalanobis_segmented = mahalanobis_segmented.astype(np.uint8)

        return mahalanobis_segmented


    def remove_outliers(self, contours):
        
        good_contours = []
        contour_size = []
        
        for contour in contours:
            size = np.sqrt(cv2.contourArea(contour))
            if (size != 0):
                contour_size.append(size)
        
        # plt.hist(contour_size, range=(0,40), bins=500)
        # plt.show()
        
        for i, size in enumerate(contour_size):
            if (size > 1):
                good_contours.append(contours[i])

        return good_contours


    def circlepass_filter(self, contours):
        
        circles = []
        thres = 0.2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            circle_likeness = (2*np.pi * area) / (perimeter**2 + 0.00001)
            
            if (circle_likeness > thres):
                circles.append(contour)
        
        return circles


    
    def show_output(self, closed_image, contours):
        
        # Visual display
        cv2.drawContours(image=self.img, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        
        # for contour in contours:
        #     M = cv2.moments(contour)
            
        #     if M["m00"] != 0:
        #         cx = int(M['m10'] / M['m00'])
        #         cy = int(M['m01'] / M['m00'])
        #         cv2.circle(self.img, (cx, cy), 8, (0, 0, 255), 2)

        print("Number of detected pumpkins: %d" % len(contours))
        cv2.imwrite("output/located-objects.jpg", self.img)



if __name__ == '__main__':
    
    path = "../2019-03-19 Images for third miniproject/"
    filename = path + "EB-02-660_0595_0007.JPG"
    # filename = "../Pumpkin_Field_Ortho.tif"
    
    # filename = "input/zoomed_original.jpg"

    image = cv2.imread(filename)
    
    pumpkins = CountPumpkins(image)

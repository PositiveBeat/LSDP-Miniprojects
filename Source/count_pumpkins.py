import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class CountPumpkins:
    def __init__(self, img, debug = False):
        
        self.debug = debug
        
        # Inversed covariance matrix and average from "colour_variance/get_threshold.py"
        # Used to determine Mahalanobis distance for colour segmentation
        self.cov_inv = np.array([[0.1425055, 0.00316633, -0.00380055],
                                 [0.00316633, 0.00282081, 0.00163333],
                                 [-0.00380055, 0.00163333, 0.00575833]])

        self.avg = np.array([17.49497487, 155.94974874, 247.45477387])
        
        self.img = img
        self.img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.pixels = np.reshape(self.img_hsv, (-1, 3))
        
        self.pumpkin_count = self.detect()
        
        

    def detect(self):
        
        img_blur = cv2.GaussianBlur(self.img_hsv, (7, 7), 0)
        
        segmented_image = self.mahalanobis(img_blur)
        
        # Morphological filtering the image
        kernel_cls = np.ones((9, 9), np.uint8)
        kernel_opn = np.ones((5, 5), np.uint8)
        morp_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel_cls)
        morp_image = cv2.morphologyEx(morp_image, cv2.MORPH_OPEN, kernel_opn)

        contours, hierarchy = cv2.findContours(morp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        image_copy = self.img.copy()
        
        contours = self.remove_outliers(contours)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        contours, clusters, size_avg = self.circlepass_filter(contours)
        extra_count = self.count_in_cluster(clusters, size_avg, image_copy)   
        pumpkin_count = len(contours) + extra_count
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        
        
        if (self.debug):
            self.show_output(image_copy, "contours.jpg", contours, (0, 0, 255))
            cv2.imwrite("output/segmented.jpg", segmented_image)
            cv2.imwrite("output/morp.jpg", morp_image)     
            print("Number of detected pumpkins: %d" % pumpkin_count)
        
        
        return pumpkin_count


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
        thres = 4.5
        
        
        for contour in contours:
            size = np.sqrt(cv2.contourArea(contour))
            if (size != 0):
                contour_size.append(size)
        
        # plt.hist(contour_size, range=(0,40), bins=500)
        # plt.show()
        
        for i, size in enumerate(contour_size):
            if (size > thres):
                good_contours.append(contours[i])

        return good_contours


    def circlepass_filter(self, contours):
        
        circles = []
        clusters = []
        size_avg = 0
        thres = 0.75
        
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            circle_likeness = (4*np.pi * area) / (perimeter**2 + 0.00001)
            
            if (circle_likeness > thres):
                circles.append(contour)
                size_avg += np.sqrt(area)
            else:
                clusters.append(contour)
                
        size_avg /= len(circles)
        
        return circles, clusters, size_avg


    def count_in_cluster(self, clusters, size_avg, image):
        
        pumpkin_cnt = 0
        
        for cluster in clusters:
            area = np.sqrt(cv2.contourArea(cluster))
            extra_cnt = 0
            
            if (area % size_avg >= 1):
                extra_cnt = round(area / size_avg)
            else:
                extra_cnt = 1
            
            # Draw number on top of image
            M = cv2.moments(cluster)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(image, str(extra_cnt),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, .7,(0,0,0),2,cv2.LINE_AA)
            
            pumpkin_cnt += extra_cnt
        
        return pumpkin_cnt


    def show_output(self, image, name, contours, colour):
        
        # Visual display
        # cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=colour, thickness=1, lineType=cv2.LINE_AA)
        
        # for contour in contours:
        #     M = cv2.moments(contour)
            
        #     if M["m00"] != 0:
        #         cx = int(M['m10'] / M['m00'])
        #         cy = int(M['m01'] / M['m00'])
        #         cv2.circle(image, (cx, cy), 8, (0, 0, 255), 2)

        cv2.imwrite("output/" + name, image)


if __name__ == '__main__':
    
    # path = "../2019-03-19 Images for third miniproject/"
    # filename = path + "EB-02-660_0595_0007.JPG"
    # filename = "../Pumpkin_Field_Clearer.tif"
    filename = "input/zoomed_original.jpg"


    image = cv2.imread(filename)
    pumpkins = CountPumpkins(image, debug = True)
    
    print("Number of detected pumpkins: %d" % pumpkins.pumpkin_count)

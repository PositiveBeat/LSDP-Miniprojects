import cv2
import numpy as np



def count_pumpkins(img):
    
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    segmented_image = cv2.inRange(img_hsv, (30, 50, 30), (80, 185, 155))
    cv2.imwrite("output/hsv_segmented.jpg", segmented_image)

    # Morphological filtering the image
    kernel = np.ones((13, 13), np.uint8)
    closed_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("output/closed.jpg", closed_image)

    # Locate contours.
    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a circle above the center of each of the detected contours.
    for contour in contours:
        M = cv2.moments(contour)
        
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), 2)


    print("Number of detected pumpkins: %d" % len(contours))

    cv2.imwrite("output/located-objects.jpg", img)




if __name__ == '__main__':
    path = "../2019-03-19 Images for third miniproject/"
    filename = path + "EB-02-660_0595_0007.JPG"

    img = cv2.imread(filename)

    count_pumpkins(img)

import cv2
import numpy as np


def get_camera_matrix():
    K = np.array([[2676.1051390718389, 0, -35.243952918157035], 
         [0, 2676.1051390718389, -279.58562078697361], 
         [0, 0, 1]])
    
    return K


# Detect and show sift features
img_image1 = cv2.imread("input/frames/frame0.jpg")
img_image2 = cv2.imread("input/frames/frame50.jpg")

gray_image1 = cv2.cvtColor(img_image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(img_image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp_image1, des_image1 = sift.detectAndCompute(gray_image1, None)
kp_image2, des_image2 = sift.detectAndCompute(gray_image2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(des_image1, des_image2)

points1_temp = []
points2_temp = []
match_indices_temp = []

for idx, m in enumerate(matches):
    points1_temp.append(kp_image1[m.queryIdx].pt)
    points2_temp.append(kp_image2[m.trainIdx].pt)
    match_indices_temp.append(idx)

points1 = np.float32(points1_temp)
points2 = np.float32(points2_temp)
# match_indices = np.int32(match_indices_temp)
ransacReprojecThreshold = 1
confidence = 0.99
cameraMatrix = get_camera_matrix()
        
# Remember that points1 and point2 should be floats.
essentialMatrix, mask = cv2.findEssentialMat(
        points1, 
        points2, 
        cameraMatrix,
        cv2.FM_RANSAC, 
        confidence,
        ransacReprojecThreshold) 

print(essentialMatrix)


# # Show matches    
# img3 = cv2.drawMatches(img_image1, kp_image1,
#         img_image2, kp_image2,
#         matches, None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv2.imwrite("output/matching_features.png", img3)


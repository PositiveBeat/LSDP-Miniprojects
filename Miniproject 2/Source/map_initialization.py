import cv2
import numpy as np
import matplotlib.pyplot as plt

# from visual_slam import *


def get_camera_matrix():
    K = np.array([[2676.1051390718389, 0, -35.243952918157035], 
         [0, 2676.1051390718389, -279.58562078697361], 
         [0, 0, 1]])
    
    return K


def get_sift(gray_image1, gray_image2):
    
    # Detect and show sift features
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

    # Show matches    
    # img3 = cv2.drawMatches(img_image1, kp_image1,
    #         img_image2, kp_image2,
    #         matches, None,
    #         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    # img1_annotated = cv2.drawKeypoints(img_image1, kp_image1, img_image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2_annotated = cv2.drawKeypoints(img_image2, kp_image2, img_image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("output/Sift1.png", img1_annotated)
    # cv2.imwrite("output/Sift2.png", img2_annotated)

    return essentialMatrix, points1, points2


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    
    return img1, img2


def cal_dst_lin2pt(pts, lines):
    distances = []
    for pt, line in zip(pts, lines):
        x = pt[0]; y = pt[1]
        a = line[0]; b = line[1]; c = line[2]
        
        dst = (a*x + b*y + c) / np.sqrt(a**2 + b**2)
        distances.append(dst)
    return distances


def get_epipolar_error():
    

    essentialMatrix, pts1, pts2 = get_sift(gray_image1, gray_image2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, essentialMatrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(gray_image1, gray_image2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, essentialMatrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(gray_image2, gray_image1, lines2, pts2, pts1)


            
    dst1 = cal_dst_lin2pt(pts1, lines1)
    dst2 = cal_dst_lin2pt(pts2, lines2)


    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    print('mean: ', np.mean(dst1))
    print('std deviation: ', np.std(dst1))
    plt.hist(dst1, bins=200)
    plt.show()

    print('mean: ', np.mean(dst2))
    print('std deviation: ', np.std(dst2))
    plt.hist(dst2, bins=200)
    plt.show()


if __name__ == '__main__':
    
    img1 = cv2.imread("input/frames/frame0.jpg")
    img2 = cv2.imread("input/frames/frame50.jpg")

    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    get_epipolar_error()

    frame1 = self.list_of_frames[-2]
    frame2 = self.list_of_frames[-1]
    self.current_image_pair = ImagePair(frame1, frame2, self.bf, self.camera_matrix)
    self.current_image_pair.match_features()
    essential_matches = self.current_image_pair.determine_essential_matrix(self.current_image_pair.filtered_matches)
    self.current_image_pair.estimate_camera_movement(essential_matches)


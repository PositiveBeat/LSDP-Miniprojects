import cv2
import numpy as np

import visual_slam as vs


def get_camera_matrix():
    K = np.array([[2676.1051390718389, 0, -35.243952918157035], 
         [0, 2676.1051390718389, -279.58562078697361], 
         [0, 0, 1]])
    
    return K


if __name__ == '__main__':
    
    img1 = cv2.imread("input/frames/frame0.jpg")
    img2 = cv2.imread("input/frames/frame50.jpg")
    img3 = cv2.imread("input/frames/frame100.jpg")

    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    
    
    # Calculates relative movement
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    frameGen = vs.FrameGenerator(cv2.SIFT_create())
    frame1 = frameGen.make_frame(gray_image1)
    frame2 = frameGen.make_frame(gray_image2)
    frame3 = frameGen.make_frame(gray_image3)
    vs.current_image_pair = vs.ImagePair(frame1, frame2, bf, get_camera_matrix())
    vs.current_image_pair.match_features()
    essential_matches = vs.current_image_pair.determine_essential_matrix(vs.current_image_pair.filtered_matches)
    vs.current_image_pair.estimate_camera_movement(essential_matches)


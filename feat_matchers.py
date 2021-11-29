import cv2
import numpy as np
from skimage.measure import label, regionprops

def orb_detector(query_img_bw : np.array, img_patch : np.array, lowe_ratio : float) -> int:
    orb = cv2.ORB_create()
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw,None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(img_patch,None)
    
    # Initialize the Matcher for matching
    # the keypoints and then match the
    # keypoints

    #matcher = cv2.DescriptorMatcher_create('BruteForce-L1') #BruteForce = L2, BruteForce-L1
    matcher = cv2.BFMatcher(cv2.NORMCONV_FILTER)
    try:
        matches = matcher.knnMatch(queryDescriptors,trainDescriptors, k=2)
    except: 
        return 0
    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    # final_img = cv2.drawMatchesKnn(query_img_bw, queryKeypoints,
    # img_patch, trainKeypoints, matches[:50],None)
    try:
        good = [[m] for m, n in matches if m.distance < lowe_ratio*n.distance]
    except:
        return 0
    
    return len(good)


def brief_detector(query_img_bw : np.array, img_patch : np.array, lowe_ratio : int) -> int:
    # BRIEF is only a descriptor not finder
    # Initiate FAST detector
    finder = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints and descriptors with FAST
    kp1 = finder.detect(query_img_bw,None)
    kp2 = finder.detect(img_patch,None)
    kp1, des1 = brief.compute(query_img_bw, kp1)
    kp2, des2 = brief.compute(img_patch, kp2)

    # BFMatcher with hamming >> L2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        matches = bf.knnMatch(des1,des2, k=2)
    except:
        return 0
    # Apply ratio test
    try:
        good = [[m] for m, n in matches if m.distance < lowe_ratio*n.distance]
    except:
        return 0
    
    return len(good)


def sift_detector(query_img_bw : np.array, img_patch : np.array, lowe_ratio : int) -> int:
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query_img_bw,None)
    kp2, des2 = sift.detectAndCompute(img_patch,None)
    
    length = 0
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except:
        return 0
    # Need to draw only good matches, so create a mask

    good = []
    # ratio test as per Lowe's paper
    try:
        for m,n in (matches):
            if m.distance <lowe_ratio*n.distance:
                good.append(m)
    except:
        return 0
            
    return len(good)




def fill_bounding_boxes(z : np.array) -> tuple[list, list]:
    box_class = []
    box_coords = []
    l = label(z)
    # print(l)
    # print(l)
    # print(regionprops(l))
    x_og = z.copy()
    
    for s in regionprops(l):
        values, counts = np.unique(x_og[s.slice], return_counts=True)
        #print(np.unique(x_og[s.slice]))
        # print(values)
        # print(counts)
        ind = np.argmax(counts)
        # print(values[ind])  # prints the most frequent element
        z[s.slice] = values[ind]
        box_class.append(z[s.slice][0][0]) #homogeneous so just take any value from it
        box_coords.append(s.slice) #save the coords
        # print(s.slice)
        # print(z[s.slice])
    return box_class, box_coords
import cv2
import glob
import os
import numpy as np
from task3_helper import *

def get_matchers():
    orb_matcher = ORBMatcher()
    brief_matcher = BRIEFMatcher()
    sift_matcher = SIFTMatcher()
    matchers = [sift_matcher]
    return matchers

def test_matchers(matchers, query_img, train_img):
    for matcher in matchers:
        print("----Matching with matcher ", matcher.get_name())
        test_matcher(matcher, query_img, train_img)

def test_matcher(matcher, query_img, train_img):
    try:
        key = train_img.get_filename().split('.')[0]

        match = matcher.match(query_img, train_img)
        res_img = match.get_result()
        cv2.imshow("Res image", res_img)
        cv2.waitKey(0)

        object_img = matcher.find_object(match)
        cv2.imshow("Res image_II", object_img)
        cv2.waitKey(0)

        file_name_obj = '{folder}/{key}_obj_{query_img}.png'.format(folder=matcher.get_name().lower(), key=key, query_img=query_img.get_filename())
        if object_img is not None:
            write_result_image(file_name_obj, object_img)
    except Exception as err:
        print("exception", err)

def write_result_image(img_file, img):
    result_folder = "result_images"
    cv2.imwrite(os.path.join(result_folder, img_file), img)

def test_matching():
    #get images
    fucus_images = get_fucus_images()
    furcellaria_images = get_furcellaria_images()
    zostera_images = get_zostera_images()

    #define matchers
    matchers = get_matchers()

    #loop through query_img and lusitania images and test matchers
    for train_img in lusitania_images:
        print("-----------------------------")
        print("Beginning matching for image ", train_img.get_filename())
        #cv2.imshow("Original", img)
        for query_img in query_images:
            #cv2.imshow("Query image", query_img)
            print("--Using query_img image", query_img.get_filename())
            test_matchers(matchers, query_img, train_img)
            print("\n")
        print("-----------------------------\n")

def match_slug_images():
    #get images
    limax_images = get_limax_images()
    lusitania_images = get_lusitania_images()
    query_images = get_query_images()

def test_image_retrieving():
    lusitania_images = get_lusitania_images()
    for img_file in lusitania_images:
        img = lusitania_images[img_file]
        if img is not None:
            cv2.imshow("Img", img)

test_matching()
cv2.waitKey(0)
cv2.destroyAllWindows()


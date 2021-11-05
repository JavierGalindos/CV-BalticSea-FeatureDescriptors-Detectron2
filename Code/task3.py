import cv2
import glob
import os
import numpy as np
from image_helper import Image
from detection_methods import SIFTDetect, ORBDetect, BRIEFDetect, Detection

MIN_MATCH_COUNT = 10


# Brute force matcher that takes descriptors from two images calculated by a detector and returns if there is a match
class BFMatcher():
    def __init__(self):
        self.detect = self.init_new_detect()
        self.bf = cv2.BFMatcher(self.get_norm_type(), crossCheck=self.get_cross_check())
        print(type(self.bf))

    def get_detect(self):
        return self.detect

    def get_name(self):
        return self.detect.get_name()

    def init_new_detect(self):
        pass

    def get_norm_type(self):
        pass

    def get_cross_check(self):
        pass

    def match(self, query_image, train_image):
        """
        match descriptors in two images: query image and train image
        """
        query = self.detect.detect(query_image)
        train = self.detect.detect(train_image)

        if query.get_descriptors() is None:
            raise Exception("No query descriptors")

        if train.get_descriptors() is None:
            raise Exception("No train descriptors")

        matches = self.bf.match(query.get_descriptors(), train.get_descriptors())
        matches = sorted(matches, key=lambda x: x.distance)
        draw_params = dict(matchColor=(0, 0, 255),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        draw_image = train_image
        matches = matches[:10]
        image_with_matches = self.draw_matches(query_image.get_content(), draw_image.get_content(),
                                               query.get_keypoints(), train.get_keypoints(), matches, draw_params)
        match = Match(query, train, matches, image_with_matches)
        return match

    def draw_matches(self, query_image: np.ndarray, train_image: np.ndarray, query_points, train_points, matches,
                     draw_params=None):
        image_with_matches = cv2.drawMatches(query_image, query_points, train_image, train_points, matches, None,
                                             **draw_params)
        return image_with_matches

    def find_object(self, match):
        matches = match.get_matches()
        query = match.get_query()
        train = match.get_train()

        if len(matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([query.get_keypoints()[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([train.get_keypoints()[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = query.get_img().get_content().shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            train_image = cv2.polylines(train.get_img().get_content(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
            matchesMask = None
            train_image = train.get_img().get_content()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        return cv2.drawMatches(query.get_img().get_content(), query.get_keypoints(), train.get_img().get_content(),
                               train.get_keypoints(), matches, None, **draw_params)


class ORBMatcher(BFMatcher):
    def init_new_detect(self):
        return ORBDetect()

    def get_norm_type(self):
        return cv2.NORM_HAMMING

    def get_cross_check(self):
        return True


class BRIEFMatcher(BFMatcher):
    def init_new_detect(self):
        return BRIEFDetect()

    def get_norm_type(self):
        return cv2.NORM_HAMMING

    def get_cross_check(self):
        return True


class SIFTMatcher(BFMatcher):
    def init_new_detect(self):
        return SIFTDetect()

    def get_cross_check(self):
        return False

    def get_norm_type(self):
        return cv2.NORM_L2

    def match(self, query_image, train_image):
        # match with sift descriptors and ratio test
        query = self.detect.detect(query_image)
        train = self.detect.detect(train_image)

        matches = self.bf.knnMatch(query.get_descriptors(), train.get_descriptors(), k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        draw_image = train_image

        if len(good_matches) < 1:
            raise Exception("No good matches")

        image_with_matches = cv2.drawMatchesKnn(query_image.get_content(), query.get_keypoints(),
                                                draw_image.get_content(), train.get_keypoints(), good_matches, None,
                                                flags=2)
        match = Match(query, train, good_matches, image_with_matches)
        return match

    def find_object(self, match):
        matches = match.get_matches()
        query = match.get_query()
        train = match.get_train()

        if len(matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([query.get_keypoints()[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([train.get_keypoints()[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = query.get_img().get_content().shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            train_image = cv2.polylines(train.get_img().get_content(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
            matchesMask = None
            train_image = train.get_img().get_content()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        return cv2.drawMatchesKnn(query.get_img().get_content(), query.get_keypoints(), train.get_img().get_content(),
                                  train.get_keypoints(), matches, None, **draw_params)

    def draw_matches(self, query_image: np.ndarray, train_image: np.ndarray, query_points, train_points, matches,
                     draw_params=None):
        img = cv2.drawMatchesKnn(query_image, query_points, train_image, train_points, matches, None, **draw_params)
        return img


class Match():
    def __init__(self, query, train, matches, result):
        self.query = query
        self.train = train
        self.matches = matches
        self.result = result

    def get_query(self):
        return self.query

    def get_train(self):
        return self.train

    def get_matches(self):
        return self.matches

    def get_result(self):
        return self.result
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


# ### Task 2: Etalons
# Select a set of etalons (e.g. small images containing a sample of some distinctive features) from the
# an image to be used for matching similar objects. Aim for a set that would be representative on at
# least 50% of the images of the particular species. Think how to deal with rotations.
# ### Task 3: Baseline
# Setting a baseline: Use at least 3 different existing conventional feature detectors provided by
# OpenCV to find matches of the etalons in the image. NB! Take into account overlaps and subtract the
# appropriate numbers from total scores.

# Evaluate on two different images (called task3a.tiff and task3b.tiff) how well the approach works and
# which feature detector performs best.

import cv2
import glob
import os
import numpy as np

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
        
#Base class for other detectors to encapsulate their inner workings
class BaseDetect():
    def draw_keypoints(self, detection) -> np.ndarray:
        cv2.drawKeyPoints(detection.get_img().get_content(), detection.get_keypoints(), None, color = (0, 0, 255), flags = 0)
        return img

    def detect(self, img):
        pass

    def get_name(self):
        pass

    def get_threshold(self):
        return self.threshold

#Orb detector that uses binary string based descriptors
class ORBDetect(BaseDetect):
    def __init__(self):
        self.threshold = 2
        self.orb = cv2.ORB_create(edgeThreshold=self.get_threshold())

    def detect(self, img):
        keypoints, descriptors = self.orb.detectAndCompute(img.get_content(), None)
        detection = Detection(img, keypoints, descriptors, self.get_name())
        return detection

    def get_norm_type(self):
        return cv2.NORM_HAMMING

    def get_name(self):
        return "ORB"


class Detection():
    def __init__(self, img, keypoints, descriptors, detector):
        self.img = img
        self.descriptors = descriptors
        self.keypoints = keypoints
        self.detector = detector

    def get_img(self):
        return self.img

    def get_descriptors(self):
        return self.descriptors

    def get_keypoints(self):
        return self.keypoints

    def get_detector(self):
        return self.detector
        
class Image():
    def __init__(self, file_name, content):
        self.file_name = file_name
        self.content = content

    def get_filename(self):
        return self.file_name

    def get_content(self):
        return self.content

def get_fucus_images():
    cwd = os.getcwd()
    fucus_img_folder = 'Processed images/Fucus'
    os.chdir(os.path.join(cwd, fucus_img_folder))
    images = []
    try:
        for file in os.listdir():
            img = cv2.imread(file)
            images.append(Image(file, img))
    except:
        print("Exception occurred in get_fucus_images()")
    os.chdir(cwd)
    return images

def get_furcellaria_images():
    cwd = os.getcwd()
    furcellaria_img_folder = 'Processed images/Furcellaria lumbricalis'
    os.chdir(os.path.join(cwd, furcellaria_img_folder))
    images = []
    try:
        for file in os.listdir():
            img = cv2.imread(file)
            images.append(Image(file, img))
    except:
        print("Exception occurred in get_furcellaria_images()")
    os.chdir(cwd)
    return images

def get_zostera_images():
    cwd = os.getcwd()
    zostera_img_folder = 'Processed images/Zostera marina'
    os.chdir(os.path.join(cwd, zostera_img_folder))
    images = []
    try:
        for file in os.listdir():
            img = cv2.imread(file)
            images.append(Image(file, img))
    except:
        print("Exception occurred in get_zostera_images()")
    os.chdir(cwd)
    return images



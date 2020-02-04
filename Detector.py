# Imports

""" Global """
import numpy as np
import nmslib
from tqdm import tqdm
from glob import glob
import os
import json

""" Local """
import constants
import utils

# Detector

class Detector(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = sorted(catalog_images_paths)
        self.get_params(params)
        self.config_matcher()

    def get_params(self, params):
        self.feature_extractor = params.get("feature_extractor", constants.DEFAULT_FEATURE_EXTRACTOR)
        self.catalog_image_widths = params.get("catalog_image_widths", constants.DEFAULT_DETECTOR_CATALOG_IMAGE_WIDTHS)
        self.query_image_width = params.get("query_image_width", constants.DEFAULT_DETECTOR_QUERY_IMAGE_WIDTH)
        self.keypoint_stride = params.get("keypoint_stride", constants.DEFAULT_DETECTOR_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get("keypoint_sizes", constants.DEFAULT_DETECTOR_KEYPOINT_SIZES)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.matcher_path = params.get("matcher_path", constants.DEFAULT_DETECTOR_MATCHER_PATH)
        self.force_matcher_compute = params.get("force_matcher_compute", constants.DEFAULT_DETECTOR_FORCE_MATCHER_COMPUTE)
        self.k_nn = params.get("k_nn", constants.DEFAULT_DETECTOR_K_NN)
        self.score_sigma = params.get("sigma", constants.DEFAULT_DETECTOR_SCORE_SIGMA)
        self.verbose = params.get("verbose", constants.VERBOSE)

    def config_matcher(self):
        self.matcher = nmslib.init(method="hnsw", space="l2")
        if not self.force_matcher_compute and os.path.exists(self.matcher_path):
            self.matcher.loadIndex(self.matcher_path)
            if self.verbose:
                print("Index loaded !")
        else:
            self.get_catalog_data()
            if self.verbose:
                print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_data["descriptors"])
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            self.matcher.saveIndex(self.matcher_path)
    
    def get_catalog_data(self):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator)
        
        self.catalog_data = {}
        for i, catalog_path in enumerate(iterator):
            for width in self.catalog_image_widths:
                img = utils.read_image(catalog_path, width=width)
                keypoints = utils.get_keypoints(img, self.keypoint_stride, self.keypoint_sizes)
                descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
                self.catalog_data["keypoints"] += list(keypoints)
                self.catalog_data["descriptors"] += list(descriptors)
                self.catalog_data["labels"] += [i] * len(keypoints)
                self.catalog_data["shapes"] += [img.shape[:2]] * len(keypoints)
        
        self.catalog_data["descriptors"] = np.array(self.catalog_data["descriptors"])

# Testing

if __name__ == "__main__":
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    query_images_paths = glob(constants.DETECTOR_QUERY_IMAGES_PATH)
    detector = Detector(catalog_images_paths)
# Imports

""" Global """
import numpy as np
import nmslib
from tqdm import tqdm
from glob import glob

""" Local """
import constants
import utils

# Classifier

class Classifier(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = catalog_images_paths
        self.get_params(params)
        self.get_catalog_descriptors()
        self.config_matcher()
    
    def get_params(self, params):
        self.feature_extractor = params.get("feature_extractor", constants.DEFAULT_FEATURE_EXTRACTOR)
        self.image_size = params.get("image_size", constants.DEFAULT_CLASSIFER_IMAGE_SIZE)
        self.keypoint_stride = params.get("keypoint_stride", constants.DEFAULT_CLASSIFIER_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get("keypoint_sizes", constants.DEFAULT_CLASSIFIER_KEYPOINT_SIZES)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.verbose = params.get("verbose", constants.VERBOSE)
    
    def get_catalog_descriptors(self):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator)

        self.catalog_descriptors = []
        for path in iterator:
            img = utils.read_image(path, size=self.image_size)
            keypoints = utils.get_keypoints(img, self.keypoint_stride, self.keypoint_sizes)
            descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
            self.catalog_descriptors.append(descriptors)
        
        self.catalog_descriptors = np.array(self.catalog_descriptors)
        self.nb_descriptors_per_image = self.catalog_descriptors.shape[1]
        self.catalog_descriptors = self.catalog_descriptors.reshape(-1, self.catalog_descriptors.shape[-1])
    
    def config_matcher(self):
        if self.verbose:
            print("Creating index...")
        
        self.matcher_index = nmslib.init(method="hnsw", space="l2")
        self.matcher_index.addDataPointBatch(self.catalog_descriptors)
        self.matcher_index.createIndex(self.matcher_index_params, print_progress=self.verbose)
        self.matcher_index.setQueryTimeParams(self.matcher_query_params)

# Testing

if __name__ == "__main__":
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    clf = Classifier(catalog_images_paths)
# Imports

import cv2

# Constants

# <-- FEATURE EXTRACTOR --> #
DEFAULT_FEATURE_EXTRACTOR = cv2.xfeatures2d.SURF_create(extended=True)

# <-- MATCHER --> #
DEFAULT_MATCHER_INDEX_PARAMS = {'M': 15, 'indexThreadQty': 8, 'efConstruction': 100, 'post': 0}
DEFAULT_MATCHER_QUERY_PARAMS = {'efSearch': 100}

# <-- CLASSIFIER --> #
DEFAULT_CLASSIFER_IMAGE_SIZE = 256
DEFAULT_CLASSIFIER_KEYPOINT_STRIDE = 8
DEFAULT_CLASSIFIER_KEYPOINT_SIZES = [12, 24, 32, 48, 64]

# <-- OTHER --> #
CATALOG_IMAGES_PATH = "../Images/Catalog/*"
VERBOSE = True
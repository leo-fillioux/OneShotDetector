###########
# Imports #
###########

""" Global """
import cv2
import numpy as np
import nmslib
from tqdm import tqdm
from glob import glob
import os
import json
from sklearn.cluster import DBSCAN
from easydict import EasyDict as edict

""" Local """
from Classifier import Classifier
import constants
import utils

############
# Detector #
############

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
            self.get_catalog_data(compute_descriptors=False)
            if self.verbose: print("Loading index...")
            self.matcher.loadIndex(self.matcher_path)
            if self.verbose: print("Index loaded !")
        else:
            self.get_catalog_data()
            if self.verbose: print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_data["descriptors"])
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            if self.verbose: print("Index created !")
            
            if self.verbose: print("Saving index...")
            self.matcher.saveIndex(self.matcher_path)
            if self.verbose: print("Index saved !")
    
    def get_catalog_data(self, compute_descriptors=True):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator, desc="Get catalog data")
        
        self.catalog_data = {
            "keypoints": [],
            "descriptors": [],
            "labels": [],
            "shapes": [],
        }
        for catalog_path in iterator:
            for width in self.catalog_image_widths:
                img = utils.read_image(catalog_path, width=width)
                label = catalog_path.split("/")[-1][:-4]
                keypoints = utils.get_keypoints(img, self.keypoint_stride, self.keypoint_sizes)
                self.catalog_data["keypoints"] += list(keypoints)
                self.catalog_data["labels"] += [label] * len(keypoints)
                self.catalog_data["shapes"] += [img.shape[:2]] * len(keypoints)
                if compute_descriptors:
                    descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
                    self.catalog_data["descriptors"] += list(descriptors)
        
        self.catalog_data["descriptors"] = np.array(self.catalog_data["descriptors"])

    def predict_query(self, query_path, classifier=None):
        # Read img
        query_img = utils.read_image(query_path, width=self.query_image_width)

        # Get keypoints
        query_keypoints = utils.get_keypoints(query_img, self.keypoint_stride, self.keypoint_sizes)
        query_kpts_data = np.array([utils.keypoint2data(kpt) for kpt in query_keypoints])
        
        # Get descriptors
        if self.verbose: print("Query description...")
        query_descriptors = utils.get_descriptors(query_img, query_keypoints, self.feature_extractor)

        # Matching
        self.get_matches_results(query_kpts_data, query_descriptors, query_img.shape)

        # Get bboxes
        bboxes = self.get_raw_bboxes(query_kpts_data)
        print(sum([len(bboxes[key]) for key in bboxes]))
        bboxes = self.filter_bboxes(bboxes, query_img.shape)
        print(sum([len(bboxes[key]) for key in bboxes]))
        bboxes = self.merge_bboxes(bboxes, query_img.shape)
        print(len(bboxes))
        if classifier is not None: bboxes = self.add_classifier_score(bboxes, query_img, classifier)
        print(len(bboxes))

        return bboxes
    
    def get_matches_results(self, query_kpts_data, query_descriptors, query_shape):
        # Matching
        if self.verbose: print("Query matching...")
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn)
        
        # Result
        trainIds = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])
        scores = np.exp(-(distances / self.score_sigma) ** 2)

        # Updating keypoints
        for i, kpt in enumerate(query_kpts_data):
            kpt.traindId = trainIds[i][0]
            kpt.score = scores[i]
            kpt.query_shape = np.array(query_shape[:2])
            kpt.catalog_pt = np.array(self.catalog_data["keypoints"][kpt.traindId].pt)
            kpt.catalog_shape = np.array(self.catalog_data["shapes"][kpt.traindId][:2])
            kpt.label = self.catalog_data["labels"][kpt.traindId]
    
    def get_raw_bboxes(self, query_kpts_data):
        iterator = query_kpts_data
        if self.verbose: iterator = tqdm(iterator, desc="Raw bboxes")

        bboxes = {}
        for kpt in iterator:
            query_coord = kpt.query_pt - kpt.query_shape / 2.
            catalog_coord = kpt.catalog_pt - kpt.catalog_shape / 2.
            catalog_center = np.array([0, 0])
            query_center = query_coord + (catalog_center - catalog_coord)
            bbox = edict({
                "kpt": kpt, "score": kpt.score,
                "feature": np.array([query_center, kpt.catalog_shape]).flatten(),
            })
            if kpt.label in bboxes: bboxes[kpt.label].append(bbox)
            else: bboxes[kpt.label] = [bbox]
        return bboxes

    def filter_bboxes(self, bboxes, query_shape):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Filtering bboxes")
        
        filtered_bboxes = {}
        for label in iterator:
            label_bboxes = np.array(bboxes[label])
            bbox_features = np.array([bbox.feature for bbox in label_bboxes])
            clusters = DBSCAN(min_samples=4).fit_predict(bbox_features)
            for k in set(clusters):
                if k != -1:
                    keypoints = np.array([bbox.kpt for bbox in label_bboxes[clusters == k]])
                    bbox = utils.find_bbox_from_keypoints(keypoints)
                    if bbox is not None:
                        if label in filtered_bboxes: filtered_bboxes[label].append(bbox)
                        else: filtered_bboxes[label] = [bbox]
            filtered_bboxes[label] = utils.apply_custom_nms(filtered_bboxes.get(label, []))
        
        return filtered_bboxes

    def merge_bboxes(self, bboxes, query_shape):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Merging bboxes")

        merged_bboxes = []
        for label in iterator:
            for bbox in bboxes[label]:
                bbox["label"] = label
                merged_bboxes.append(bbox)
        
        merged_bboxes = utils.apply_custom_nms(merged_bboxes, threshold=0.2)
        for bbox in merged_bboxes:
            xmin, ymin, xmax, ymax = bbox["coords"]
            xmin = max(xmin, 0); ymin = max(ymin, 0)
            xmax = min(xmax, query_shape[1] - 1)
            ymax = min(ymax, query_shape[0] - 1)
            bbox["coords"] = [xmin, ymin, xmax, ymax]
        
        return merged_bboxes

    def add_classifier_score(self, bboxes, query_img, classifier):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Classifier score")

        for bbox in iterator:
            xmin, ymin, xmax, ymax = map(int, bbox["coords"])
            if xmin < xmax and ymin < ymax:
                crop_img = query_img[ymin:ymax, xmin:xmax]
                label, score = classifier.predict_query(crop_img)
                bbox["score"] = np.sqrt(score * bbox["score"]) if label == bbox["label"] else 0
            else:
                bbox["score"] = 0
        
        bboxes = list(filter(lambda bbox: bbox["score"] > 0, bboxes))
        
        return bboxes

########
# Main #
########

if __name__ == "__main__":
    from time import time
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    query_images_paths = glob(constants.DETECTOR_QUERY_IMAGES_PATH)
    query_path = np.random.choice(query_images_paths)

    detector = Detector(catalog_images_paths)
    classifier = Classifier(catalog_images_paths)
    
    tmp = time()
    bboxes = detector.predict_query(query_path, classifier=classifier)
    print("Done in {}s".format(time() - tmp))
    print(query_path, len(bboxes))

    query_img = utils.read_image(query_path, width=detector.query_image_width)
    boxes_img = utils.draw_bboxes(query_img, bboxes)
    cv2.imwrite("./tmp.jpg", boxes_img)
    
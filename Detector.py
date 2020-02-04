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
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
        self.n_ransac = params.get("n_ransac", constants.DEFAULT_DETECTOR_N_RANSAC)
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
        query_img = utils.read_image(query_path, width=self.query_image_width)
        query_keypoints = utils.get_keypoints(query_img, self.keypoint_stride, self.keypoint_sizes)
        if self.verbose: print("Query description...")
        query_descriptors = utils.get_descriptors(query_img, query_keypoints, self.feature_extractor)
        bboxes = self.get_raw_bboxes(query_keypoints, query_descriptors)
        bboxes = self.filter_bboxes(bboxes)
        bboxes = self.merge_bboxes(bboxes, query_img.shape)
        if classifier is not None: bboxes = self.add_classifier_score(bboxes, query_img, classifier)
        return bboxes
    
    def get_raw_bboxes(self, query_keypoints, query_descriptors):
        if self.verbose: print("Query matching...")
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn)
        trainIdx = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])
        scores = np.exp(-(distances / self.score_sigma) ** 2)

        iterator = trainIdx
        if self.verbose: iterator = tqdm(iterator, desc="Raw bboxes")

        bboxes = {}
        for ind, trainIds in enumerate(iterator):
            for k, idx in enumerate(trainIds):
                label = self.catalog_data["labels"][idx]
                x_query, y_query = query_keypoints[ind].pt
                x_catal, y_catal = self.catalog_data["keypoints"][idx].pt
                h_catal, w_catal = self.catalog_data["shapes"][idx]
                x_center = x_query + (w_catal / 2. - x_catal)
                y_center = y_query + (h_catal / 2. - y_catal)
                bbox = {
                    "score": scores[ind, k],
                    "feat": np.array([x_center, y_center, h_catal, w_catal]),
                    "raw_coords": np.array([x_query, y_query, x_catal, y_catal, h_catal, w_catal]),
                }
                if label in bboxes: bboxes[label].append(bbox)
                else: bboxes[label] = [bbox]
        return bboxes

    def filter_bboxes(self, bboxes):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Filtering bboxes")
        
        filtered_bboxes = {}
        for label in iterator:
            current_bboxes = np.array(bboxes[label])
            labels = DBSCAN(eps=50, min_samples=2).fit_predict(np.array([bbox["feat"] for bbox in current_bboxes]))
            for k in set(labels):
                if k != -1:
                    coords = [bbox["raw_coords"] for bbox in current_bboxes[labels == k]]
                    bbox_coords = []

                    for _ in range(self.n_ransac):
                        ind1, ind2 = np.random.choice(range(len(coords)), size=2, replace=False)
                        coords1, coords2 = coords[ind1], coords[ind2]
                        x_query1, y_query1, x_catal1, y_catal1, h_catal1, w_catal1 = coords1
                        x_query2, y_query2, x_catal2, y_catal2, h_catal2, w_catal2 = coords2
                        norm1 = np.linalg.norm(np.array([x_query2 - x_query1, y_query2 - y_query1]))
                        norm2 = np.linalg.norm(np.array([x_catal2 - x_catal1, y_catal2 - y_catal1]))
                        if norm1 != 0 and norm2 != 0:
                            xmin1, ymin1 = np.array([x_query1, y_query1]) - norm1/norm2 * np.array([x_catal1, y_catal1])
                            xmin2, ymin2 = np.array([x_query2, y_query2]) - norm1/norm2 * np.array([x_catal2, y_catal2])
                            xmax1, ymax1 = np.array([x_query1, y_query1]) + norm1/norm2 * np.array([w_catal1 - x_catal1, h_catal1 - y_catal1])
                            xmax2, ymax2 = np.array([x_query2, y_query2]) + norm1/norm2 * np.array([w_catal2 - x_catal2, h_catal2 - y_catal2])
                            bbox_coords.append([xmin1, ymin1, xmax1, ymax1])
                            bbox_coords.append([xmin2, ymin2, xmax2, ymax2])

                    if len(bbox_coords) > 0:
                        bbox_coords = np.mean(bbox_coords, axis=0)
                        combined_box = {
                            "score": np.sum([bbox["score"] for bbox in current_bboxes[labels == k]]),
                            "coords": bbox_coords,
                        }
                        if label in filtered_bboxes: filtered_bboxes[label].append(combined_box)
                        else: filtered_bboxes[label] = [combined_box]

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
            crop_img = query_img[ymin:ymax, xmin:xmax]
            label, score = classifier.predict_query(crop_img)
            bbox["score"] = np.sqrt(score * bbox["score"]) if label == bbox["label"] else 0
        
        bboxes = list(filter(lambda bbox: bbox["score"] > 0, bboxes))
        
        return bboxes

########
# Main #
########

if __name__ == "__main__":
    from time import time
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    query_images_paths = glob(constants.DETECTOR_QUERY_IMAGES_PATH)
    query_path = "../Images/Query Raw/JPEGImages/IMG_20190621_161004.jpg" #np.random.choice(query_images_paths)

    detector = Detector(catalog_images_paths)
    classifier = Classifier(catalog_images_paths)
    
    tmp = time()
    bboxes = detector.predict_query(query_path, classifier=classifier)
    print("Done in {}s".format(time() - tmp))
    print(query_path, len(bboxes))

    query_img = utils.read_image(query_path, width=detector.query_image_width)
    boxes_img = utils.draw_bboxes(query_img, bboxes)
    cv2.imwrite("./tmp.jpg", boxes_img)
    
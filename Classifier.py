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

""" Local """
import constants
import utils

##############
# Classifier #
##############

class Classifier(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = sorted(catalog_images_paths)
        self.get_params(params)
        self.config_matcher()
    
    def get_params(self, params):
        self.feature_extractor = params.get("feature_extractor", constants.DEFAULT_FEATURE_EXTRACTOR)
        self.image_size = params.get("image_size", constants.DEFAULT_CLASSIFER_IMAGE_SIZE)
        self.keypoint_stride = params.get("keypoint_stride", constants.DEFAULT_CLASSIFIER_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get("keypoint_sizes", constants.DEFAULT_CLASSIFIER_KEYPOINT_SIZES)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.matcher_path = params.get("matcher_path", constants.DEFAULT_CLASSIFIER_MATCHER_PATH)
        self.force_matcher_compute = params.get("force_matcher_compute", constants.DEFAULT_CLASSIFIER_FORCE_MATCHER_COMPUTE)
        self.k_nn = params.get("k_nn", constants.DEFAULT_CLASSIFIER_K_NN)
        self.score_sigma = params.get("sigma", constants.DEFAULT_CLASSIFIER_SCORE_SIGMA)
        self.verbose = params.get("verbose", constants.VERBOSE)
    
    def config_matcher(self):
        self.matcher = nmslib.init(method="hnsw", space="l2")
        if not self.force_matcher_compute and os.path.exists(self.matcher_path):
            if self.verbose: print("Loading index...")
            self.matcher.loadIndex(self.matcher_path)
            if self.verbose: print("Index loaded !")
        else:
            self.get_catalog_descriptors()
            if self.verbose: print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_descriptors)
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            if self.verbose: print("Index created !")

            if self.verbose: print("Saving index...")
            self.matcher.saveIndex(self.matcher_path)
            if self.verbose: print("Index saved !")

    def get_catalog_descriptors(self):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator, desc="Catalog description")

        self.catalog_descriptors = []
        for path in iterator:
            img = utils.read_image(path, size=self.image_size)
            keypoints = utils.get_keypoints(img, self.keypoint_stride, self.keypoint_sizes)
            descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
            self.catalog_descriptors.append(descriptors)
        
        self.catalog_descriptors = np.array(self.catalog_descriptors)
        self.catalog_descriptors = self.catalog_descriptors.reshape(-1, self.catalog_descriptors.shape[-1])

    def predict_query(self, query, score_threshold=None):
        if type(query) in [str, np.string_]: query_img = utils.read_image(query, size=self.image_size)
        else: query_img = cv2.resize(query, (self.image_size, self.image_size))
        query_keypoints = utils.get_keypoints(query_img, self.keypoint_stride, self.keypoint_sizes)
        query_descriptors = utils.get_descriptors(query_img, query_keypoints, self.feature_extractor)
        scores = self.get_query_scores(query_descriptors)
        label = sorted(scores.keys(), key=lambda x: -scores[x])[0]
        label_score = scores[label]
        if score_threshold is not None and label_score < score_threshold:
            label = constants.BACKGROUND_LABEL
        return label, label_score
    
    def predict_query_batch(self, query_paths, score_threshold=None):
        iterator = query_paths
        if self.verbose: iterator = tqdm(iterator, desc="Query prediction")

        results = {}
        for query_path in iterator:
            query_id = query_path.split("/")[-1]
            results[query_id] = self.predict_query(query_path, score_threshold=score_threshold)
        
        return results

    def get_query_scores(self, query_descriptors):
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn)
        trainIdx = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])
        scores_matrix = np.exp(-(distances / self.score_sigma) ** 2)
        scores = {}
        for ind, nn_trainIdx in enumerate(trainIdx):
            for k, idx in enumerate(nn_trainIdx):
                catalog_path = self.catalog_images_paths[idx // query_descriptors.shape[0]]
                catalog_label = catalog_path.split("/")[-1][:-4]
                scores[catalog_label] = scores.get(catalog_label, 0) + scores_matrix[ind, k]
        return scores

    def get_best_threshold(self, query_paths, ground_truth_path):
        with open(ground_truth_path, "r") as f: ground_truth = json.load(f)
        accuracies = []
        threshold_values = range(0, 5000)
        predictions = self.predict_query_batch(query_paths)
        for threshold in threshold_values:
            accuracies.append(self.compute_accuracy(ground_truth, predictions, threshold=threshold))
        return threshold_values[np.argmax(accuracies)], np.max(accuracies) * 100

    def compute_accuracy(self, ground_truth, predictions, threshold=None):
        nb_correct = counter = 0
        for img_id in ground_truth:
            if img_id in predictions:
                predicted_label, predicted_score = predictions[img_id]
                if threshold is not None and predicted_score < threshold: predicted_label = constants.BACKGROUND_LABEL
                if predicted_label == ground_truth[img_id]: nb_correct += 1
                counter += 1
        return 1. * nb_correct / counter

########
# Main #
########

if __name__ == "__main__":
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    query_images_paths = glob(constants.CLASSIFICATION_QUERY_IMAGES_PATH)
    # query_images_paths = np.random.choice(query_images_paths, size=200)
    ground_truth_path = "../Images/classification_data.json"
    clf = Classifier(catalog_images_paths)
    best_threshold, accuracy = clf.get_best_threshold(query_images_paths, ground_truth_path)
    print("The best threshold is: {} - For an accuracy of {} %".format(best_threshold, accuracy))
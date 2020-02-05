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
import keras
import imgaug.augmenters as iaa

""" Local """
import constants
import utils

##############
# Classifier #
##############

class ClassifierDeep(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = sorted(catalog_images_paths)
        self.get_params(params)
        self.get_model()
        # if self.force_train or not os.path.exists(self.model_path):
        #     if self.verbose: print("Training model ...")
        #     self.train()
        self.config_matcher()

    def get_params(self, params):
        self.image_size = params.get("image_size", constants.DEFAULT_CLASSIFER_DEEP_IMAGE_SIZE)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.matcher_path = params.get("matcher_path", constants.DEFAULT_CLASSIFIER_DEEP_MATCHER_PATH)
        self.force_matcher_compute = params.get("force_matcher_compute", constants.DEFAULT_CLASSIFIER_DEEP_FORCE_MATCHER_COMPUTE)
        self.k_nn = params.get("k_nn", constants.DEFAULT_CLASSIFIER_DEEP_K_NN)
        self.score_sigma = params.get("sigma", constants.DEFAULT_CLASSIFIER_DEEP_SCORE_SIGMA)
        self.margin = params.get("margin", constants.DEFAULT_CLASSIFIER_DEEP_TRIPLET_MARGIN)
        self.n_epochs = params.get("n_epochs", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_EPOCHS)
        self.batch_size = params.get("batch_size", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_BATCH_SIZE)
        self.augment_factor = params.get("augment_factor", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_AUGMENT_FACTOR)
        self.model_path = params.get("model_path", constants.DEFAULT_CLASSIFIER_DEEP_MODEL_PATH)
        self.force_train = params.get("force_train", constants.DEFAULT_CLASSIFIER_DEEP_FORCE_TRAIN)
        self.verbose = params.get("verbose", constants.VERBOSE)
    
    def get_model(self):
        # def loss_function(y_true, y_pred): return utils.batch_all_triplet_loss(y_true, y_pred, self.margin)[0]
        # def metric_function(y_true, y_pred): return utils.batch_all_triplet_loss(y_true, y_pred, self.margin)[1]
        vgg16 = keras.applications.VGG16(include_top=False)
        self.model = keras.models.Model(vgg16.input, vgg16.get_layer(name="block3_conv3").output)
        # model_input = keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        # encoder = keras.layers.GlobalMaxPool2D()(vgg16(model_input))
        # model_output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(encoder)
        # self.model = keras.models.Model(model_input, model_output)
        # self.model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss=loss_function, metrics=[metric_function])
        # if os.path.exists(self.model_path):
        #     if self.verbose: print("Loading weights ...")
        #     self.model.load_weights(self.model_path)
    
    def get_augmenter(self):
        seq = iaa.Sequential([
            iaa.SomeOf((1, 4), [
                iaa.Fliplr(0.5),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.MotionBlur(angle=(0, 360)),
                iaa.Add((-20, 20)),
                iaa.AddElementwise((-10, 10)),
                iaa.AdditiveGaussianNoise(scale=0.05*255),
                iaa.Multiply((0.5, 2)),
                iaa.SaltAndPepper(p=(0.1, 0.3)),
                iaa.JpegCompression(compression=(20, 90)),
                iaa.Grayscale((0.0, 1.0)),
                iaa.Affine(shear=(-15, 15)),
                iaa.Affine(rotate=(-10, 10)),
            ])
        ])
        return seq
    
    def get_batch(self, paths, augment=True):
        # Original
        original_images = [utils.read_image(path, size=self.image_size) for path in paths]
        original_labels = [self.catalog_images_paths.index(path) for path in paths]

        # Augmented
        if augment:
            augmenter = self.get_augmenter()
            augmented_images = [image for image in original_images for k in range(self.augment_factor)]
            augmented_labels = [label for label in original_labels for k in range(self.augment_factor)]
            augmented_images = augmenter(images=augmented_images)
        else:
            augmented_images = []
            augmented_labels = []

        # Merge
        total_images = np.array(augmented_images + original_images)
        total_labels = np.array(augmented_labels + original_labels)
        
        # Shuffle
        indexes = list(range(len(total_images)))
        np.random.shuffle(indexes)
        total_images = total_images[indexes]
        total_labels = total_labels[indexes]

        total_images = total_images[:, :, :, ::-1] / 255.0

        return total_images, total_labels

    def train(self):
        for epoch in range(self.n_epochs):
            paths = np.random.choice(self.catalog_images_paths, size=self.batch_size)
            batch_img, batch_labels = self.get_batch(paths)
            loss, metric = self.model.train_on_batch(batch_img, batch_labels)
            print("Epoch: {} | Loss = {} | Metric = {}%".format(epoch, loss, metric * 100))
            self.model.save_weights(self.model_path)

    def get_catalog_descriptors(self):
        iterator = range(0, len(self.catalog_images_paths), self.batch_size)
        if self.verbose: iterator = tqdm(iterator, desc="Catalog description")

        self.catalog_descriptors = []
        for i in iterator:
            batch_imgs, _ = self.get_batch(self.catalog_images_paths[i:i+self.batch_size], augment=False)
            descriptors = self.model.predict(batch_imgs)
            descriptors = descriptors.reshape(-1, descriptors.shape[-1])
            self.catalog_descriptors += list(descriptors)
        
        self.catalog_descriptors = np.array(self.catalog_descriptors)

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

    def predict_query(self, query, score_threshold=None):
        if type(query) in [str, np.string_]: query_img = utils.read_image(query, size=self.image_size)
        else: query_img = cv2.resize(query, (self.image_size, self.image_size))
        query_img = query_img[:, :, ::-1] / 255.0
        query_descriptors = self.model.predict(np.array([query_img]))
        query_descriptors = query_descriptors.reshape(-1, query_descriptors.shape[-1])
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
        scores_matrix = (np.max(distances) - distances) / (np.max(distances) - np.min(distances))
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
    import matplotlib.pyplot as plt
    catalog_images_paths = glob(constants.CATALOG_IMAGES_PATH)
    query_images_paths = glob(constants.CLASSIFICATION_QUERY_IMAGES_PATH)
    query_images_paths = np.random.choice(query_images_paths, size=200)
    ground_truth_path = "../Images/classification_data.json"
    clf = ClassifierDeep(catalog_images_paths)
    best_threshold, accuracy = clf.get_best_threshold(query_images_paths, ground_truth_path)
    print("The best threshold is: {} - For an accuracy of {} %".format(best_threshold, accuracy))
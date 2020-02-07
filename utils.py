###########
# Imports #
###########

""" Global """
import cv2
import imutils
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

""" Local """
import constants

#############
# Functions #
#############

def read_image(path, width=None, size=None):
    img = cv2.imread(path)
    if width is not None:
        img = imutils.resize(img, width=width)
    elif size is not None:
        img = cv2.resize(img, (size, size))
    return img

def get_keypoints(img, kpt_stride, kpt_sizes):
    return [
        cv2.KeyPoint(x, y, size)
        for size in kpt_sizes
        for x in range(0, img.shape[1], kpt_stride)
        for y in range(0, img.shape[0], kpt_stride)
    ]

def keypoint2data(keypoint):
    data = edict({})
    data.query_pt = np.array(keypoint.pt)
    data.size = keypoint.size
    return data

def get_descriptors(img, keypoints, feature_extractor):
    _, descriptors = feature_extractor.compute(img, keypoints)
    return descriptors

def apply_custom_nms(bboxes, threshold=0.5):
    if len(bboxes) == 0: return []
    
    # Init. variables
    final_bboxes = []
    x1 = np.array([bbox["coords"][0] for bbox in bboxes])
    y1 = np.array([bbox["coords"][1] for bbox in bboxes])
    x2 = np.array([bbox["coords"][2] for bbox in bboxes])
    y2 = np.array([bbox["coords"][3] for bbox in bboxes])
    score = np.array([box["score"] for box in bboxes])
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(score)

    # Loop
    while I.size != 0:
        
        last = I.size
        i = I[last - 1]
        merge = [last - 1]
        
        for pos in range(last - 1):
            j = I[pos]
            w = min(x2[i], x2[j]) - max(x1[i], x1[j]) + 1
            h = min(y2[i], y2[j]) - max(y1[i], y1[j]) + 1
            iou = w * h / area[j]
            if w > 0 and h > 0 and iou > threshold: merge.append(pos)
        
        scores = [bbox["score"] for bbox in np.array(bboxes)[I[merge]]]
        combined_box = {
            "score": np.sum(scores),
            "coords": np.average([bbox["coords"] for bbox in np.array(bboxes)[I[merge]]], axis=0, weights=scores),
        }
        
        if "label" in bboxes[0]:
            label_scores = {}
            for bbox in np.array(bboxes)[I[merge]]:
                label_scores[bbox["label"]] = label_scores.get(bbox["label"], 0) + bbox["score"]
            combined_box["label"] = sorted(label_scores.keys(), key=lambda label: label_scores[label])[-1]
        
        final_bboxes.append(combined_box)
        I = np.delete(I, merge)

    return final_bboxes

def draw_text_box_pil(im, text, x, y, rectangle_bgr=(255, 255, 255), size=10, thickness=1):
    font = ImageFont.truetype('./fonts/Arial.ttf', size)
    w, h = font.getsize(text)
    draw = ImageDraw.Draw(im)
    draw.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), fill=rectangle_bgr)
    draw.rectangle((x - 2, y - 2, x + w + 1, y + h + 1), outline=(80, 80, 80))
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

def draw_bboxes(img, bboxes, alpha=0.4):
    bbox_img = img.copy()
    labels_set = list(set([bbox["label"] for bbox in bboxes]))
    for bbox in bboxes:
        label = bbox["label"]
        xmin, ymin, xmax, ymax = map(int, bbox["coords"])
        cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), constants.COLORS[labels_set.index(label) % len(constants.COLORS)], -1)
        cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), constants.COLORS[labels_set.index(label) % len(constants.COLORS)], 2)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), constants.COLORS[labels_set.index(label) % len(constants.COLORS)], 2)
    cv2.addWeighted(bbox_img, alpha, img, 1 - alpha, 0, img)
    image_pil = Image.fromarray(img)
    for bbox in bboxes:
        label = bbox["label"]
        xmin, ymin, xmax, ymax = map(int, bbox["coords"])
        draw_text_box_pil(image_pil, label, xmin, ymin - 13,
                          rectangle_bgr=constants.COLORS[labels_set.index(label) % len(constants.COLORS)],
                          size=10, thickness=2)
    return np.array(image_pil)

def find_bbox_from_keypoints(keypoints):
    # Get coords
    query_coords = np.array([kpt.query_pt for kpt in keypoints])
    query_shapes = np.array([kpt.query_shape for kpt in keypoints])[:, ::-1]

    # Get shapes
    catalog_coords = np.array([kpt.catalog_pt for kpt in keypoints])
    catalog_shapes = np.array([kpt.catalog_shape for kpt in keypoints])[:, ::-1]

    # Put the center in the middle
    query_coords = query_coords - query_shapes / 2.
    catalog_coords = catalog_coords - catalog_shapes / 2.

    # Find homography
    H, _ = cv2.findHomography(catalog_coords, query_coords, cv2.RANSAC)
    if H is None:
        return None
    
    # Get catalog bbox coords
    catalog_bbox_coords1 = np.ones((len(catalog_coords), 3))
    catalog_bbox_coords2 = np.ones((len(catalog_coords), 3))
    catalog_bbox_coords3 = np.ones((len(catalog_coords), 3))
    catalog_bbox_coords4 = np.ones((len(catalog_coords), 3))
    catalog_bbox_coords1[:, 0] = -0.5 * catalog_shapes[:, 0]
    catalog_bbox_coords1[:, 1] = -0.5 * catalog_shapes[:, 1]
    catalog_bbox_coords2[:, 0] = +0.5 * catalog_shapes[:, 0]
    catalog_bbox_coords2[:, 1] = -0.5 * catalog_shapes[:, 1]
    catalog_bbox_coords3[:, 0] = -0.5 * catalog_shapes[:, 0]
    catalog_bbox_coords3[:, 1] = +0.5 * catalog_shapes[:, 1]
    catalog_bbox_coords4[:, 0] = +0.5 * catalog_shapes[:, 0]
    catalog_bbox_coords4[:, 1] = +0.5 * catalog_shapes[:, 1]

    # Get query bbox coords
    query_bbox_coords1 = (H.dot(catalog_bbox_coords1.T)).T
    query_bbox_coords2 = (H.dot(catalog_bbox_coords2.T)).T
    query_bbox_coords3 = (H.dot(catalog_bbox_coords3.T)).T
    query_bbox_coords4 = (H.dot(catalog_bbox_coords4.T)).T
    
    # Divide by last coord
    query_bbox_coords1 = 1. * query_bbox_coords1 / query_bbox_coords1[:, [-1]]
    query_bbox_coords2 = 1. * query_bbox_coords2 / query_bbox_coords2[:, [-1]]
    query_bbox_coords3 = 1. * query_bbox_coords3 / query_bbox_coords3[:, [-1]]
    query_bbox_coords4 = 1. * query_bbox_coords4 / query_bbox_coords4[:, [-1]]

    # Get mean
    query_bbox_coords1 = np.mean(query_bbox_coords1, axis=0)
    query_bbox_coords2 = np.mean(query_bbox_coords2, axis=0)
    query_bbox_coords3 = np.mean(query_bbox_coords3, axis=0)
    query_bbox_coords4 = np.mean(query_bbox_coords4, axis=0)

    # Get bbox coords
    xmin = int(min(query_bbox_coords1[0], query_bbox_coords2[0], query_bbox_coords3[0], query_bbox_coords4[0]) + query_shapes[0, 0] / 2.)
    xmax = int(max(query_bbox_coords1[0], query_bbox_coords2[0], query_bbox_coords3[0], query_bbox_coords4[0]) + query_shapes[0, 0] / 2.)
    ymin = int(min(query_bbox_coords1[1], query_bbox_coords2[1], query_bbox_coords3[1], query_bbox_coords4[1]) + query_shapes[0, 1] / 2.)
    ymax = int(max(query_bbox_coords1[1], query_bbox_coords2[1], query_bbox_coords3[1], query_bbox_coords4[1]) + query_shapes[0, 1] / 2.)
    if xmin >= xmax or ymin >= ymax:
        return None
    
    # Get bbox
    bbox = edict({
        "score": np.sum([kpt.score for kpt in keypoints]),
        "coords": np.array([xmin, ymin, xmax, ymax])
    })

    return bbox

def _pairwise_distances(embeddings):
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)
    return distances

def _get_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin):
    labels = tf.squeeze(labels)
    pairwise_dist = _pairwise_distances(embeddings)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)
    triplet_loss = tf.maximum(triplet_loss, 0.0)
    positive_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets


import tensorflow as tf
import numpy as np
import ops.tensor_ops as tops

EPSILON = 1e-8 # a constant used to approximate the infinity small in math

"""
Bounding box collections used by sliding window methods generally.
"""

def area(box):
    """Computes area of boxes.
    B: batch_size
    N: number of boxes
    Args:
      box: a float Tensor with [N, 4], or [B, N, 4].
    Returns:
      a float Tensor with [N], or [B, N]
    """
    with tf.name_scope('BBox/bbox_area'):
        y_min, x_min, y_max, x_max = tf.split(
            value=box, num_or_size_splits=4, axis=-1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)

def intersection(box1, box2, dtype = 'float32'):
    """
    input box1 or box2 : a float Tensor with [N, 4], or [B, N, 4]
    """
    with tf.name_scope('BBox/bbox_intersection'):
        x_min1, y_min1, x_max1, y_max1 = tf.split(value = box1, num_or_size_splits = 4, axis = -1)
        x_min2, y_min2, x_max2, y_max2 = tf.split(value = box2, num_or_size_splits = 4, axis = -1)

        boxes_rank = len(box1.shape)
        perm = [1, 0] if boxes_rank == 2 else [0, 2, 1]
        y_min_max = tf.minimum(y_max1, tf.transpose(y_max2, perm))
        y_max_min = tf.maximum(y_min1, tf.transpose(y_min2, perm))
        x_min_max = tf.minimum(x_max1, tf.transpose(x_max2, perm))
        x_max_min = tf.maximum(x_min1, tf.transpose(x_min2, perm))
       
        intersect_heights = y_min_max - y_max_min
        intersect_widths = x_min_max - x_max_min
        zeros_t = tf.cast(0, intersect_heights.dtype)
        intersect_heights = tf.maximum(zeros_t, intersect_heights)
        intersect_widths = tf.maximum(zeros_t, intersect_widths)
        return intersect_heights * intersect_widths

def iou(gt_boxes, boxes):
    """
    Computes pairwise intersection-over-union between box collections.
    Args:
      gt_boxes: a float Tensor with [N, 4].
      boxes: a float Tensor with [M, 4].
    Returns:
      a Tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope('BBox/bbox_IOU'):
        intersections = intersection(gt_boxes, boxes)
        gt_boxes_areas = area(gt_boxes)
        boxes_areas = area(boxes)
        boxes_rank = len(boxes_areas.shape)
        boxes_axis = 1 if (boxes_rank == 2) else 0
        gt_boxes_areas = tf.expand_dims(gt_boxes_areas, -1)
        boxes_areas = tf.expand_dims(boxes_areas, boxes_axis)
        unions = gt_boxes_areas + boxes_areas
        unions = unions - intersections
        return tf.where(
            tf.equal(intersections, 0.0), tf.zeros_like(intersections),
            tf.truediv(intersections, unions))

def box_iou_match(x, gt_boxes,
                    match_threshold =0.5,
                    unmatch_threshold = 0.4):
    """
    the input [N, 4], and the gt_boxes [M,4] tensor, matching each boxes to gt_boxes. The highest
    similarity of the gt_boxes associated to the box will be assigned as the label for this boxes
    if the similarity higher than the threshold.

    One gt_box maybe match to more than one boxes, the return value is indices of the matched boxes
    and the shape will be [N,1], same length as the input x
    """
    similarity = iou(gt_boxes, x)
    matches = tf.math.argmax(similarity, axis=0, output_type=tf.int32)
    max_iou = tf.math.reduce_max(input_tensor = similarity, axis = 0)
    un_matches = tf.math.greater(unmatch_threshold, max_iou)
    gray_matches = tf.math.logical_and(
        tf.greater(match_threshold, max_iou),
        tf.greater_equal(max_iou, unmatch_threshold))

    matches = tops.set_values_by_indicator(x = matches, indicator = gray_matches, value = -1)
    matches = tops.set_values_by_indicator(x = matches, indicator = un_matches,   value = -2)
    return matches

def pcoordinate_to_ccoordinate(boxes):
    """
    convert the two point two-points pixel coordinate [x0,y0, x1,y1] to center + size coordinate [xc,yc, w,h]
    input should be [N, 4] or [B, N, 4], the coordiates are in the last dimension
    """
    with tf.name_scope('BBox/converter'):
        xmin, ymin, xmax, ymax = tf.unstack(boxes, axis= -1)
        width = xmax - xmin
        height = ymax - ymin
        ycenter = ymin + height / 2.
        xcenter = xmin + width / 2.
        return [xcenter, ycenter, width, height]

def generate_target(labels, indices, unmatched_value, ignored_value):
    """
    generate the targets from the groundtruth pool based on the indices.
    For instance, if labels are the gt_box and the indices are the matched results, 
    it can generate the targets for box regression

    The label -2 for unmatched and -1 for ignored (those values between the match/unmatch threshold).
    """
    input_tensor = tf.concat(
        [tf.stack([unmatched_value, ignored_value]), labels], axis=0)
    gather_indices = tf.maximum(indices + 2, 0)
    gathered_tensor = tf.gather(input_tensor, gather_indices)
    return gathered_tensor

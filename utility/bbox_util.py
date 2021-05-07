
import tensorflow as tf
import numpy as np

def get_bbox_from_mask(mask):
    """
    return the bounding box pixel index as coordinates [xmin, xmax, ymin, ymax]
    """
    rows = np.any(mask, axis=0)
    cols = np.any(mask, axis=1)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

def box_IOU(box1, box2):
    """
    box input: x0, y0, x1, y1
    return the IOU of the two boxes
    """
    isintersect = False
    if box1[0] <= box2[0] and box1[2] >= box2[0]:
        if box1[3] < box2[1] or box1[1] > box2[3]: pass
        else: isintersect = True
    elif box1[0] >= box2[0] and box1[0] <= box2[2]:
        if box1[3] < box2[1] or box1[1] > box2[3]: pass
        else: isintersect = True

    if not isintersect: 
        return 0
   
    x = sorted([box1[0], box1[2], box2[0], box2[2]])
    w = x[2]-x[1]
    y = sorted([box1[1], box1[3], box2[1], box2[3]])
    h = y[2]-y[1]

    inter = w*h
    total = abs((box1[0]-box1[2])*(box1[1]-box1[3]))+abs((box2[0]-box2[2])*(box2[1]-box2[3]))
    iou = inter/(total-inter)
    return iou

def intersection(box1, box2):
    """
    input box1 or box2 : a float Tensor with [N, 4], or [B, N, 4]
    """
    with tf.namespace('utility/box_util_intersection'):
        x_min1, y_min1, x_max2, y_max2 = tf.split(value = box1, num_or_size_splits = 4, axis = -1)
        x_min2, y_min2, x_max2, y_max2 = tf.split(value = box1, num_or_size_splits = 4, axis = -1)

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
    """Computes pairwise intersection-over-union between box collections.
    Args:
      gt_boxes: a float Tensor with [N, 4].
      boxes: a float Tensor with [M, 4].
    Returns:
      a Tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope('utility/box_util_IOU'):
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



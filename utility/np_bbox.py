
"""
bounding box related functions depends purely on numpy library
"""

import numpy as np

def generate_bbox_from_mask(mask):
    """
    generating bounding boxes from the pixel masks of objects. The requirments of the masks are:
    1. the background should be labeled as 0 while positive integers are used to mark object. 
    2. The pixels of the same object/mask should be the same value.
    3. Input shape should be [W,H], ie. rank 2 tensor
    4. maximum integer of the mask indicates the number of objects in this mask

    Return values: [N, 4], N bboxes with coordinate [x0, y0, x1, y1] and N = np.amax(mask)
    """
    nobj = np.amax(mask)
    boxes = []
    for n in range(1, nobj+1):
        a = np.where(mask == n)
        boxes.append([np.min(a[1]), np.min(a[0]), np.max(a[1]),  np.max(a[0])])
    return boxes

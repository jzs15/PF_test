import numpy as np
from nms.nms import py_hybrid_wrapper


def remove_over(boxes, info):
    width = info['width']
    height = info['height']
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    idx = np.where(((x1 > 0) & (y1 > 0) & (x2 < width) & (y2 < height))==True)
    return boxes[idx]


def post_process(boxes, info):
    for i in range(1, 3):
        hybrid = py_hybrid_wrapper(info['thresh'][1], info['soft_thresh'], max_dets=info['max_dets'])
        boxes[i] = remove_over(boxes[i], info)
        boxes[i] = hybrid(boxes[i])

    p_x1 = boxes[1][:, 0]
    p_y1 = boxes[1][:, 1]
    p_x2 = boxes[1][:, 2]
    p_y2 = boxes[1][:, 3]
    p_a = (p_x2 - p_x1 + 1) * (p_y2 - p_y1 + 1)

    keep = []
    for i in range(len(boxes[2])):
        f_box = boxes[2][i]
        f_x1 = f_box[0]
        f_y1 = f_box[1]
        f_x2 = f_box[2]
        f_y2 = f_box[3]

        xx1 = np.maximum(f_x1, p_x1)
        yy1 = np.maximum(f_y1, p_y1)
        xx2 = np.minimum(f_x2, p_x2)
        yy2 = np.minimum(f_y2, p_y2)
        f_a = (f_x2 - f_x1 + 1) * (f_y2 - f_y1 + 1)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (p_a + f_a - inter)

        if (ovr > 0.0).any():
            keep.append(f_box)
    return boxes[1], np.array(keep)

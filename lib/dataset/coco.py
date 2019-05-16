import cPickle
import os
import numpy as np

from imdb import IMDB

# coco api
from .pycocotools.coco import COCO


def coco_results_one_category_kernel(data_pack):
    cat_id = data_pack['cat_id']
    indices = data_pack['indices']
    boxes = data_pack['boxes']
    cat_results = []
    for im_ind, index in enumerate(indices):
        dets = boxes[im_ind].astype(np.float)
        if len(dets) == 0:
            continue
        scores = dets[:, -1]
        xs = dets[:, 0]
        ys = dets[:, 1]
        ws = dets[:, 2]
        hs = dets[:, 3]
        result = [{'image_id': index,
                   'category_id': cat_id,
                   'bbox': [xs[k], ys[k], ws[k], hs[k]],
                   'score': scores[k]} for k in xrange(dets.shape[0])]
        cat_results.extend(result)
    return cat_results


class coco(IMDB):
    def __init__(self, image_set, root_path, result_path, is_test=False):
        super(coco, self).__init__('COCO', image_set, root_path, result_path)
        self.root_path = root_path
        if not is_test:
            self.coco = COCO(self._get_ann_file())
        else:
            self.test_dir = os.path.join(self.root_path, 'images', image_set)
            self.coco = COCO(None)
        self.is_test = is_test

        # deal with class names
        cats = ['pedestrian', 'face']
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_ind = {'pedestrian': 1, 'face': 2}
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.data_name = image_set

    def _get_ann_file(self):
        return os.path.join(self.root_path, 'annotations', self.image_set + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        if not self.is_test:
            image_ids = self.coco.getImgIds()
        else:
            image_ids = os.listdir(self.test_dir)
        return image_ids

    def image_path_from_index(self, index):
        filename = self.coco.imgs[index]['file_name']
        image_path = os.path.join(self.root_path, 'images', self.data_name, filename)
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        if not self.is_test:
            gt_roidb = [self._load_coco_annotation(index) for index in self.image_set_index]
        else:
            gt_roidb = [self.load_test_gt(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def get_test_file_name(self, index):
        return os.path.join(self.test_dir, index)

    def load_test_gt(self, index):
        boxes = np.zeros((1, 4), dtype=np.uint16)
        gt_classes = np.zeros((1), dtype=np.int32)
        overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
        roi_rec = {'image': self.get_test_file_name(index),
                   'height': 0,
                   'width': 0,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        return roi_rec

    def _load_coco_annotation(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ix, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            if obj['iscrowd']:
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        roi_rec = {'image': self.image_path_from_index(index),
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}
        return roi_rec

    def evaluate_detections(self, detections, result_dir='mAP'):
        self._write_coco_results(detections, result_dir)

    def _write_coco_results(self, all_boxes, result_dir='mAP'):
        result_dir = os.path.join(result_dir, 'predicted')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        for im_ind, index in enumerate(self.image_set_index):
            f = open(result_dir + '/' + index.replace('.jpg', '.txt'), 'w+')
            is_first = True
            for cls_ind, cls in enumerate(self.classes):
                boxes = all_boxes[cls_ind][im_ind]
                for box in boxes:
                    if not is_first:
                        f.write('\n')
                    is_first = False
                    f.write(cls + ' ' + str(box[4]) + ' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(
                        box[2]) + ' ' + str(box[3]))

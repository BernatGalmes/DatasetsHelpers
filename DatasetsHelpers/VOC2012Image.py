import os

import cv2
import numpy as np

from .DatasetImage import DatasetImage


class VOC2012Image(DatasetImage):

    PATH_DATASET = None

    # Save to the d2l package.
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    # Save to the d2l package.
    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird',
                   'boat', 'bottle', 'bus', 'car',
                   'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train',
                   'tv/monitor']

    def __init__(self, p_im):
        super().__init__(p_im)
        self.name_im = os.path.basename(p_im)
        self.classes = cv2.cvtColor(cv2.imread(p_im, -1), cv2.COLOR_BGR2RGB)

        self.labels = np.zeros(self.classes.shape[:2], np.uint8)
        for i, className in enumerate(VOC2012Image.VOC_CLASSES):
            mask_class = self.getClassMask(className)
            self.labels[mask_class] = i
        self.im_bgr = cv2.imread(self.PATH_DATASET + "/JPEGImages/" + self.name_im.split(".")[0] + ".jpg")
        self.isCorrect = self.im_bgr is not None

        self.im_rgb = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2RGB)
        self.im_lab = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2LAB)

    @staticmethod
    def class_names():
        return VOC2012Image.VOC_CLASSES

    @staticmethod
    def class_colors():
        return VOC2012Image.VOC_COLORMAP

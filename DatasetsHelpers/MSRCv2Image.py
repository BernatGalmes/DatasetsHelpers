import os

import cv2
import numpy as np

from .DatasetImage import DatasetImage


class MSRCv2Image(DatasetImage):

    PATH_DATASET = None

    # Save to the d2l package.
    MSRC_COLORMAP = [
        ["void", [0, 0, 0]],
        ["building", [128, 0, 0]],
        ["grass", [0, 128, 0]],
        ["tree", [128, 128, 0]],
        ["cow", [0, 0, 128]],
        # ["horse", [128, 0, 128]],
        ["sheep", [0, 128, 128]],
        ["sky", [128, 128, 128]],
        # ["mountain", [64, 0, 0]],
        ["aeroplane", [192, 0, 0]],
        ["water", [64, 128, 0]],
        ["face", [192, 128, 0]],
        ["car", [64, 0, 128]],
        ["bicycle", [192, 0, 128]],
        ["flower", [64, 128, 128]],
        ["sign", [192, 128, 128]],
        ["bird", [0, 64, 0]],
        ["book", [128, 64, 0]],
        ["chair", [0, 192, 0]],
        ["road", [128, 64, 128]],
        ["cat", [0, 192, 128]],
        ["dog", [128, 192, 128]],
        ["body", [64, 64, 0]],
        ["boat", [192, 64, 0]]
     ]

    def __init__(self, p_im):
        super().__init__(p_im)
        self.name_im = os.path.basename(p_im)
        self.im_bgr = cv2.imread(p_im)
        self.classes = cv2.imread(self.PATH_DATASET + "/GroundTruth/" +
                                  self.name_im.split(".")[0] + "_GT.bmp")
        self.classes = cv2.cvtColor(self.classes, cv2.COLOR_BGR2RGB)

        self.labels = np.zeros(self.classes.shape[:2], np.uint8)

        for i, className in enumerate(self.class_names()):
            mask_class = self.getClassMask(className)
            self.labels[mask_class] = i

        self.isCorrect = self.im_bgr is not None

        self.im_rgb = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2RGB)
        self.im_lab = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2LAB)

    @staticmethod
    def class_names():
        return [item[0] for item in MSRCv2Image.MSRC_COLORMAP]

    @staticmethod
    def class_colors():
        return [item[1] for item in MSRCv2Image.MSRC_COLORMAP]


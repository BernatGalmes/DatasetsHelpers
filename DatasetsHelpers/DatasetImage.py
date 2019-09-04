import numpy as np


class DatasetImage:

    def __init__(self, p_im):
        self.name_im = p_im
        self.classes = None

        self.labels = None

        self.im_bgr = None
        self.isCorrect = None

        self.im_rgb = None
        self.im_lab = None

    def getClassMask(self, className):
        mask = np.zeros(self.classes.shape[:2], dtype=np.bool)

        i_class = self.class_names().index(className)
        isClass = (self.classes[:, :, 0] == self.class_colors()[i_class][0]) & \
                  (self.classes[:, :, 1] == self.class_colors()[i_class][1]) & \
                  (self.classes[:, :, 2] == self.class_colors()[i_class][2])

        notBG = (self.classes[:, :, 0] != 0) | (self.classes[:, :, 1] != 0) | (self.classes[:, :, 2] != 0)
        mask[notBG & isClass] = 1
        return mask

    @staticmethod
    def class_names():
        return None

    @staticmethod
    def class_colors():
        return None

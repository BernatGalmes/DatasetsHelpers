# DatasetsHelpers
Code that helps processing images from different public datasets.

## Suported Datasets

### VOC2012

Dataset with 20 classes.
The train/val data has 11,530 images containing 27,450 ROI annotated objects
and 6,929 segmentations.

You can get more information from the
[Oficial page](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

You can download the dataset from the next link:

[Download link](https://www.kaggle.com/huanghanchina/pascal-voc-2012)

### MSRC v2

Dataset with 591 images, and 23 object classes labelled.

The dataset can be download from the next link:

[Download link](http://research.microsoft.com/vision/cambridge/recognition/)

## Functionality

### Image wrapper class

Example of object instantiation:
```python
from DatasetsHelpers.VOC2012Image import VOC2012Image

path_im = "/path/to/SegmentationClass/image"

im = VOC2012Image(path_im)
```

#### Attributes and properties

Get the name of the image:
```python
im.name_im
```

Boolean indicating if object has been loaded successfully:
```python
im.isCorrect
```

Image with the map of the labels of each pixel (RGB format):
```python
im.classes
```

Single channel image with the label of each pixel.
```python
im.labels
```

BGR image from the input color image.
```python
im.im_bgr
```

RGB image of the input color image.
```python
im.im_rgb
```

Get a list of all the label names in the same order as his mapped value.
```python
im.class_names()
```

Get a list of the RGB value of the labels in the im.classes image.
Ordered in the same order as im.class_names()
```python
im.class_colors()
```

robotics-vision - v3 2026-04-19 3:40am
==============================

This dataset was exported via roboflow.com on April 19, 2026 at 7:41 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 824 images.
Signs are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Fit (black edges))
* Grayscale (CRT phosphor)
* Auto-contrast via histogram equalization

The following augmentation was applied to create 3 versions of each source image:
* Random exposure adjustment of between -14 and +14 percent
* Salt and pepper noise was applied to 0.38 percent of pixels



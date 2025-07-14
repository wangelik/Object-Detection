# Object Detection

This repository contains different experiments and simulations of machine learning based object detection to compare algorithms, frameworks, and computation load.  
Currently, only images are supported and tested. Moreover, we only conduct inference, no finetuning or training on new data.

## Structure

The data is structured as follows:  

* `images` - folder of test images
* `output` - folder of annotated images (outputs)
* `detectors` - folder of different object detector functions, e.g., transformers or YOLO
* `detectors/utils` - folder of utility functions, e.g., bounding box or label visualizers
* `*.ipynb` - different notebooks to test and run sample detectors on test images

## Detectors

Over time, more and more detectors are tested. The current list contains:

* `Detection Transformers` within the `YOLO` and `transformers` framework [1] 

## Usage

To test and run experiments, load one of the different jupyter notebook files in the main directory.

## References

[1] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020, August). End-to-end object detection with transformers. In European Conference on Computer Vision (pp. 213–229). Springer, Cham.
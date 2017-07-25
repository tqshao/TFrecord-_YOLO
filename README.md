# TFrecord-_YOLO
This code is for creating TFrecord file that can be used to Google faster RCNN API. The orignial annotation files are in YOLO format.
1. Change the list 'all_categ' with custom dataset class names
2. Change 'direc_file' with custom txt file that contains all image locations.

Note: To train the network using Google faster RCNN API, class number should start from 1, instead of 0 in YOLO.

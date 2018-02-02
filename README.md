# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[final]: ./output_images/result.png "final output"

![alt_text][final]

## Overview

Relevant Files and Folders:
* `vehicle_detection.ipynb`
* `vehicle_detection.py`
* `train.py`
* `svc_pickle.p`
* `video.py`
* `project_out.mp4`
* `project_video.mp4`
* `output_images/`
* `test_images/`
* `non-vehicles/`
* `vehicles/`

The computer vision pipeline is defined in `vehicle_detection.py` in the `Pipeline` class. The pipeline is implemented 
with examples in the Jupyter Notebook: `vehicle_detection.ipynb`. The Linear SVC is trained and a pickle file 
`svc_pickle.p` is generated in `train.py`. The final output is implemented again in `video.py` to generate the video.

Images referenced in this writeup are stored in `ouput_images/`. 

If you wish to read a description of how the algorithms work, as well as shortcoming and ideas for improvement,
please read [writeup.md](https://github.com/jyork03/vehicle-detection/blob/master/writeup.md).

## Running the Code

First clone the project down.

```bash
git clone git@github.com:jyork03/vehicle-detection.git
```

Running it with Docker:

```bash
docker pull udacity/carnd-term1-starter-kit
cd ./vehicle-detection
docker run -it --rm -v `pwd`:/src -p 8888:8888 udacity/carnd-term1-starter-kit
```

If you don't use docker, you'll need to install the dependencies yourself.

The dependencies include:
* python==3.5.2
* numpy
* matplotlib
* jupyter
* opencv3
* ffmpeg
* sklearn
* scipy
* skimage
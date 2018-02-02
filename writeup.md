# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/color_hist.png
[image3]: ./output_images/hog.png
[image4]: ./output_images/result.png
[image5]: ./output_images/bboxes_to_results.png
[image6]: ./output_images/YCrCb_color_space.png
[image7]: ./output_images/sliding_windows.png
[video]: ./project_out.mp4

## Feature Extraction & Classifier Training

### Histogram of Oriented Gradients (HOG), Binned Spatial & Color Histograms Features

The parameters chosen for color space, feature extraction, etc. can be found in the `Pipeline` `__init__()` constructor
function under `# Tweakable Parameters` in [`vehicle_detection.py`](./vehicle_detection.py).  Each of these parameters
was selected through an iterative process of trial and error.

`YUV`, and `YCrCb` color spaces yielded better results during testing, but I eventually settled on `YCrCb` because it
seemed to produce fewer false positives in runs with the test video, though the differences were minor. An example of
the `YCrCb` color channels can be seen below:

![alt text][image6]

Here is an example HOG output of all three color channels in the `YCrCb` color space with HOG parameters of 
`orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, which are used throughout the pipeline:

![alt text][image3]

The HOG parameters above were chosen because they provided a high degree of accuracy, while still operating fast enough.

My processing pipeline is using OpenCV's hog function, as it proved faster than `skimage.feature.hog()`.

In addition to HOG features, I also included Binned Spatial Features with a spatial size of 32x32 and Color Histograms 
Features with 32 histogram bins.  A graph of the color histogram of the same image above is displayed below.

![alt text][image2]

All of these features are extracted from each 64x64 image (during training), or subimage (during production) and
combined together, then scaled/normalized before being fed into the classifier.

The Training features are extracted in `extract_features()` in [`vehicle_detection.py`](./vehicle_detection.py), and then 
combined, scaled and normalized in `combine_normalize_features()`, also in [`vehicle_detection.py`](./vehicle_detection.py)

### Classifier Training

I started by reading in all the `vehicle` and `non-vehicle` images.  This is found in `load_training_data()` in 
[`vehicle_detection.py`](./vehicle_detection.py). Here is an example of one of each of the `vehicle` and `non-vehicle` 
classes:

![alt text][image1]

Next, the features were extracted and scaled (as explained above), and training label vectors were defined (also in 
`combine_normalize_features()`). 

Then, in `fit_model()` a randomized test set and training set are separated with `sklearn.model_selection.train_test_split()` 
at an 80/20 split, and a Linear SVC model is fitted and finally scored.  The model with the parameters described above 
achieved an accuracy > 99.4%, which was good enough for my needs.


## Sliding Window Search

The code for the sliding window search can be found in the `find_cars()` function in [`vehicle_detection.py`](./vehicle_detection.py)

Each frame was read in as an image, and three sliding window searches were applied at varying scales and y-coordinate
ranges with a window size of 64px (matching the size of the training images). An example of the search space for each 
scale/range is shown below:

![alt text][image7]

As seen above, I chose to perform a sliding window search for three scales (1, 1.25 and 1.5) which overlap with each
other at varying degrees.  The basis for my decisions here was in trying to balance effectiveness with computational
cost.  Scale 1 was limited to the smallest search space because it was far more computationally costly.  However, it is
still very effective in the space it is search in.  It is looking farther our, thus searching more ground in a smaller 
pixel space.  The other two scales both completely overlap.  This choice was made because they are cheaper, but are 
effective at finding vehicles with different sized features.  I have to admit, though, that the 1.25 scale is more effective
than the 1.5 scale.

Each resulting window is treated as a subimage, of which features are extracted and fed into the classifier.

For each positive detection, bounding box coordinates were saved with an additional padding of 15 pixels.  This padding
helped to improve overlapping and grouping of nearby detections.  Then, all the bounding box information was converted
to a heatmap and thresholded out in the `run()` function in [`vehicle_detection.py`](./vehicle_detection.py).  This
thresholded heatmap information was then passed into `scipy.ndimage.measurements.label()` to generate labels for our
`draw_labeled_bboxes()` function to draw bounding boxes around.  At this point, each label is assumed to be a vehicle.

An example of this process is seen here:

![alt text][image5]

The final labeled result on several images/frames:

![alt text][image4]

## Video Implementation

Here's a [link to my video result](./project_out.mp4)

The code that was used to generate the video can be found in [`video.py`](./video.py)

Each frame is processed with the function `run()` in [`vehicle_detection.py`](./vehicle_detection.py)

The bounding box results from all the steps above, for each of the past 10 frames of the video, were added to an array,
and then a threshold of 6 was applied to the overlapping heatmaps.  This served to force the bounding boxes from the
past 10 frames to be in strong agreement with one another for a vehicle detection to register.

As a means to speed up processing (I as getting a little faster than 2 fps), I was only calculating new detections on
every other frame.

## Discussion

The pipeline I've implemented does a pretty good job recognizing cars in the test and project videos, but a place of
definite improvement would be to find ways of improving the processing speed without hurting accuracy.  I believe coding
for multithreaded processing and compiling OpenCV for GPU processing would make a huge difference.

Additional room for improvement would be for the recognition for cars that are not fully into view yet (ie. half the 
vehicle is off the screen), as well as maintaining an accurate vehicle count/detection for vehicles that may be behind 
another one.

Also worth noting, is this pipeline has not been tested in environments with vehicles of large variation, such as
motorcycles, semi-trucks, etc.  So more testing in that regard would be necessary.
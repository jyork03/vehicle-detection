import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from collections import deque


class Pipeline:
    def __init__(self):
        self.cars = []
        self.notcars = []
        self.X = None
        self.X_scaler = None
        self.scaled_X = None
        self.y = None
        self.svc = None
        self.box_list = deque(maxlen=10)

        # Tweakable Parameters
        self.color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb, GRAY
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.find_params = [(1.0, (400, 500)), (1.25, (400, 675)), (1.5, (400, 675))] # Scale & Min and max in y for sliding window search

        win_size = (64, 64)
        block_size = (self.pix_per_cell * self.cell_per_block, self.pix_per_cell * self.cell_per_block)
        block_stride = (8, 8)
        cell_size = (self.pix_per_cell, self.pix_per_cell)
        nbins = self.orient

        self.hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    def visualize_hog_features(self, img):
        features, hog_image = hog(img, orientations=self.orient,
                                  pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                  cells_per_block=(self.cell_per_block, self.cell_per_block),
                                  transform_sqrt=True,
                                  visualise=True, feature_vector=True)
        return features, hog_image

    def get_hog_features(self, img):

        return self.hog_descriptor.compute(img)

    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        # Return the feature vector
        return features

    def color_hist(self, img, bins_range=(0, 256), plot=False):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:, :, 0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(img[:, :, 1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(img[:, :, 2], bins=self.hist_bins, range=bins_range)

        if plot is True:
            # Generating bin centers
            bin_edges = channel1_hist[1]
            bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

            # Plot a figure with all three bar charts
            fig = plt.figure(figsize=(12, 3))
            plt.subplot(131)
            plt.bar(bin_centers, channel1_hist[0])
            plt.xlim(0, 256)
            plt.title('Channel 1 Histogram')
            plt.subplot(132)
            plt.bar(bin_centers, channel2_hist[0])
            plt.xlim(0, 256)
            plt.title('Channel 2 Histogram')
            plt.subplot(133)
            plt.bar(bin_centers, channel3_hist[0])
            plt.xlim(0, 256)
            plt.title('Channel 3 Histogram')
            plt.show()
        else:
            # Concatenate the histograms into a single feature vector
            hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

            # Return the individual histograms, bin_centers and feature vector
            return hist_features

    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        print("training image shape:", cv2.imread(imgs[0]).shape)
        for file in imgs:
            file_features = []
            # Read in each one by one
            image = cv2.imread(file)
            # apply color conversion if other than 'RGB'
            feature_image = cv2.cvtColor(image, getattr(cv2, "COLOR_BGR2" + self.color_space))
            if self.spatial_feat is True:
                spatial_features = self.bin_spatial(feature_image)
                file_features.append(spatial_features)
            if self.hist_feat is True:
                # Apply color_hist()
                hist_features = self.color_hist(feature_image)
                file_features.append(hist_features)
            if self.hog_feat is True:
                # Call get_hog_features() with vis=False, feature_vec=True
                if self.hog_channel == 'ALL':
                    hog1 = self.get_hog_features(feature_image[:, :, 0])
                    hog2 = self.get_hog_features(feature_image[:, :, 1])
                    hog3 = self.get_hog_features(feature_image[:, :, 2])
                    hog_features = np.vstack((hog1, hog2, hog3)).ravel()
                else:
                    hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel]).ravel()
                # Append the new feature vector to the features list
                file_features.append(hog_features)
            features.append(np.concatenate(file_features))
        # Return list of feature vectors
        return features

    def load_training_data(self):
        # Read in cars and notcars
        t = time.time()
        images = glob.glob('*vehicles/**/*.png')
        for image in images:
            if 'non-vehicles' in image:
                self.notcars.append(image)
            else:
                self.cars.append(image)

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to load and sort list of images...')

    def combine_normalize_features(self):
        t = time.time()
        car_features = self.extract_features(self.cars)
        notcar_features = self.extract_features(self.notcars)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract features...')
        print("car features shape", np.array(car_features).shape)

        t3 = time.time()
        # Create an array stack of feature vectors
        self.X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(self.X)
        # Apply the scaler to X
        print("X shape", self.X.shape)
        self.scaled_X = self.X_scaler.transform(self.X)
        print("scaled X shape", self.scaled_X.shape)
        # Define the labels vector
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        t4 = time.time()
        print(round(t4 - t3, 2), 'Seconds to scale and normalize features...')

    def fit_model(self):
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            self.scaled_X, self.y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell, 'pixels per cell and', self.cell_per_block,
              'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))

    def find_cars(self, img, scale=1.0, ystart=350, ystop=650):

        box_list = []
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = cv2.cvtColor(img_tosearch, getattr(cv2, "COLOR_BGR2" + self.color_space))
        print("ctrans_tosearch.shape", ctrans_tosearch.shape)

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ctrans_tosearch.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        # nfeat_per_block = self.orient * self.cell_per_block ** 2

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        for xb in range(nxsteps):
            for yb in range(nysteps):
                # features = []
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                features = []

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                # Should already be 64x64 if window is set to 64, but lets be safe
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                if self.spatial_feat is True:
                    spatial_features = self.bin_spatial(subimg)
                    features.append(spatial_features)

                if self.hist_feat is True:
                    hist_features = self.color_hist(subimg)
                    features.append(hist_features)

                if self.hog_feat is True:
                    # Call get_hog_features() with vis=False, feature_vec=True
                    if self.hog_channel == 'ALL':
                        hog1 = self.get_hog_features(subimg[:, :, 0])
                        hog2 = self.get_hog_features(subimg[:, :, 1])
                        hog3 = self.get_hog_features(subimg[:, :, 2])
                        hog_features = np.vstack((hog1, hog2, hog3)).ravel()
                    else:
                        hog_features = self.get_hog_features(subimg[:, :, self.hog_channel]).ravel()
                    features.append(hog_features)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack(features).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)
                if test_prediction == 1:
                    padding = 15  # helps to increase chance of overlapping bboxes for the heatmaps
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale) + padding
                    box_list.append(((xbox_left - padding, ytop_draw + ystart - padding),
                                     (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        self.box_list.append(box_list)

    def visualize_sliding_windows(self, img, scale=1.0, ystart=350, ystop=650, padding=15):
        # Make a copy of the image
        imcopy = np.copy(img)

        img_tosearch = imcopy[ystart:ystop, :, :]

        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        # Define blocks and steps as above
        nxblocks = (img_tosearch.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (img_tosearch.shape[0] // self.pix_per_cell) - self.cell_per_block + 1

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale) + padding

                imcopy = cv2.rectangle(imcopy,
                                       (xbox_left - padding, ytop_draw + ystart - padding),
                                       (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 3)

        return imcopy

    def add_heat(self, heatmap):
        # Iterate through list of bboxes
        for boxes_in_frame in self.box_list:
            print("number of boxes", len(boxes_in_frame))
            for box in boxes_in_frame:
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Make a copy of the image
        imcopy = np.copy(img)

        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(imcopy, bbox[0], bbox[1], (0, 0, 255), 3)
        # Return the image
        return imcopy

    def draw_boxes(self, img, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for boxes_in_frame in self.box_list:
            # print("number of boxes", len(boxes_in_frame))
            for bbox in boxes_in_frame:
                # Draw a rectangle given bbox coordinates
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def run(self, img, calc=True, clear_boxes=False, thresh=6):
        if clear_boxes is True:
            self.box_list.clear()

        if calc is True:
            t = time.time()

            for params in self.find_params:
                self.find_cars(img, scale=params[0], ystart=params[1][0], ystop=params[1][1])

            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to find cars...')
        # detected_cars = self.draw_boxes(img)

        t3 = time.time()
        # Add heat to each box in box list
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = self.add_heat(heat)

        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat, thresh)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        t4 = time.time()
        print(round(t4 - t3, 2), 'Seconds to apply heatmaps...')

        t5 = time.time()
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(img, labels)
        t6 = time.time()
        print(round(t6 - t5, 2), 'Seconds to draw final boxes...')

        return draw_img, heatmap

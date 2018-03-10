# **CarND Project 5 Writeup** 
# **Vehicle Detection** 


### This writeup includes a description for each of the following goals/steps:

#### [1. Classifier Training](#Classifier-Training),
#### [2. Sliding Window based Detection](#Sliding-Window-based-Detection),
#### [3. Heatmap Thresholding](#Heatmap-Thresholding),
#### [4. Custom Heatmap Class](#Custom-Heatmap-Class), and
#### [6. Full Pipeline Result](#Full-Pipeline-Result)

This repository contains a stand-alone approach for detecting cars in videos captured while driving on highways.

[//]: # (Image References)

[EXMPLE]: ./output_images/car_noncar_samples.jpg "Training data examples"
[FEAT]:   ./output_images/features.jpg "Features example extracted from images"
[HOTBB]:  ./output_images/hot_bboxes.jpg "Positively identified windows in an image"
[HEAT]:   ./output_images/heat_map.jpg "Heat map based on detections"
[ANN]:    ./output_images/test_image_ann.jpg "Thresholded heat map based annotations"

---

### Classifier Training

#### Data

One approach to vehicles detection is by using a classifier, that can identify whether there is a vehicle or not in a given image. To train such a classifier, labeled data with vehicles as well as without vehicles is required. For this purpose, the [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and [KITTI](http://www.cvlibs.net/datasets/kitti/) databases provide very useful data of 64x64 images, labeled as either having a vehicle or not:

![alt text][EXMPLE]

#### Feature extraction

The raw image pixels can be fed as features to the classification algorithm, yet specially extracted features often work better. According with the Udacity class, three feature extracting algorithms are included in this pipeline:

    # raw downsampled pixel features
    def bin_spatial(img, output_space='RGB', size=32):
        (...)
    return features

    # color histogram features
    def color_hist(img, output_space='RGB', nbins=32, bins_range=(0, 256)):
        (...)
    return features

    # HOG features
    def get_hog_features(img, output_space='RGB', channel='all', orient=9, pix_per_cell=8, cell_per_block=2, vis=False, 
                         feature_vec=True):
        (...)
    return hog_features

A wrapper function is included to make the code easier:

    # feature extractor from image file names, combining above feature methods
    def extract_features(img_files, spatial, hist, hog, cspace='BGR', verbose=False):
        (...)
    return features

This wrapper functions takes in all the required hyperparameters for the three extracting algorithms and passes them accordingly:
    
    # spatial binning hyperparams
    spatial_params = {
        'use':    True,
        'cspace': 'RGB',
        'size':   16}

    # color channel histogram hyperparams
    hist_params = {
        'use':    True,
        'cspace': 'RGB',
        'nbins':  32,
        'range':  (0, 256)}

    # HOG hyperparams
    hog_params = {
        'use':            True,
        'cspace':         'gray',
        'channel':        'all', # can be 'all' string, or 1 2 or 3 int
        'orient':         9,
        'pix_per_cell':   8,
        'cell_per_block': 2}

    # combine hyperparams in one dictionary
    feat_params_dict = {
        'cspace':  'YCrCb', # overall cspace
        'spatial': spatial_params,
        'hist':    hist_params,
        'hog':     hog_params}

The spatial extractor rescales the image to *['size']* x *['size']* and returns these raw pixels as features. The histogram extractor bins all pixel values into *['nbins']* number of bins in the range *['range']*. The hog extractor (from [sci-kit image](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog)) calculates the Histogram Oriented Gradients for the image's color channel *['channel']*. None of the *['cspace']*  keys for the three feature extractor functions is in use. Instead, the overal *feat_params_dict['cspace']* is being used instead to change the image's color space for all the extractor functions at once.

Choosing these hyperparameters proves quite hard, and a sort of trial and error process. On smaller sets, (useful for prototyping) even just the histogram features with 16 bins for each channel is enough for decent classification. For the colorspace the 'YCrCb' space is also hard to interpret, but in line with the class works rather well. After trying several settings, the above mentioned hyperparameters are adopted. A visual representation of the features can be seen in the following picture. It excludes the spatial feature, as the plot would be in a weird not easily understandable colormap.

![alt text][FEAT]

#### Classifier training

Before we can actually train/fit a classifier, we should divide the data into a training and a validation fold. This validation fold is useful, as the classifier can be tested on it in order to have a metric representing the classifiers performance. This splitting is done by the following sklearn function, with a *val_size* of 0.2, i.e. 20% data is separated for the validation set:

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

Classifiers work better most of the times with normalized data. To have an objective performance test with the validation set, it is important that the classifier *never* sees this data before the test. As such, the normalization should be based on only the training set. The validation set, and later images, *do* need to be normalized still as well however, according to the training set's normalization parameters

    feat_scaler = StandardScaler().fit(X_train)
    
    X_train = feat_scaler.transform(X_train);
    X_val   = feat_scaler.transform(X_val);

For classification, a powerful algorithm is the Support Vector Machine (SVM). This algorithm takes in several hyperparameters, and in order to choose the best one amongst them, a *hyper_params* dictionary can be made end fed to the GridSeachCV() function, that created a classifier for every option of choosing the hyper parameters, and returns the best performing one. The actual training only starts after the function *fit()* is being called on the *GridSearch()* object:
    
    hyper_params = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                    'C':      [0.001, 0.05, 0.1, 0.5, 1, 10, 100],
                    'gamma':  [0.001, 0.05, 0.1, 0.5, 1, 10, 100]}
    svc = svm.SVC()
    
    clf = GridSearchCV(svc, hyper_params)
    clf.fit(X_train, y_train)

Fitting this full set of hyperparameters takes extremely long. With smaller tests it seems the 'poly' kernel works better than the other two non-linear kernels, and thus the hyperparameter dictionary can be made smaller. Running the *GridSearch()* partially is still time consuming though, and actually the *LinearSVC()* class runs much faster and works very well. To save on time, this classifier is therefor chosen:

    hyper_params = {'C': [0.001, 0.05, 0.1, 0.5, 1]}
    svc = svm.LinearSVC()

After training, the validation accuracy is checked, and all the feature extraction parameters, as well as the classifier are stored in a pickle object, to allow for easy access later on.

### Sliding Window based Detection

This is where the test pipeline begins. Now that we have a classifier that only works on features extracted from an 64x64 image patch, how do we classify any patch within a bigger image or video? By sliding a window over the image, and making a prediction at every location. *The get_hot_boxes()* function implements this, and returns a list of all the windows/boxes that where classified as belonging to the vehicle class:

    get_hot_bboxes(img, cspace, bboxes_params, clf, feat_scaler,
                   spatial_params, hist_params, hog_params, confidence_thresh, verbose=False)

More specifically, this function takes in an image, converts the colorspace to *cspace*, crops the part of interest according to the *bboxes_params* dictionary, and scales the cropped part according to *bboxes_params['scales']*. On this cropped image part it then calculates all possible 64x64 patches based on a defined step size. This step size is 16 pixels (in the rescaled coordinate system), which is based on the HOG feature block calculations.

For each window the features are calculated based on *spatial_params* *hist_params* and *hog_params*, and the resulting features are scaled with *feat_scaler*. The trained classifier *clf* is then used to make a prediction consisting of a class and a confidence. If the class equals 1, and the confidence > *confidence_thresh*, that window patch outline is taken as a box and added to the positively identified box list.

Basing the step size for the windows on the HOG blocks, allows an efficient implementation of the HOG feature extractor. This is especially useful since the HOG feature calculations are the bottleneck in diminishing the time for the full pipeline. The efficiency lies in calculating the HOG features only once for the full image, and being able to easily extract the correct ones for each window when it is that window's turn to be classified. This prevents the re-calculation of HOG features on overlapping window parts. Currently this efficiency boost *is* implements, but regretfully not working properly. Due to time constraints I was not able to fix this issue. 

This whole process is repeated for several scales, and all the box detections are added together. The below image shows the resulting boxes drawn on the image they were detected on:

![alt text][HOTBB]

### Heatmap Thresholding

As can be seen in the picture above, the sliding window approach output contains several boxes, some correctly identifying the same vehicle and overlapping, some misclassifications and outliers on their own. In order to filter out these outliers, and combine the correct detections into a single box, a heat map approach is used.

This approach comprises of four steps/functions:

    def add_heat(heatmap, bbox_list):
        (...)
    return heatmap
    
    def draw_labeled_heatmap(img, labels):
        (...)
    return img

First, *add_heat()* takes in a list of positively classified bounding boxes from the sliding window results, and a heatmap. This heatmap is initialized with zeros, and has the size of the image on which the box detections were made. For each box, all pixels within that box's location on the heatmap get a +1. A single misclassification will result in a square patch of 1's somewhere, and overlapping detections will result in a highest value there where most boxes overlap, and values in between where they overlap to a lesser extend. Shown on the left of the picture below.
    
    def apply_threshold(heatmap, threshold):
        (...)
    return heatmap
    
Now in order to filter out the single patch of 1's from the misclassification, a general threshold is applied to the entire heatmap, zeroing every value < *threshold*. Shown on the middle of the picture below.

    labels = label(heat_thresh)
    
*label()* is a scipy imported function, that combines all connected patches in the thresholded heatmap under the same identifying label, each patch having its own label. Shown on the right of the picture below.

![alt text][HEAT]

    def draw_labeled_heatmap(img, labels):
        (...)
    return img
    
The last step predicts the final boxes by taking the max and min y and x values for each labeled patch, creating the box with top left corner (ymin, xmin) and bottom right corner (ymax, xmax). The *draw_labeled_heatmap()* function furthermore draws these boxes on an image *img*.

![alt text][ANN]

### Custom Heatmap Class

In order to make the heat map more robust, and to leverage temporal information we have when working with consecutive video frames instead of independent images, this simple track_labels() class is helpful:

    class track_labels():
        def __init__(self, H, W, threshold, temporal_history=7):
            (...)

        def __Clear(self, track):
            (...)

        def __Update(self, hot_bboxes):
            (...)

        def GetHeat(self, hot_bboxes):
            (...)
        return thresholded_heatmap

Basically, this class remembers the box detections from the last *temporal_history* amount of frames, adds them together in one single temporal heatmap, and returns this heatmap after thresholding with *threshold*.

### Full Pipeline Result

Now that we have all the building blocks in place, we can annotate the full project video. All the feature extractor parameters have been decided during the classifier training phase, which leaves the following parameters for now to choose still:

    # Define threshold for detection confidence and heat maps, and history length for heat maps
    heat_map_time_frames = 15
    heat_threshold       = 20
    confidence_thresh    = 0.5
    
    bboxes_params = {
    'idx':     0,
    'scales':  [1, 1.5],
    'y_start': [int(H*18/32), int(H*9/16), int(H*9/16)],
    'x_start': [int(W*2/8),   int(W*8/32), int(W*8/32)],
    'y_stop':  [int(H*12/16), int(H*7/8),  int(H*7/8)],
    'x_stop':  [int(W*16/16), int(W*16/16),       W]}

The *heat_map_time_frames* is the amount of frames that the *track_labels()* class remembers from the past. The *heat_threshold* is the threshold applied to the cumulative heat map of these past frames. The *confidence_thresh* is the threshold used during classification. These numbers relate to each other with some logic and should be tuned together. A lower *confidence_thresh* results in more detections, and might thus require a higher *heat_threshold*. A smaller number of frames being remembered from the past, results in lower values on the cumulative heat map, and might require in turn a lower *heat_threshold*.

The *bboxes_params* dictionary represents the scaling information. For each scale in *['scales']* the corresponding part of the image that should be cropped is contained at the same index location of the *['y_start']*, *['x_start']*, *['y_stop']* & *['x_stop']* lists.

Even with these logical intuitions, it is still a sort of trial and error to choose the best ones. The final video is made with the above listed settings.

#### Video result

An annotated video is created by looping over each frame of the video, while keeping track through time over the detections by means of the detection_history() class object. After applying the pipe_line to each frame, they are put back together to form the resulting video:

[Video can be found here.](https://drive.google.com/open?id=1_czpQYQxwkScnPqkoOtQgtBYlUTEc3fT)

#### Discussion

For the major part of the video, the pipeline works very decently, showing the merit of the pipeline approach and all its building blocks. The detections are perhaps not snug bounding boxes around the car, but they are correct identifications of car objects within the video.

There are a few parts however where the pipeline seems to shortly break. Similar to the **Advanced Lane Detection** project, the section with shadows and different colored tarmac/concrete proves hard. The current pipeline gets a few quick false positives around during those frames. Furthermore, somehow the pipeline loses the white care for a short while altogether, just before the black car overtakes it. Also the right side of the picture seems to have some issues, especially when the white car is leaving. This however is understandable, as we humans can identify half of a car, but the classifier is mostly trained on images with full cars inside, which is not the case anymore for *any* window as soon as the car is leaving the video.

One thing that is noticeable as well is that the heat_map need to 'warm up'. Since it takes information over several time frames, it requires a vehicle also to be detected in several time frames. Therefor, vehicle detections in the first few timeframes with a new vehicle get filtered out, due to the sort of averaging effect of the *track_labels* class (by taking a cumulative but also a corresponding higher threshold, this is a form of averaging).

#### Conclusions and remarks

One issue I had, that I still don't fully understand, is that to prevent jpg vs png vs video loading differences, I was preemptively scaling every image to the range [0-1.0] and casting them to np.float32. Training worked fine, but somehow the videoframes where performing significantly worse, even after color conversion, scaling and casting. When I saved these frames as jpg to disk, and read them in as single images, they would also never give me the exact same pixel values. In the end I prevented this issue from hampering me by switching to loading everything in with cv2, resulting in [0-255] integer ranges.

HOG features are a key feature of this current implementation, but also timewise the main bottleneck. As explained above, a more efficient implementation exists of extracting all the required HOG features for a full image. Yet this implementation is not working at this moment and therefor uncommented in the current code.

The threshold on the heat map is set rather severe, in order to filter out more noise. This also results in smaller boxes for the actual detection being drawn, as can be seen in the video at around 00:25-00:28 (the seemingly hardest part for the pipeline). A possible solution for this could be to go back to the heat map, *after* the initial thresholding, and grow the initial threshold detection regions with a lower threshold, adding less 'heat-confidence' edges of the cars to the bounding box being drawn. Currently this is not implemented.









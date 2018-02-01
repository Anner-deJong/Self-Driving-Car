# **CarND Project 4 Writeup** 
# **Advanced Lane Finding** 


### This writeup includes a description for each of the following goals/steps:

#### [1. Camera Calibration](#Camera-Calibration),
#### [2. Color and Gradient Threshold Preprocessing](#Color-and-Gradient-Threshold-Preprocessing),
#### [3. Birds-eye View Perspective Transform](#Birds-eye-View-Perspective-Transform),
#### [4. Lane Line Detection and Fitting](#Lane-Line-Detection-and-Fitting),
#### [5. Curvature and Vehicle Position](#Curvature-and-Vehicle-Position), and
#### [6. Full Pipeline Result](#Full-Pipeline-Result)

This repository contains a stand-alone approach to annotating lane lines in videos captured while driving on highways.

[//]: # (Image References)

[CHESS]: ./output_images/Chessboard_corners_found.jpg "Chessboard corner detection example"
[UND_1]: ./output_images/Undistorted_chessboard.jpg "Chessboard undistortion example"
[UND_2]: ./output_images/Undistortion.jpg "Road undistortion example"
[THRES]: ./output_images/Binary_threshold.jpg "Color and gradient thresholding example"
[WARP]:  ./output_images/Warp_transform.jpg "Birds-eye view warp example"
[LINE]:  ./output_images/Line_detections.jpg "Lane line detection example"

---

### Camera Calibration

Camera captures, whether images or videos, are often distorted due to their inherent physical workings, think for example of the fish-eye effect. Luckily, opencv comes with easy functions to undistort a certain camera's feed:
    
    _, corners = cv2.findChessboardCorners(chessboard_image, (nx,ny), None)
    _, mtx, dist, _, _ = cv2.calibrateCamera(base_points, corner_points, img_size, None, None)
    un_dst = cv2.undistort(img, mtx, dist, None, mtx)

The findChessboardCorners() functions automatically detects and returns chessboard corners in an image taken before with a camera. nx and ny are the amount of corners it should detect. The result of some chessboard_image detection looks like this:

![alt text][CHESS]

After detecting the corners in some images, these corner_points can be references to base_points. These base_points basically represent the underlying ground-truth relation between the points, and in the case of a chessboard it is a simple grid (since chessboard corners fit in a perfect grid). With these base_points, the calibrateCamera() function automatically calculates a camera specific matrix mtx and vector dist, that can be used for undistorting an image with the undistort() function.
This results in the following effect:

![alt text][UND_1]

With the chessboard image it is easy to see the amelioration: the curved lines that should be perpendicular and uncurved, indeed become like that. For a road image it is harder to immediately interpret the undistorted image as the more true to reality one, yet if it comes from the same camera we know the same undistortion should apply:

![alt text][UND_2]

### Color and Gradient Threshold Preprocessing

Our final goal is to detect the lane lines. In order to detect these, various filters are applied to the undistorted image in order to create a binary image map with the lane line pixels and the rest blacked out as much as possible. To do this, several threshold functions are included, plus a wrapper with all functions inside:

    # sobel/derivative threshold function
    def sob_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # sobel/derivative angle threshold function
    sob_angle_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # BGR threshold
    bgr_threshold(img, channel_choice='r', thresh=(0, 255)):
    
    # HLS threshold
    hls_threshold(img, channel_choice='s', thresh=(0, 255)):
    
    # full wrapper for multiple thresholding functions
    def threshold_filter_image(img, sob_k=11, sob_thr=(20, 100),
                           sob_angle_k=9, sob_angle_thr=(1/6*math.pi, 1.3),
                           r_thr=(150, 255), s_thr=(170, 255), h_thr=(15, 90))

The sob_threshold() function applies a sobel operator with kernel size specified in either an x or a y direction, or in both, specified by the orient parameter. Only output values between thresh[0] and thresh[1] are afterwards included in the binary map.

The sob_angle_threshold() function applied a sobel kernel in both x and y directions, and checks the corner of the 'sobel derivate'. Again only values within the thresholds are included in the binary map.

The bgr_thresholds() function simply outputs a binary map only for values of a specified channel within the threshold range, same counts for the hls_threshold() function.

The threshold_filter_image() is a wrapper function for all the thresholding functions at the same time, in which all the hyperparameters can be loaded. This function holds a combination of several thresholding functions into a final binary output. Choosing this combination is tedious and not straight forward, yet highly influential on the final detection performance. The combination that was chosen in the end is:

    final_binary[(sy_binary==0) & ((h_binary == 1) & (r_binary == 1) & (s_binary == 1) | (sx_binary == 1))] = 1

sy_binary and sx_binary are derived from the sob_threshold() with y and x direction respectively. h/r/s_binary represent h/r/s channel thresholds. For each of these the hyperparameters are chosen as in the wrapper function above. These threshold values result in a binary map like this:

![alt text][THRES]

### Birds-eye View Perspective Transform

Warping an binary map result from the previous step into a birds-eye view is done by the opencv getPerspectiveTransform() and warpPerspective() functions. (In this project these are wrapped in the warp_image() function). H & W are image Height and Width respectively. This warp will zoom in on the relevant part of the image (the road right in front), and hopefully make the lane lines parallel and easier to detect.

    # Decide source and destination points
    src  = np.float32([[0.15*W, H], [0.444*W, 0.65*H], [0.56*W, 0.65*H], [0.88*W, H]])
    dst  = np.float32([[0.25*W, H], [0.25*W, 0],      [0.75*W, 0],       [0.75*W, H]])
    M    = cv2.getPerspectiveTransform(src, dst)
    wrp  = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)

The source and destination points are manually chosen, according to this example:

![alt text][WARP]

The same source and destination coordinates are used throughout the entire video. This might not be the best warp for each frame, but 1) it is hard to automate the picking of the warp coordinates, and 2) if the highway is assumed flat, and thus a constant plane, the optimal warps for each frame shouldn't differ so much.
NB in the regular pipeline only the binary maps get warped, but in order to show the warping, and to define the source and destination points, a regular image in used and shown.
NNB By manually defining the warp, we include a specific zooming of the image as well. Later on the radii and vehicle location are being calculated based on some pixel to meter ratios. However, these are less accurate because the zooming in this warping step is not taken into account.

### Lane Line Detection and Fitting

After obtaining a warped binary image, we need to find the 1/True pixels that together form the lane lines. This is done by dividing the image in multiple rows, and for each row checking which horizontal location includes the most pixels. More specifically, the previous row's detected horizontal location is taken as the center for the next row's search. From somewhere left of this center position to somewhere right, a window convolution is taken, of which the argmax() gives the horizontal location of the window that contains the most pixels in said row. For the bottom row the center of the previous image is taken as start. This procedure is conducted for both the left lane line as well as the right lane line. If no prior center locations are available, the entire row is included in the search. Furthermore, if no prior detection exists for the first row, it will flatten the entire lower half of the image, over which a convolution is performed (with a window height of 1), again taking the argmax(). 

    def detect_lines(img, win_w=50, win_h=80, margin=100, det_hist=detection_history())

This function implements this row-wise detection. The window sizes for the convolution are given by win_w & win_h. How far left and right of the previous center location a current row will perform the convolution is defined by margin. The detection history is a class object that keeps track of prior detections, and is passed here for initializing the center locations for the first row search. (The class is explained in more detail [below](#Detection-History)).
    
Once a window has been decided on as containing the most pixels and thus forming part of the lane line, it is passed to window_ind() function to store the pixel coordinates:

    def window_ind(win_w, win_h, img, center, level)

Center refers to the horizontal location of the window, and level to which row. After obtaining all the pixels, numpy.polyfit() is able to fit a polynomial to them:

    # fit curves
    l_curve = np.polyfit(l_ind_y, l_ind_x, 2)
    r_curve = np.polyfit(r_ind_y, r_ind_x, 2)

The l/r_ind_x/y indicate the x/y coordinates for the obtained pixel expected to belong to the left/right lane line. The 2 indicates the aim to fit a *second* order polynomial, which suffices for limited lane line sections. This function returns three values: a, b & c, representing the curve: f(y)=Ay^2+By+C. The curves are fitted to y instead of x, since they are expected to be much more vertical than horizontal, for which a fitting to y will give more accurate results. 
   
### Curvature and Vehicle Position

Once the lane line fits are obtained, the curvature at any point can be calculated by

    R_curve(y) = (1 + (2*A*y + B)**2)**(3/2) / abs(2*A)
    
However, this radius is in pixel parameter space, not in meters. Therefore, for each (pixel) row in the warped image, a point on the curve is calculated (with f(y) defined above). These points are than transformed from pixel space to meter space by multiplying the x and y coordinates with horizontal and vertical pixel to meter ratios:

    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

To calculate the car's relative position to the lane center, the x coordinates of the two lane lines are taken at the bottom of the image (with f(y=H), H being the height of the image). Taking center of the warped image as the car's position (the warp keeps the bottom center position untouched), and comparing these two the middle of the two x coordinates, the difference in pixels and hence in meters can be calculated:

    lane_cen = -((r_lane_pos-l_lane_pos)/2 + l_lane_pos - W/2)*xm_per_pix
    
lane_cen is the calculated difference in meters, l/r_lane_pos are the x coordinates for the left and right lane lines respectively all the way at the bottom of the warped image.

### Full Pipeline Result

A full pipeline is composed by adding all the above functionality together, taking in an image (and hyperparameters) and returning an undistorted annotated version of that image:

    def full_pipeline(det_hist, img, mtx, dist, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix, src=None, dst=None):
    H, W = img.shape[:2]
    # Undistort the image
    un_dst = cv2.undistort(img, mtx, dist, None, mtx)
    # Threshold the undistorted image
    binary = threshold_filter_image(un_dst, sob_k=11, sob_thr=(20, 100),
                                    sob_angle_k=9, sob_angle_thr=(1/6*math.pi, 1.3),
                                    r_thr=(150, 255), s_thr=(170, 255), h_thr=(15, 90))
    # Warp the binary to a birds-eye view
    warped = warp_image(binary, src, dst)
    # Detect lines
    l_curve, r_curve, l_ind_y, l_ind_x, r_ind_y, r_ind_x = detect_lines(warped, win_w=80, win_h=80,
                                                                        margin=100, det_hist=det_hist)
    # Detect curvatures in meters
    l_curve_rad, r_curve_rad = detect_radii(l_curve, r_curve, H, ym_per_pix, xm_per_pix)
    # Update detection history object
    check = det_hist.updater(l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W)
    # Create annotated image
    annotated = annotate(un_dst, warped.shape[0], warped.shape[1], det_hist)
    
    return annotated

#### Detection History

In order to make the detections more robust, a history_detection() class object is included. This class is updated after line detection in each frame, and passed to the next frame before detection. Its has two main functionalities: 1) prevent detection outliers, by averaging the detection results over time and by skipping detections that fail sanity checks, and 2) to speed up the detection in a next frame by providing the centers required inside the detect_lines() function.
The most important parameters the class keeps track of are: last_detection (a bool keeping track of whether the last detection passed the sanity checks), l_curve (the A, B & C values for the left curve fit), r_curve (idem for right curve), curve_radius (the radius of the detections, in meters) & lane_center (the relative position of the car to the lane center, in meters).

##### Time averaging
If the sanity checks described below judge the new detection to be valid and comparable to the previous detection(s), the history_detection.update() is called, which updates all the class parameters 

##### Sanity checks
How the object is updated with a new detection, depends on how that detection passes through two sanity checks:
1) a detection validity check over the new detection, to check whether it is a valid detection or not, and 2) a  we check with prior results whether the newly detected lines are close to our previous detections

det_validity_check(self, l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix=3.7/700, H=720, W=1280)

    val_check = self.det_validity_check(l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W)
    con_check = self.det_consistency_check(l_curve, r_curve, l_curve_rad, r_curve_rad, H)
    
The validity check looks purely at a new detection, not taking into account prior detection information. It consists of three subchecks, returning False if any of these three fails, and True if they all pass:

* Subcheck 1: Do both detected lines have similar curvatures? To prevent this check for straight lanes, a cap is included as well.
* Subcheck 2: Are both detected lines roughly parallel?
* Subcheck 3: Are both detected lines separated by approximately the right distance horizontally?
This actually requires to calculate the zoom during the warp, to calibrate the pixel/m ratio.
This is not included in this work, and seems somewhat too tedious for a robust check.

The consistency check looks at the new detection and compares it to the previous stored detection. It consists of two subchecks, returning again False if either fails, and True if not:

* Subcheck 1: are both newly detected radii not too different from previous radii? (subcheck not included for straight lanes)
* Subcheck 2: are starting points of the newly detected curves and previous curves close?

The way an instance of this class is updated depends on the outcome of the two checks.The logic goes as follows:

If both checks are True, it means our detection is a valid and timewise consistent one, so we update the internal values of the instance with the new detection values.

If they are not both, the last_detection parameter is set to False, in order to have the line pixel detector function search globally and not based on previous detected centers (as explained above).

If the validation check is True, but the consistency check is False, we start a counter. For every new detection that passes the validation check but fails the consistency check, the counter goes up by 1, until a limit is reached. Up until this limit, new detections that differ from the existing detection inside the instance will be discarded. The limit is interpreted as the case where too many new detections have been inconsistent with the existing detection, that the existing detection should be discarded instead. The instance will be re-initialised in that case with the last detection.

If, before the limit is reached, both checks are True for a new detection, the counter is reset. In the case the validation check fails, nothing is updated.

#### Video result

An annotated video is created by looping over each frame of the video, while keeping track through time over the detections by means of the detection_history() class object. After applying the pipe_line to each frame, they are put back together to form the resulting video:

<video src="videos/project_video_annotated.mp4" width="640" height="360" controls preload></video>

#### Conclusion

Compared to the first Udacity lane detection project, this video offers an enormous performance boost. Yet, although the video perhaps provides *okay* results, it is definitely not outstanding, as there is clearly room for improvement looking at the video.

The main video contains two lighter colored tarmac section, both with challenging shadow and lightning conditions. Outside of these sections, the pipeline performs decently well, yet within these two sections (especially the first one) it's not very robust.

It can both be seen that our history_detection fails to properly check some frames, as the annotations shows weird unnatural lane curves, as well as that it might be filtering out too much, as some parts of the video have a static annotation (i.e. the annotation is not being changed over time, likely because the history_detection class discards several new detections in a row as not valid).

The two main critical points in the pipeline deserve to be mentioned here once more, as investigating and improving those further might have the most chance of improving the video quality:

1) The thresholding can be done in many different ways, depending on the hyperparameters for each thresholding channel as well as the way the channels are combined into a final threshold. Due to the many possibilities, and the influence it has on the detection result, it is important yet very hard to obtain the optimal settings. More tweaking might therefore increase the annotation performance.

2) To warp every image to a birds-eye perspective, the same warping coordinates are used. Since these coordinates are based on a single image, they might not represent the best universal warping coordinates, rendering some (or many) warps sub-optimal for lane detection. Especially when the assumption of a complete flat road doesn't hold, the warp will likely bring about wrong birds-eye views. Somehow automating or choosing better warping coordinates might therefore also boost the annotation performance.




# **CarND Project 1 Writeup** 
# **Finding Lane Lines on the Road** 


### This writeup includes:
#### 1. pipeline description,
#### 2. performance and shortcomings, and
#### 3. thoughts on improvements

[//]: # (Image References)
[image_pipe_parta]: ./test_images/whiteCarLaneSwitch_noExtr_noAvg_output.jpg
[image_pipe_partb]: ./test_images/whiteCarLaneSwitch_output.jpg
[image_challenge_noAvg_noExtr]: ./test_videos/challenge_frame_95_noExtr_noAvg.jpg
[image_challenge]: ./test_videos/challenge_frame_95.jpg

---

### 1. Pipeline description

#### Pipeline part a
The full pipeline can be divided into 2 parts. Part a of the pipeline includes everything we learned in class, with the lines predicted by the hough transform as output.

First the images are rendered to grayscale (for canny edge detection (why actually can't we do canny edge detection for 3 separate channels?)) if they're not already:
    
    if (len(img.shape) < 3): # if already gray/2D scale
        gray     = np.copy(img)
    else:
        gray     = grayscale(img)

Secondly, the image is blurred for stability and to prevent artifacts and such, as recommended by the lectures (gb_k is the kernel parameter):

    blur_gray    = gaussian_blur(gray, gb_k)

With canny edge detection the 'strong grayscale derivatives' are then calculated, as well as a mask is applied to a user defined polygon (can_thr are the canny edge threshold parameters, verts the user defined polygon vertices):

    edge         = canny(blur_gray, can_thr[0], can_thr[1])
    mask_edge    = region_of_interest(edge, [verts])
    
Finally, based on the masked canny edge detection output, the hough transform is used to predict possible lines of the lanes (parameters are the basic hough transform parameters):
    
    hgh_lines    = hough_lines(mask_edge, rho, theta, threshold, min_line_len, max_line_gap)
    
Here is what an image looks like when annotated with part a of the pipeline:

![alt text][image_pipe_parta]

#### Pipeline part b
Part b of the pipeline consists of some creative functions that aim to ameliorate the predicted hough transform lines. I did not include them in the draw_lines() function, but preferred to keep them separate. They basically take as input the predicted hough transform lines, and have as output a left lane line and a right lane line that can be drawn on the original picture with the draw_lines() function.

First all the predicted lines are filtered and separated. If the abs(slope) of a line is less than the 'min_slope' parameter, the line is discarded. This aims to filter out horizontal artifacts. Based on a negative or positive slope the lines are separated in left lane lines and right lane lines:

    left_lane_lines, right_lane_lines = separate_slopes(hgh_lines, min_slope)
    
These lines are then averaged, separately for left and for right, returning one left line and one right line (if inputs are not empty):
    
    average_left_line, average_right_line = avg_per_slope(left_lane_lines, right_lane_lines)
    
Finally, both lines are than extrapolated downwards to the bottom of the picture and upwards (or sometimes cut off) to the intersection of both lines. This returns a triangle with one side on the bottom of the image. If either line is empty, the other line is only extrapolated downwards:
    
    ameliorated_lines    = extrapolate_lines(average_left_line, average_right_line, Height_image, Width_image)

Here is what an image looks like after the amelioration:

![alt text][image_pipe_partb]

I included a full wrapper for the pipeline as well, which takes as input an image. It has a standard value for most parameters, but allows for passing a specific value at each call as well. It also allows to output the annotated image of the pure hough lines, without any amelioration, by including 'avg_extr=False' as parameter.

    annotated_image = full_pipeline(image, avg_extr=True, gb_k=3, can_thr=(50, 150), rho=1, theta=np.pi/180, threshold=20, min_line_len=100, max_line_gap=40, region_verts=None):

### 2. Performance & shortcomings

In terms of the goal:

***Make a pipeline that finds lane lines on the road***

it seems the current pipeline performs reasonably well. Most of the *solidWhiteRight* and *solidYellowLeft* video frames look like the image after amelioration above. There is no disturbing jitter of the lines, only a left line and a right line and no other line artifacts, nor do the lines divert too far from the actual lane lines.

* **Intersection Extrapolation** One interesting fact, is that I extrapolated the lines upwards to their intersection, disregarding the earlier defined region of interest borders.
This uppermost bit of the lines is likely the most susceptible to errors, but perhaps also the least important part of the lines. It really depends on the bigger scope of what a car wants to do after it knows the lane lines, that it will become apparent whether extrapolating the lines like this is good or bad. It might give a false sense of prediction accuracy far out (at the top of the triangle), but on the other hand this decreases the storage for each frame (only 3 (x,y) points necessary instead of 4). This is a 25% decrease, but again, whether it's relevant at all really depends on further steps after line detection.

Looking at the *challenge* video frames below, it can be seen that the current pipeline is not at all sufficient yet though. The most obvious issues are:

* **curved lane lines** If a car is moving through a curve in the road, the lane lines will be curves as well. Our current videos are not that curved, but just imagine driving in the mountains. The current pipeline can only fit straight lines, which might become an issue when lane lines are excessively curved (somewhat in line with the intersection extrapolation discussion)
* **Jittering predicted lines** The first part and last part of the video are annotated decently well with the amelioration pipeline part b. However, especially in the middle, the left lane line starts jittering from the actual line position to the left and back. As can be seen in the frame without amelioration, there are a lot of wrongly predicted lines. The slope filter however filters most of these out, because they are horizontal. However, since these artifact lines are also formed due to the shadows, we were just lucky the shadows in this video are horizontal. If the shadows are not horizontal, or if there are other line invoking element in the image such as the concrete (?) border between the two directions on the highway, these line predictions are not filtered out. In the frame without amelioration it can clearly be seen that where the correct prediction for the left line stops, another curved prediction continues upwards in the frame at this border. Averaging over correct and such incorrect lines is what likely causes jitter, as the resulting average of the line lies somewhere in the middle, which is exactly what we can see in the ameliorated frame.

Frame 95 of the *challenge* video, without the amelioration (only hough pipeline part a annotations):
![alt text][image_challenge_noAvg_noExtr]

Frame 95 of the *challenge* video, with the amelioration/ pipeline part b annotations:
![alt_text][image_challenge]

### 3. Improvements

Too overcome the curved lane line problem, we need a method that can not only fit straight lines, but also curves. It seems possible to do a Hough transform to a 3D space to fit curves like *y = ax^2 + bx + c*, which allows for detection of polynomials in the original frame. A quick google search indeed shows that it is possible to detect curves and even random shapes with hough transforms. This is likely more computationally expensive, so perhaps a tradeoff would be made when straight lines are not sufficient anymore.

Too overcome jitter, we could leverage the synergy that a video provides over just the combination of a bunch of frames: lane lines don't jitter from one image to the next. This information can be used to filter out detected lines in frame t+1 that are too far from the final predicted lane line position in frame t (assuming frame t is predicted correctly). This time smoothing is however not available for single images, but assuming an SDC needs a continuous video anyway, this smoothing could provide the perfect solution.
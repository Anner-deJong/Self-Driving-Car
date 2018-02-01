"""
Helper file with functions each implementing part of a full, automatic lane line detection and annotation pipeline
"""

import numpy as np
import cv2
import math

# sobel/derivative threshold function
def sob_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient=='x':    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient=='y':  abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    elif orient=='xy':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            abs_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    else: raise ValueError('uncorrect orient, choose between \'x\', \'y\' or \'xy\'')
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sca_sobel = (abs_sobel / abs_sobel.max() * 255).astype(np.uint8)
    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary = np.zeros_like(sca_sobel)
    binary[(sca_sobel > thresh[0]) & (sca_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary

# sobel/derivative angle threshold function
def sob_angle_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobx = abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    soby = abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(soby, sobx)
    # 5) Create a binary mask where direction thresholds are met
    binary = np.zeros_like(grad_dir)
    binary[(grad_dir > thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary

# BGR threshold
def bgr_threshold(img, channel_choice='r', thresh=(0, 255)):
    # Choose correct channel
    if   channel_choice=='b': channel = img[:,:,0]
    elif channel_choice=='g': channel = img[:,:,1]
    elif channel_choice=='r': channel = img[:,:,2]
    else: raise ValueError('Incorrect channel choice, choose between \'b\', \'g\' or \'r\'')
    # Create binary channel for output
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary

# HLS threshold
def hls_threshold(img, channel_choice='s', thresh=(0, 255)):
    # Convert img to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    # Choose correct channel
    if   channel_choice=='h': channel = hls[:,:,0]
    elif channel_choice=='l': channel = hls[:,:,1]
    elif channel_choice=='s': channel = hls[:,:,2]
    else: raise ValueError('Incorrect channel choice, choose between \'h\', \'l\' or \'s\'')
    # Create binary channel for output
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary
    
# create full wrapper for multiple thresholding functions
# Efficiency gain: comment out those thresholding functions that are not being used
def threshold_filter_image(img, sob_k=11, sob_thr=(20, 100),
                           sob_angle_k=9, sob_angle_thr=(1/6*math.pi, 1.3),
                           r_thr=(150, 255), s_thr=(170, 255), h_thr=(15, 90)):
    
    sx_binary         = sob_threshold(img, orient='x', sobel_kernel=sob_k, thresh=sob_thr)
    sy_binary         = sob_threshold(img, orient='y', sobel_kernel=sob_k, thresh=sob_thr)
    sa_binary         = sob_angle_threshold(img, sobel_kernel=sob_angle_k, thresh=sob_angle_thr)
    r_binary          = bgr_threshold(img, channel_choice='r', thresh=r_thr)
    s_binary          = hls_threshold(img, channel_choice='s', thresh=s_thr)
    h_binary          = hls_threshold(img, channel_choice='h', thresh=h_thr)
    
    # Create binary output filter(s)
    final_binary = np.zeros_like(img[:, :, 0])
    # Choose which and how to combine different masks (hyperparameter)
    #setting #1:
    final_binary[(sy_binary==0) & ((h_binary == 1) & (r_binary == 1) & (s_binary == 1) | (sx_binary == 1))] = 1
    #setting #2:
    #final_binary[((h_binary == 1) & (s_binary == 1)) | (sx_binary == 1)] = 1
    #setting #3:
    #final_binary[((sx_binary == 1) | (s_binary == 1)) & (sa_binary == 1)] = 1
    
    return final_binary

# function that warps the image according to source points (src) and destination points (dst)
def warp_image(img, src=None, dst=None, warp_back=False):
    H, W = img.shape[:2]
    if src is None: src  = np.float32([[0.15*W, H], [0.444*W, 0.65*H], [0.56*W, 0.65*H], [0.88*W, H]])
    if dst is None: dst  = np.float32([[0.25*W, H], [0.25*W, 0],       [0.75*W, 0],      [0.75*W, H]])
    
    if warp_back: M = cv2.getPerspectiveTransform(dst, src)
    else:         M = cv2.getPerspectiveTransform(src, dst)
    
    wrp  = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    
    return wrp


# Class that keeps track of detection history for averaging and better robustness
class detection_history:
    def __init__(self, last_detection=False, l_curve=np.zeros(3), r_curve=np.zeros(3),
                 curve_radius=0, lane_center=0, moving_avg=0.7, verbose=False):
        self.last_detection = last_detection
        self.l_curve        = l_curve
        self.r_curve        = r_curve
        self.curve_radius   = curve_radius
        self.lane_center    = lane_center
        self.moving_avg     = moving_avg
        self.counter        = 0
        self.max_wrong_det  = 6 # maximum number of nonconsistent detections before reinitialization
        self.verbose        = verbose
        self.just_init      = True
    
    def update_step(self, l_curve, r_curve, radius, lane_center):
        # update with moving average
        self.l_curve = self.moving_avg*self.l_curve + (1-self.moving_avg)*l_curve
        self.r_curve = self.moving_avg*self.r_curve + (1-self.moving_avg)*r_curve
        self.curve_radius = min(2000, self.moving_avg*self.curve_radius + (1-self.moving_avg)*radius)
        self.lane_center = self.moving_avg*self.lane_center + (1-self.moving_avg)*lane_center
    
    def re_init(self, l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W):
        # reinitialize if detection is not consistent for self.max_wrong_det time frames
        if self.verbose: print('Reinitializing history object..')
        self.last_detection = True
        self.l_curve        = l_curve
        self.r_curve        = r_curve
        self.curve_radius   = (l_curve_rad + r_curve_rad)/2
        l_lane_pos          = l_curve[0]*H**2+l_curve[1]*H+l_curve[2]
        r_lane_pos          = r_curve[0]*H**2+r_curve[1]*H+r_curve[2]
        self.lane_center    = -((r_lane_pos-l_lane_pos)/2 + l_lane_pos - W/2)*xm_per_pix
    
    def get_centers(self, H):
        # function specifically for the line detection function, so it can use the already detected line centers
        l_lane_pos = self.l_curve[0]*H**2+self.l_curve[1]*H+self.l_curve[2]
        r_lane_pos = self.r_curve[0]*H**2+self.r_curve[1]*H+self.r_curve[2]
        return l_lane_pos, r_lane_pos
    
    def updater(self, l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W):
        if self.just_init: # Inefficient! implement a re_init step if all values are initialized with 0 
            self.just_init = False; self.re_init(l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W)
            return True
        val_check = self.det_validity_check(l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W)
        con_check = self.det_consistency_check(l_curve, r_curve, l_curve_rad, r_curve_rad, H)
        if val_check & con_check:
            if self.last_detection: self.counter =0
            self.last_detection = True
            l_lane_pos          = l_curve[0]*H**2+l_curve[1]*H+l_curve[2]
            r_lane_pos          = r_curve[0]*H**2+r_curve[1]*H+r_curve[2]
            lane_cen            = -((r_lane_pos-l_lane_pos)/2 + l_lane_pos - W/2)*xm_per_pix
            self.update_step(l_curve, r_curve, (l_curve_rad + r_curve_rad)/2, lane_cen)
        else:
            self.last_detection = False # Search globally for lines in next frame, i.e. not based on last detection
    
        if not (not val_check | con_check):
            if self.counter < self.max_wrong_det:
                self.counter += 1
            else: # reached maximum amount of allowed for inconsistent detections
                self.re_init(l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix, H, W)
                self.counter = 0
        return (val_check & con_check)
    
    # There are two sanity checks
    # First, a detection validity check over the new results, to check whether its a valid detection or not
    # Secondly, we check with prior results whether the newly detected lines are close to our previous detections
    def det_validity_check(self, l_curve, r_curve, l_curve_rad, r_curve_rad, xm_per_pix=3.7/700, H=720, W=1280):
        # Check that both detected lines have similar curvature
        # To prevent this check for straight lanes, a cap is included as well. 
        if ((l_curve_rad < 0.75*r_curve_rad) | (l_curve_rad > 1.25*r_curve_rad)) & \
        ((l_curve_rad < 1000) | (r_curve_rad < 1000)):
            if self.verbose: print('failed similar curvature')
            return False
        # Checking that they are roughly parallel
        dif_bot = l_curve[0]*H**2+l_curve[1]*H+l_curve[2] - r_curve[0]*H**2+r_curve[1]*H+r_curve[2]
        dif_up  = l_curve[2] - r_curve[2]
        if (dif_bot < 0.75*dif_up) | (dif_bot > 1.25*dif_bot):
            if self.verbose: print('failed parallel')
            return False
        # Check that both detected lines are separated by approximately the right distance horizontally
        # This actually requires to calculate the zoom during the warp, to calibrate the pixel/m ratio
        # This is not included in this work, and seems somewhat too tedious for a robust check
        l_lane_pos = l_curve[0]*H**2+l_curve[1]*H+l_curve[2]
        r_lane_pos = r_curve[0]*H**2+r_curve[1]*H+r_curve[2]
        lane_w = (r_lane_pos-l_lane_pos)*xm_per_pix
        if (lane_w < 1.5) | (lane_w > 4):
            if self.verbose: print('failed lane width')
            return False
        return True
    
    def det_consistency_check(self, l_curve, r_curve, l_curve_rad, r_curve_rad, H=720):
        # Check that both radii are not too different from previous radii (not for straght lanes!)
        if ((l_curve_rad < 1000) & (r_curve_rad < 1000)):
            if ((l_curve_rad < 0.9*self.curve_radius) | (l_curve_rad > 1.1*self.curve_radius)) & \
            ((r_curve_rad < 0.9*self.curve_radius) | (r_curve_rad > 1.1*self.curve_radius)):
                if self.verbose: print('failed similar curvature previous')
                return False
        # Checking that starting points are close
        l_lane_pos = l_curve[0]*H**2+l_curve[1]*H+l_curve[2]
        r_lane_pos = r_curve[0]*H**2+r_curve[1]*H+r_curve[2]
        prev_l_lane_pos, prev_r_lane_pos = self.get_centers(H)
        if (l_lane_pos < 0.95*prev_l_lane_pos) | (l_lane_pos > 1.05*prev_l_lane_pos) | \
        (r_lane_pos < 0.95*prev_r_lane_pos) | (r_lane_pos > 1.05*prev_r_lane_pos):
            if self.verbose: print('failed starting points previous')
            return False
        return True

# Returns binary pixel locations in a certain window. Used by detect_lines() function below
def window_ind(width, height, img, center, level):
    y_ind, x_ind = img[int(img.shape[0]-(level+1)*height):int(img.shape[0]-level*height),
                           max(0,int(center-width/2)):min(int(center+width/2),img.shape[1])].nonzero()
    x_ind += max(0,int(center-width/2))
    y_ind += int(img.shape[0]-(level+1)*height)
    return x_ind, y_ind

# Detect lines based on a black and white warped image 
def detect_lines(img, win_w=50, win_h=80, margin=150, det_hist=detection_history()):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(win_w) # Create our window template that we will use for convolutions
    
    if det_hist.last_detection == False:
        # 'Blindly' search for lines
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(img[int(img.shape[0]/2):,:int(img.shape[1]/2)], axis=0)
        l_cen = np.argmax(np.convolve(window,l_sum))-win_w/2
        r_sum = np.sum(img[int(img.shape[0]/2):,int(img.shape[1]/2):], axis=0)
        r_cen = np.argmax(np.convolve(window,r_sum))-win_w/2+int(img.shape[1]/2)
        window_centroids.append((l_cen,r_cen))
        start = 1
    else:
        # search more accurately in a window around the last detection
        l_cen, r_cen = det_hist.get_centers(img.shape[0])
        if l_cen < 0: l_cen = 0
        if r_cen > img.shape[1]: r_cen = img.shape[1]
        start = 0
    
    # Go through each layer looking for max pixel locations
    for level in range(start,int(img.shape[0]/win_h)):
        # convolve the window into the vertical slice of the image
        img_layer   = np.sum(img[int(img.shape[0]-(level+1)*win_h):int(img.shape[0]-level*win_h),:], axis=0)
        conv_signal = np.convolve(window, img_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = win_w/2
        l_min_ind = int(max(l_cen+offset-margin,0))
        l_max_ind = int(min(l_cen+offset+margin,img.shape[1]))
        l_cen = np.argmax(conv_signal[l_min_ind:l_max_ind])+l_min_ind-offset
        # Find the best right centroid by using past right center as a reference
        r_min_ind = int(max(r_cen+offset-margin,0))
        r_max_ind = int(min(r_cen+offset+margin,img.shape[1]))
        r_cen = np.argmax(conv_signal[r_min_ind:r_max_ind])+r_min_ind-offset
        # Add what we found for that layer
        window_centroids.append((l_cen,r_cen))

    # If we found any window centers
    if len(window_centroids) > 0:
        
        # Points used to draw all the left and right windows
        l_ind_x  = []
        l_ind_y  = []
        r_ind_x  = []
        r_ind_y  = []
        
        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # window_ind is a function that returns nonzero pixel indices within a window
            x_ind, y_ind = window_ind(win_w,win_h,img,window_centroids[level][0],level)
            l_ind_x.append(x_ind); l_ind_y.append(y_ind)
            x_ind, y_ind = window_ind(win_w,win_h,img,window_centroids[level][1],level)
            r_ind_x.append(x_ind); r_ind_y.append(y_ind)
    
    l_ind_y = np.concatenate(l_ind_y)
    l_ind_x = np.concatenate(l_ind_x)
    r_ind_y = np.concatenate(r_ind_y)
    r_ind_x = np.concatenate(r_ind_x)
    
    # fit curves
    l_curve = np.array([0,0,0]) if (l_ind_x.shape[0] < 1) else np.polyfit(l_ind_y, l_ind_x, 2)
    r_curve = np.array([0,0,0]) if (r_ind_x.shape[0] < 1) else np.polyfit(r_ind_y, r_ind_x, 2)
    
    return l_curve, r_curve, l_ind_y, l_ind_x, r_ind_y, r_ind_x

# Calculate radii in meters
def detect_radii(l_curve, r_curve, H=720, ym_per_pix=30/720, xm_per_pix=3.7/700):
    curve_y   = np.linspace(0, H-1, H)
    l_curve_x = l_curve[0]*curve_y**2 + l_curve[1]*curve_y + l_curve[2]
    r_curve_x = r_curve[0]*curve_y**2 + r_curve[1]*curve_y + r_curve[2]
    # Fit new polynomials to x,y in world space
    l_curve_m = np.polyfit(curve_y*ym_per_pix, l_curve_x*xm_per_pix, 2)
    r_curve_m = np.polyfit(curve_y*ym_per_pix, r_curve_x*xm_per_pix, 2)
    # Calculate the new radii of curvature, all the way at the bottom of the detection (max y value)
    l_curve_rad = ((1 + (2*l_curve_m[0]*np.max(curve_y)*ym_per_pix + l_curve_m[1])**2)**1.5) \
    / np.absolute(2*l_curve_m[0])
    r_curve_rad = ((1 + (2*r_curve_m[0]*np.max(curve_y)*ym_per_pix + r_curve_m[1])**2)**1.5) \
    / np.absolute(2*r_curve_m[0])
    # Now our radius of curvature is in meters
    return l_curve_rad, r_curve_rad

# Annotate all findings on an image
def annotate(img, wrp_H, wrp_W, det_hist):
    H = img.shape[0]
    # Recast the x and y curve points into usable format for cv2.fillPoly()
    curve_y   = np.linspace(0, H-1, H)
    l_curve_x = det_hist.l_curve[0]*curve_y**2 + det_hist.l_curve[1]*curve_y + det_hist.l_curve[2]
    r_curve_x = det_hist.r_curve[0]*curve_y**2 + det_hist.r_curve[1]*curve_y + det_hist.r_curve[2]
    l_pts = np.array([np.transpose(np.vstack([l_curve_x, curve_y]))])
    r_pts = np.array([np.flipud(np.transpose(np.vstack([r_curve_x, curve_y])))])
    pts   = np.hstack((l_pts, r_pts))
    annotations = np.zeros((wrp_H, wrp_W, 3), dtype=np.uint8)
    cv2.fillPoly(annotations, np.int_([pts]), (0, 255, 0))
    
    # Warp back the annotations and add them to the original picture
    unwrp_ann = warp_image(annotations, warp_back=True) # warp back the annotations to the original format
    annotated = cv2.addWeighted(img, 1, unwrp_ann, 0.3, 0.0) # overlay the orignal road image with window results
    
    # Add textual annotations for curvature radius and car distance from center
    curve_str  = '{:.0f}m'.format(det_hist.curve_radius) if det_hist.curve_radius <2000 else '-'
    curve_str  = 'Average curve radius: '+curve_str
    cv2.putText(annotated,curve_str,(30,60), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),3)
    dist_str   = 'Distance from lane center: {:.2f}m'.format(det_hist.lane_center)
    cv2.putText(annotated,dist_str,(30,100), cv2.FONT_HERSHEY_PLAIN, 3,(255,255,255),3)
    
    return annotated

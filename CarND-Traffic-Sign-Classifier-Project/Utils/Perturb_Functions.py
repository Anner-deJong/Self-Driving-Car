"""
Helper file with perturbation functions for image data augmentation
"""

import numpy as np
import scipy.stats as stats
import cv2
import math

def image_Single_Flip(img):
    '''
    Flip image
    input  - Image in shape H, W, C
    return - Image in same shape, with W inverted
    '''
    return np.vstack((img[None, :], img[None, :, ::-1, :]))

def image_Single_Translate(img, crop_ratio): #### ACTUALLY IMPLEMENTS A CROP!!
    '''
    Translate image
    input  - Image in shape H, W, C
    return - Image in smaller shape, in each of 4 corners and center centere
    '''

    H, W, C = img.shape[-3:]
    h       = round(H * crop_ratio)
    w       = round(W * crop_ratio)
    h_c     = round(H * (1-crop_ratio) / 2)
    w_c     = round(W * (1-crop_ratio) / 2)

    img_trans  = np.zeros((6, h, w, C))

    img_trans[0, ] = cv2.resize(img, (w, h))
    img_trans[1, ] = img[  0:0+h,     0:0+w,   :]
    img_trans[2, ] = img[  0:0+h,   W-w:W,     :]
    img_trans[3, ] = img[H-h:H,       0:0+w,   :]
    img_trans[4, ] = img[H-h:H,     W-w:W,     :]
    img_trans[5, ] = img[h_c:h_c+h, w_c:w_c+w, :]

    return img_trans

def image_Single_Rotate(img, angles):
    '''
    rotate image
    input  - Image in shape H, W, C
    return - Image rotated with random
    '''
    H, W, C = img.shape[-3:]
    background_color = img[3, 3, :].tolist()

    img_rot = np.zeros((len(angles), H, W, C))

    for i, angle in enumerate(angles):
        M   = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (W, H),
                             borderMode  = cv2.BORDER_REPLICATE)
        img_rot[i, ] = rot

    return img_rot

def image_Single_Perspective(img): #fancy PCA?

    H, W = img.shape[-2:]
    background_color = 0

    lower, upper, mu, sigma = 0, 0.95, 0.4, 0.3 # alpha distribution
    alpha = lambda : stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # horizontal size reduction factor
    beta  = lambda a: 0.9 + 0.1 * a # vertical size reduction factor
    gamma = lambda a: 0.5 - 0.5 * a # uplift factor for turned away edge
    max_height   = math.ceil(H * (beta(a=0) + gamma(a=0)))
    img_persp    = np.zeros((7, max_height, W))
    img_persp[0] = cv2.resize(img, (W, max_height))

    for i in range(3):

        # right side away turn
        alp            = alpha()
        rect           = np.array([[0, 0],  [W, 0], [0, H], [W, H]], dtype="float32")
        dst            = np.array([[0, 0], [W * alp, H * (1 - beta(alp) - gamma(alp))], [0, H], [W * alp, H * (1 - gamma(alp))]], dtype="float32")
        M              = cv2.getPerspectiveTransform(rect, dst)
        img_persp[1+i] = cv2.resize(cv2.warpPerspective(img, M, (W, H),
                                                        borderMode  = cv2.BORDER_CONSTANT,
                                                        borderValue = background_color),
                                    (W, max_height))
        # left side away turn
        alp            = alpha()
        rect           = np.array([[0, 0], [W, 0], [0, H], [W, H]], dtype="float32")
        dst            = np.array([[W * (1 - alp), H * (1 - beta(alp) - gamma(alp))], [W, 0], [W * (1 - alp), H * (1 - gamma(alp))], [W, H]], dtype="float32")
        M              = cv2.getPerspectiveTransform(rect, dst)
        img_persp[4+i] = cv2.resize(cv2.warpPerspective(img, M, (W, H),
                                                        borderMode=cv2.BORDER_CONSTANT,
                                                        borderValue=background_color),
                                    (W, max_height))

    return img_persp

def image_Single_Padding(img):
    '''
    Adds random padding to image 5 times, tl, tr, bl, br, center, each with random factor
    input  - Image in shape [H, W]
    return - 5 images in shape [5, H, W], randomly zoomed in 5 directions
    '''
    H, W    = img.shape
    max_pad = [H, H]

    top     = max_pad[0]
    bottom  = max_pad[0]
    left    = max_pad[1]
    right   = max_pad[1]

    background_color = 0

    img_pad = np.zeros((5, H + max_pad[0], W + max_pad[1]))

    img_pad[0, ] = cv2.resize(cv2.copyMakeBorder(img, round(top * np.random.uniform()), 0, round(left * np.random.uniform()), 0,
                                                 borderType  = cv2.BORDER_CONSTANT),
                              (W + max_pad[1], H + max_pad[0]))

    img_pad[1, ] = cv2.resize(cv2.copyMakeBorder(img, round(top * np.random.uniform()), 0, 0, round(right * np.random.uniform()),
                                                 borderType = cv2.BORDER_CONSTANT),
                              (W + max_pad[1], H + max_pad[0]))

    img_pad[2, ] = cv2.resize(cv2.copyMakeBorder(img, 0, round(bottom * np.random.uniform()), round(left * np.random.uniform()), 0,
                                                 borderType = cv2.BORDER_CONSTANT),
                              (W + max_pad[1], H + max_pad[0]))

    img_pad[3, ] = cv2.resize(cv2.copyMakeBorder(img, 0, round(bottom * np.random.uniform()), 0, round(right * np.random.uniform()),
                                                 borderType = cv2.BORDER_CONSTANT),
                              (W + max_pad[1], H + max_pad[0]))

    img_pad[4, ] = cv2.resize(cv2.copyMakeBorder(img,
                                                 top    = round(top/2 * np.random.uniform()),
                                                 bottom = round(bottom/2  * np.random.uniform()),
                                                 left   = round(left/2 * np.random.uniform()),
                                                 right  = round(right/2 * np.random.uniform()),
                                                 borderType = cv2.BORDER_CONSTANT),
                              (W + max_pad[1], H + max_pad[0]))
    return img_pad

def image_Single_Scale(img, scale=[0.9, 1.1]):
    '''
    Randomly zooms in OR out of an image, and rescales to original image size
    input  - Image in shape [H, W, C]
    return - 2 images in shape [2, H, W, C]
    '''
    H, W, C = img.shape
    img_sca = np.zeros((2, H, W, C))

    # scale up
    H_i = (int(H*scale[1]) - H)//2 # start index for the bigger array
    W_i = (int(W*scale[1]) - W)//2 # start index for the bigger array
    img_sca[0] = cv2.resize(  img, (int(W*scale[1]), int(H*scale[1]))  )[ H_i:H_i+H, W_i :W_i+W]

    # scale down
    tp = (H - int(H*scale[0]))//2; bp = H-int(H*scale[0])-tp    # top & bottom padding
    lp = (W - int(W*scale[0]))//2; rp = W-int(W*scale[0])-lp    # left & right padding
    img_sca[1] = cv2.copyMakeBorder(cv2.resize(img, (int(W*scale[0]), int(H*scale[0])) ),
            tp, bp, lp, rp, cv2.BORDER_REPLICATE)

    return img_sca

def shift_hor(img, shift):
    if shift>0:
        return cv2.copyMakeBorder(img[:, :-shift], 0, 0,  shift, 0, cv2.BORDER_REPLICATE)
    elif shift<0:
        return cv2.copyMakeBorder(img[:, -shift:], 0, 0, 0, -shift, cv2.BORDER_REPLICATE)
    else:
        raise NameError('shift is 0!')

def shift_ver(img, shift):
    if shift>0:
        return cv2.copyMakeBorder(img[shift:, :], 0,  shift, 0, 0, cv2.BORDER_REPLICATE)
    elif shift<0:
        return cv2.copyMakeBorder(img[:shift, :], -shift, 0, 0, 0, cv2.BORDER_REPLICATE)
    else:
        raise NameError('shift is 0!')

def image_Single_Shift(img, pixels=2):
    '''
    Randomly shifts the image a 'pixels' amount of pixels
    cuts off on one side and pads on the other side
    input  - Image in shape [H, W, C]
    return - 2 images in shape [2, H, W, C]
    '''
    H, W, C = img.shape
    img_sca = np.zeros((2, H, W, C))

    # defining the directions of the shifts (x1, y1), (x2, y2)
    switch = [1 if x > 0.5 else -1 for x in np.random.randn(4)]
    shifts = (np.random.randint(1, pixels+1, 4))* switch

    # making sure both directions are not the same
    if (shifts[0]>0) & (shifts[2]>0) | (shifts[0]<0) & (shifts[2]<0):
        if (shifts[1]>0) & (shifts[3]>0) | (shifts[1]<0) & (shifts[3]<0):
            cor = np.random.randint(2, 4)
            shifts[cor] *= -1

    # scale the actual image with the helper functions
    img_sca[0, ] = shift_ver(shift_hor(img, shifts[0]), shifts[1])
    img_sca[1, ] = shift_ver(shift_hor(img, shifts[2]), shifts[3])

    return img_sca

def image_Flip(img, lbl):
    '''
    Flip image
    input  - Image in shape H, W, C
    return - Image in same shape, with W inverted
    '''
    N_yn = len(img.shape) - 3

    if N_yn:
        N = img.shape[0]
        img_flip_list = []
        lbl_flip_list = []

        for n in range(N):
            cur_img = img[n, ]
            img_flip_list.append(image_Single_Flip(cur_img))
            lbl_flip_list.append(np.tile(lbl[n, ], (img_flip_list[-1].shape[0], 1)))

        img_flip = np.vstack(img_flip_list)
        lbl_flip = np.vstack(lbl_flip_list)

    else:
        img_flip = image_Single_Flip(img)
        lbl_flip = np.tile(lbl, (2, 1))

    return img_flip, lbl_flip

def image_Translate(img, lbl, crop_ratio): #### ACTUALLY IMPLEMENTS A CROP!!
    '''
    Translate image
    input  - Image in shape H, W, C
    return - Image in smaller shape, in each of 4 corners and center centere
    '''
    N_yn    = len(img.shape) - 3

    if N_yn:
        N = img.shape[0]
        img_trans_list = []
        lbl_trans_list = []

        for n in range(N):
            cur_img    = img[n, ]
            img_trans_list.append(image_Single_Translate(cur_img, crop_ratio))
            lbl_trans_list.append(np.tile(lbl[n, ], (img_trans_list[-1].shape[0], 1)))

        img_trans = np.vstack(img_trans_list)
        lbl_trans = np.vstack(lbl_trans_list)

    else:
        img_trans = image_Single_Translate(img, crop_ratio)
        lbl_trans = np.tile(lbl, (2, 1))

    return img_trans, lbl_trans

def image_Rotate(img, lbl, angles):
    '''
    rotate image
    input  - Image in shape H, W, C
    return - Image rotated with random
    '''
    N_yn = len(img.shape) - 3
    if len(lbl.shape) < 2: lbl = lbl[:, None]

    if N_yn:
        N = img.shape[0]
        img_rot_list = []
        lbl_rot_list = []

        for n in range(N):
            cur_img = img[n,]
            img_rot_list.append(image_Single_Rotate(cur_img, angles))
            lbl_rot_list.append(np.tile(lbl[n,], (img_rot_list[-1].shape[0], 1)))

        img_rot = np.vstack(img_rot_list)
        lbl_rot = np.vstack(lbl_rot_list)

    else:
        img_rot = image_Single_Rotate(img, angles)
        lbl_rot = np.tile(lbl, (2, 1))

    return img_rot, lbl_rot

def image_Perspective(img):
    '''
    Translate image
    input  - Image in shape H, W
    return - Image
    '''
    N_yn = len(img.shape) - 2

    if N_yn:
        N = img.shape[0]
        img_persp_list = []

        for n in range(N):
            cur_img    = img[n, ]
            img_persp_list.append(image_Single_Perspective(cur_img))

        img_persp = np.vstack(img_persp_list)

    else:
        img_persp = image_Single_Perspective(img)

    return img_persp

def image_Padding(img):
    '''
    Adds random padding to image 5 times, tl, tr, bl, br, center, each with random factor
    input  - Image in shape [H, W]
    return - 5 images in shape [5, H, W]
    '''
    N_yn = len(img.shape) - 2

    if N_yn:
        N = img.shape[0]
        img_pad_list = []

        for n in range(N):
            cur_img    = img[n, ]
            img_pad_list.append(image_Single_Padding(cur_img))

        img_pad = np.vstack(img_pad_list)

    else:
        img_pad = image_Single_Padding(img)

    return img_pad

def image_Scale(img, lbl, scale=[0.9, 1.1]):
    '''
    scaled images
    input  - N images in shape
    return - N images in same shape
    '''
    N_yn = len(img.shape) - 3
    if len(lbl.shape) < 2: lbl = lbl[:, None]

    if N_yn:
        N = img.shape[0]
        img_sca_list = []
        lbl_sca_list = []

        for n in range(N):
            cur_img = img[n,]
            img_sca_list.append(image_Single_Scale(cur_img, scale))
            lbl_sca_list.append(np.tile(lbl[n,], (img_sca_list[-1].shape[0], 1)))
        img_sca = np.vstack(img_sca_list)
        lbl_sca = np.vstack(lbl_sca_list)

    else:
        img_sca = image_Single_Scale(img, scale)
        lbl_sca = np.tile(lbl, (2, 1))

    return img_sca, lbl_sca

def image_Shift(img, lbl, pixels=2):
    '''
    scaled images
    input  - N images in shape
    return - N images in same shape
    '''
    N_yn = len(img.shape) - 3
    if len(lbl.shape) < 2: lbl = lbl[:, None]

    if N_yn:
        N = img.shape[0]
        img_shi_list = []
        lbl_shi_list = []

        for n in range(N):
            cur_img = img[n,]
            img_shi_list.append(image_Single_Shift(cur_img, pixels))
            lbl_shi_list.append(np.tile(lbl[n,], (img_shi_list[-1].shape[0], 1)))

        img_shi = np.vstack(img_shi_list)
        lbl_shi = np.vstack(lbl_shi_list)

    else:
        img_shi = image_Single_Shift(img, pixels)
        lbl_shi = np.tile(lbl, (2, 1))

    return img_shi, lbl_shi

def image_Reshape(img, lbl, size, flatten=False):
    '''
    reshape and resizes the final images
    input  - N images in shape [-, -]
    return - N images in shape (size)
    '''
    N         = img.shape[0]

    if flatten:
        augm_data = np.zeros((N, np.prod(size) * 3))
        for n in range(N):
            augm_data[n] = cv2.resize(img[n], size).flatten()
    else:
        augm_data = np.zeros(([N, size[0], size[1], 3]))
        for n in range(N):
            augm_data[n] = cv2.resize(img[n], size)

    return augm_data, lbl

def image_Color_Perturb(img, lbl, no_pert):

    if (len(img.shape) - 3) < 1:
        raise NotImplementedError

    N, H, W, C     = img.shape
    img_norm       = np.reshape(img, (N*H*W, C)).astype('float64')
    img_norm      -= img_norm.mean(0)

    img_cp         = np.tile(img, (no_pert + 1, 1, 1, 1))
    lbl_rot        = np.tile(lbl, (no_pert + 1, 1))

    # with help from https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    cov            = np.cov(img_norm.T)
    eigval, eigvec = np.linalg.eigh(cov)

    for i in range(no_pert):
        rand_a                 = np.random.randn(3, N) * 0.1
        factor                 = np.multiply(np.sqrt(eigval)[:, None], rand_a)
        perturb                = np.dot(eigvec, factor).T

        img_cp[i*N : i*N + N] += perturb[:, None, None, :]

    return img_cp, lbl_rot

def image_normalize(img):

    N, W, H, C = img.shape
    img_norm   = np.reshape(img, (N*W*H, C))
    img_norm  -= img_norm.mean(0)

    return img_norm

def image_Noise(img, lbl, std):

    img_noi = img + np.random.normal(0.0, std)

    return img_noi, lbl

# Zoom

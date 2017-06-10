import numpy as np
import glob
import matplotlib.image as mpimg
import cv2


def undistort(img, objpoints, imgpoints):
    # Get the size of the image
    img_size = (img.shape[1], img.shape[0])
    # Get distortion coefficients (dist) and camera matrix (mtx)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # Calibrate Camera
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def perspective_transform(img):
    # four original source points to transform.
    calib_img_x = img.shape[1]
    calib_img_y = img.shape[0]
    margin = 0.05

    src_top_right = [750, 450]
    src_bot_right = [1200, 680]
    src_bot_left = [100, 680]
    src_top_left = [550, 450]

    src = np.float32([src_top_right,
                      src_bot_right,
                      src_bot_left,
                      src_top_left])

    # four destination points
    dst_top_right = [int(calib_img_x * (1 - margin)), 0]
    dst_bot_right = [int(calib_img_x * (1 - margin)) - 125, calib_img_y]
    dst_bot_left = [int(calib_img_x * margin) + 125, calib_img_y]
    dst_top_left = [int(calib_img_x * margin), 0]
    dst = np.float32([dst_top_right,
                      dst_bot_right,
                      dst_bot_left,
                      dst_top_left])

    ## Compute and apply perspective transform
    img_size = (calib_img_x, calib_img_y)
    # Get Transform Matrix and its Inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Return Warped Image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return Minv, warped


def mask(img):
    # Thresholds
    yellow_lower_filter = np.array([0, 100, 100])
    yellow_upper_filter = np.array([80, 255, 255])

    white_lower_filter = np.array([200, 200, 200])
    white_upper_filter = np.array([255, 255, 255])

    # yellow masking
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_mask = cv2.inRange(hsv, yellow_lower_filter, yellow_upper_filter)
    yellow_mask = cv2.bitwise_and(img, img, mask=yellow_mask)

    # white masking
    rgb = img
    white_mask = cv2.inRange(rgb, white_lower_filter, white_upper_filter)
    white_mask = cv2.bitwise_and(img, img, mask=white_mask)

    # combined masks
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # convert to binary image
    gray_mask = cv2.cvtColor(combined_mask, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray_mask)
    binary[(gray_mask > 0)] = 1
    return yellow_mask, white_mask, binary


def final_mask(img):
    # Color masking
    y, w, color_mask = mask(img)
    # Sobel Masking
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    sobelx_light_mask = abs_sobel_thresh(l, orient='x', thresh=(50, 225))
    sobelx_saturation_mask = abs_sobel_thresh(s, orient='x', thresh=(50, 225))
    combined_sobel = cv2.bitwise_or(sobelx_light_mask, sobelx_saturation_mask)

    # Final Mask
    final_mask = cv2.bitwise_or(combined_sobel, color_mask)
    binary_warped = np.copy(final_mask)

    return binary_warped


def gaussian_blur(img, kernel=5):
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return blur


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Applying gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create copy and apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # apply a threshold, and create a binary image result
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

import numpy as np
import os
import pickle
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import cv2
import time
from scipy.signal import find_peaks_cwt


def calibrate_camera():
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    cal_img = mpimg.imread(images[0])
    t = time.time()
    # Step through the list and search for chessboard corners
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)

            # Get the size of the image
    img_size = (cal_img.shape[1], cal_img.shape[0])
    # Get distortion coefficients (dist) and camera matrix (mtx)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    t2 = time.time()
    print('Finished getting camera coefficients, time take was {} '.format(t2 - t))
    # Save calibration coefficients in dictionary pickle file
    coefficeints_pickle = {}
    coefficeints_pickle["dist"] = dist
    coefficeints_pickle['mtx'] = mtx
    pickle.dump(coefficeints_pickle, open('camera_coeff.p', 'wb'))


def get_calibration(return_time=False):
    # Starting time to check how long process takes
    t = time.time()
    # Checks if pickle file already exists
    if not (os.path.isfile('camera_coeff.p')):
        calibrate_camera()
        return get_calibration()
    else:
        # Returns mtx and dist from pickle file
        coefficeints_pickle = pickle.load(open('camera_coeff.p', 'rb'))
        mtx, dist = coefficeints_pickle['mtx'], coefficeints_pickle['dist']
        t2 = time.time()
        time_taken = t2 - t
        if return_time == True:
            print('Loading camera calibration coefficients')
            print("Time take to load is {}".format(time_taken))
        return mtx, dist


def undistort_image(img, mtx, dist):
    # Calibrate Camera
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted


def get_video_frames(video_path,total_frames=400, chosen_frame=None):
    # Grabs indiviual video frames for debugging
    video = cv2.VideoCapture(video_path)
    i = 0
    #for specific frame
    if not (chosen_frame == None):
        pass
    else:
        while i < total_frames:
            if not (os.path.isfile("frame{}.jpg".format(i))):
                ret, frame = video.read()
                cv2.imwrite("frame{}.jpg".format(i), frame)
            i = i + 1


def perspective_transform(img, src, dst):
    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
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
    binary_warped = binary_warped[:, :]

    return binary_warped


def moving_average(a, n=25):
    # Moving average
    average = np.cumsum(a, dtype=float)
    average[n:] = average[n:] - average[:-n]
    return average[n - 1:] / n


def first_frame_detection(binary_warped):
    img_sizex = binary_warped.shape[1]
    img_sizey = binary_warped.shape[0]
    bottom_half = binary_warped[img_sizey // 2:, :]

    initial_peaks = np.sum(bottom_half, axis=0)
    initial_peaks = moving_average(initial_peaks)

    # Get peaks of highest pixel concentration
    peak_indexes = find_peaks_cwt(initial_peaks, [100], max_distances=[800])

    index_right_peak = peak_indexes[0]
    index_left_peak = peak_indexes[1]
    # Peaks index, left should be smaller then right, if not switch
    if index_right_peak < index_left_peak:
        temporary_index = index_right_peak
        index_right_peak = index_left_peak
        index_left_peak = temporary_index
    # Zeros array to place mask in
    left_mask = np.zeros_like(binary_warped)
    right_mask = np.zeros_like(binary_warped)
    # Storing peaks to help with outliers
    index_prev_right_peak = index_right_peak
    index_prev_left_peak = index_left_peak

    window_size = 30
    nwindows = 8

    plt.figure(figsize=(10, 6))
    # Slide window up for each 1/8th of image
    for window in range(nwindows):
        window_y_low = int(img_sizey - img_sizey * window / nwindows)
        window_y_high = int(img_sizey - img_sizey * (window + 1) / nwindows)

        # Histogram of image to find peaks
        window_histogram = np.sum(binary_warped[window_y_high:window_y_low, :], axis=0)
        window_histogram = moving_average(window_histogram)
        peak_indexes = find_peaks_cwt(window_histogram, [100], max_distances=[800])

        # If 2 peaks are detected make sure left side index is larger then right side index.
        if len(peak_indexes) > 1.5:
            index_right_peak = peak_indexes[0]
            index_left_peak = peak_indexes[1]
            if index_right_peak < index_left_peak:
                temporary_index = index_right_peak
                index_right_peak = index_left_peak
                index_left_peak = temporary_index

        else:
            # If 1 peak is detected, assign peak to closest peak in previous iteration of the image.
            if len(peak_indexes) == 1:
                if np.abs(peak_indexes[0] - index_prev_right_peak) < np.abs(peak_indexes[0] - index_prev_left_peak):
                    index_right_peak = peak_indexes[0]
                    index_left_peak = index_prev_left_peak
                else:
                    index_left_peak = peak_indexes[0]
                    index_right_peak = index_prev_right_peak
            # If no peaks are found then assign previous peaks
            else:
                index_left_peak = index_prev_left_peak
                index_right_peak = index_prev_right_peak
        # Rejects peaks that deviate too far (60 pixels) from the previous peak to ensure bad frames don't ruin the algorithm
        if np.abs(index_left_peak - index_prev_left_peak) >= 60:
            index_left_peak = index_prev_left_peak
        if np.abs(index_right_peak - index_prev_right_peak) >= 60:
            index_right_peak = index_prev_right_peak

        # Binary image showing a window for the left and right lane of the chosen window size
        left_window_x1 = index_left_peak - window_size
        left_window_x2 = window_size + index_left_peak

        right_window_x1 = index_right_peak - window_size
        right_window_x2 = window_size + index_right_peak

        left_mask[window_y_high:window_y_low, left_window_x1:left_window_x2] = 1.
        right_mask[window_y_high:window_y_low, right_window_x1:right_window_x2] = 1.

        # Store index values of found peaks to be used in next iteration
        index_prev_left_peak = index_left_peak
        index_prev_right_peak = index_right_peak

    return left_mask, right_mask


def following_frames(binary_warped, fit_line):
    # This function returns masks based on the previous polynomial fit which will be applied to upcoming polynomial fit.
    mask_poly = np.zeros_like(binary_warped)
    img_size = np.shape(binary_warped)

    for window in range(8):
        # Y position of window and width
        win_y_low = int(img_size[0] - img_size[0] * window / 8)
        win_y_high = int(img_size[0] - img_size[0] * (window + 1) / 8)
        window_width = 30

        middle_y = (win_y_low + win_y_high) / 2
        # Predicted middle of window based on found fit line
        poly_pt = np.round(fit_line[0] * middle_y ** 2 + fit_line[1] * middle_y + fit_line[2])

        # Binary image
        mask_poly[win_y_high:win_y_low, int(poly_pt - window_width):int(poly_pt + window_width)] = 1.

    return mask_poly


def poly_curvature(fit_poly, y_point):
    # Returns radius of curvature of found polynomial fit
    A = fit_poly[0]
    B = fit_poly[1]
    curvature = (1 + (2 * A * y_point + B) ** 2) ** 1.5 / 2 / A
    return curvature


def draw_lane(image, binary_warped, Minv, left_fit, right_fit, frame):
    # left and right lane x values
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    # Not currently used
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Find the center of the lane
    midpoint = np.int(image.shape[1] / 2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

    # Left Curvature
    left_curverad = poly_curvature(left_fit, image.shape[1]) / 1000  # in kilometers
    if np.abs(left_curverad) > 15:
        left_curverad = 'Straight Section'
        left_text = "Left Lane Radius: " + "{}".format(left_curverad)
    else:
        left_text = "Left Lane Radius: " + "{:0.2f}".format(left_curverad) + 'km'
    # Right Curvature
    right_curverad = poly_curvature(right_fit, image.shape[1]) / 1000  # in kilometers
    if np.abs(right_curverad) > 15:
        right_curverad = 'Straight Section'
        right_text = "Right Lane Radius: " + "{}".format(right_curverad)
    else:
        right_text = "Right Lane Radius: " + "{:0.2f}".format(right_curverad) + 'km'

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    non_persp_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    cv2.putText(result, left_text, org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(result, right_text, org=(50, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(result, "Lane Center: " + "{:0.2f}".format(offset) + 'm', org=(50, 150),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(result, "Frame Number: " + "{}".format(frame), org=(50, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    return result, non_persp_warp


def diagnostic(image_list):
    # Diagnoistc screen for debugging
    img_height = 1080
    img_width = 1920

    diagScreen = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    diag0 = image_list[0]
    diag1 = image_list[1]
    diag2 = image_list[2]
    diag3 = image_list[3]

    diagScreen[0:540, 0:960] = cv2.resize(diag0, (960, 540), interpolation=cv2.INTER_AREA)
    diagScreen[0:540, 960:1920] = cv2.resize(diag1, (960, 540), interpolation=cv2.INTER_AREA)
    diagScreen[540:1080, 960:1920] = cv2.resize(diag2, (960, 540), interpolation=cv2.INTER_AREA)
    diagScreen[540:1080, 0:960] = cv2.resize(diag3, (960, 540), interpolation=cv2.INTER_AREA)

    return diagScreen


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


def image_comparison(image1, title1, image2, title2):
    # Gridspec to customize image location
    plt.figure(figsize=(15, 10))
    gs1 = gs.GridSpec(nrows=1, ncols=2)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_title(title1)
    plt.imshow(image1)
    plt.axis('off')

    ax2 = plt.subplot(gs1[0, 1])
    ax2.set_title(title2)
    plt.imshow(image2)
    plt.axis('off')

    plt.show()

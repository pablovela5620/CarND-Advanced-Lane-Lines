import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gs
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from helper_function import undistort, perspective_transform, final_mask

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')
drawn_images = []

# Step through the list and search for chessboard corners
for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        drawn_images.append(img)


def sliding_window(binary_warped):
    out_img = np.dstack((binary_warped,  # Create an output image to draw on and visualize result
                         binary_warped,
                         binary_warped)) * 255
    #Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_fitx, right_fitx, ploty, left_curverad, right_curverad


def draw_lane(image, binary_warped, Minv, left_fitx, right_fitx, ploty, left_curverad, right_curverad):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    #Find the center of the lane
    midpoint = np.int(image.shape[1] / 2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    cv2.putText(result, "Left Lane Radius: " + "{:0.2f}".format(left_curverad / 1000) + 'km', org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(result, "Right Lane Radius: " + "{:0.2f}".format(right_curverad / 1000) + 'km', org=(50, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    cv2.putText(result, "Lane Center: " + "{:0.2f}".format(offset) + 'm', org=(50, 150),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), lineType=cv2.LINE_AA, thickness=2)
    return result


def pipeline(image, video=True):
    if video == False:
        # Load Image
        image = mpimg.imread(image)

        # Undistort Image
        undist = undistort(image, objpoints, imgpoints)

        # Perspective Transform
        Minv, birds_view = perspective_transform(undist)

        # Masking
        binary_warped = final_mask(birds_view)

        # Find corresponding lane pixels
        left_fitx, right_fitx, ploty, left_curverad, right_curverad = sliding_window(binary_warped)

        # Draw lane lines onto image
        drawn_lines = draw_lane(image, binary_warped, Minv, left_fitx, right_fitx, ploty, left_curverad, right_curverad)
        return drawn_lines




    else:
        # Undistort Image
        undist = undistort(image, objpoints, imgpoints)

        # Perspective Transform
        Minv, birds_view = perspective_transform(undist)

        # Masking
        binary_warped = final_mask(birds_view)

        # Find corresponding lane pixels
        left_fitx, right_fitx, ploty, left_curverad, right_curverad = sliding_window(binary_warped)

        # Draw lane lines onto image
        drawn_lines = draw_lane(image, binary_warped, Minv, left_fitx, right_fitx, ploty, left_curverad, right_curverad)
        return drawn_lines


def show_test_imgs(images):
    plt.figure(figsize=(15, 10))
    gs1 = gs.GridSpec(nrows=3, ncols=3)

    for i in range(0, len(images)):
        ax = plt.subplot(gs1[i])
        plt.imshow(images[i])
    plt.show()


test_imgs = glob.glob('test_images/test*.jpg')
straight_imgs = glob.glob('test_images/straight_lines*.jpg')
all_imgs = straight_imgs + test_imgs

mod_imgs = []

for img in all_imgs:
    mod_imgs.append(pipeline(img, video=False))

if __name__ == '__main__':
     # plt.imshow(mod_imgs[0])
     # plt.show()

    # show_test_imgs(mod_imgs)

    # output_video = 'video_output/new_project_video_output.mp4'
    # output_clip = VideoFileClip('project_video.mp4')
    # project_clip = output_clip.fl_image(pipeline)  # NOTE: this function expects color images!!
    # project_clip.write_videofile(output_video, audio=False)

    challenge_video = 'video_output/new_challenge_video_output.mp4'
    challenge_clip = VideoFileClip('challenge_video.mp4')
    project_clip = challenge_clip.fl_image(pipeline)  # NOTE: this function expects color images!!
    project_clip.write_videofile(challenge_video, audio=False)

    harder_video = 'video_output/new_harder_video_output.mp4'
    harder_clip = VideoFileClip('harder_challenge_video.mp4')
    project_clip = harder_clip.fl_image(pipeline)  # NOTE: this function expects color images!!
    project_clip.write_videofile(harder_video, audio=False)

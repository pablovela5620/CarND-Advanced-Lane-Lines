import os
import time
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from helper_function import get_calibration, undistort_image, image_comparison, perspective_transform, final_mask, \
    first_frame_detection, draw_lane, following_frames, get_video_frames, diagnostic


def lanefinding_pipeline(image):
    # Global Variables to store fit polynomials and found masks
    global frame
    global previous_left_fit
    global previous_right_fit
    global left_mask
    global right_mask
    global diagnostic_screen

    # Undistort image
    undist = undistort_image(image, mtx, dist)

    # Get perspective transform and inverse matrix
    Minv, persp = perspective_transform(undist, src, dst)

    # Color and sobel thresholding
    binary_warped = final_mask(persp)
    out_img = np.dstack((binary_warped,  # Create an output image to draw on and visualize result
                         binary_warped,
                         binary_warped)) * 255

    # Initial lane finding using computer vision techniques on the first frame
    if frame == 0:
        left_mask, right_mask = first_frame_detection(binary_warped)

    # Right lane polynomial fit
    right_lane = cv2.bitwise_and(binary_warped, right_mask)
    # All nonzero pixels found in image where both mask and binary warped are nonzero
    right_nonzero = right_lane.nonzero()

    # Ensures that the first frame has a polynomial fit
    if frame == 0:
        rightx = np.array(right_nonzero[0])
        righty = np.array(right_nonzero[1])
        right_fit = np.polyfit(rightx, righty, 2)
    # Rejects found polynomial if there are less then 5000 pixels and uses previously found
    elif len(right_nonzero[0]) < 5000:
        right_fit = previous_right_fit
    else:
        rightx = np.array(right_nonzero[0])
        righty = np.array(right_nonzero[1])
        right_fit = np.polyfit(rightx, righty, 2)

    # Left lane polynomial fit
    left_lane = cv2.bitwise_and(binary_warped, left_mask)
    # All nonzero pixels found in image where both mask and binary warped are nonzero
    left_nonzero = left_lane.nonzero()

    # Ensures that the first frame has a polynomial fit
    if frame == 0:
        leftx = np.array(left_nonzero[0])
        lefty = np.array(left_nonzero[1])
        left_fit = np.polyfit(leftx, lefty, 2)
    # Rejects found polynomial if there are less then 5000 pixels and uses previously found
    elif len(left_nonzero[0]) < 2000:
        left_fit = previous_left_fit
    else:
        leftx = np.array(left_nonzero[0])
        lefty = np.array(left_nonzero[1])
        left_fit = np.polyfit(leftx, lefty, 2)

    # If first frame of video, increase frame  and saves found polynomial fit
    if frame == 0:
        frame = frame + 1
        previous_left_fit = left_fit
        previous_right_fit = right_fit
    else:
        frame = frame + 1

    result, non_persp_warp = draw_lane(image, binary_warped, Minv, left_fit, right_fit, frame)

    # Computing the masks after the first frame to significantly reduce computation time using the first found polynomial fit and basing mask from this
    left_mask = following_frames(binary_warped, left_fit)
    right_mask = following_frames(binary_warped, right_fit)

    # Saving polynomial fits for left and right lanes
    previous_left_fit = left_fit
    previous_right_fit = right_fit

    # Choose if diagnostic screen is output or not
    if diagnostic_screen == True:
        final = diagnostic([result, non_persp_warp, out_img, persp])
        return final
    else:
        return result


if __name__ == '__main__':
    # Initial Setup
    video = 0
    diagnostic_screen = True
    frame = 0
    previous_left_fit = np.array([0, 0, 0])
    previous_right_fit = np.array([0, 0, 0])

    # Calibrates camera and get matrix
    mtx, dist = get_calibration()

    # Load In test images
    test_imgs = glob.glob('test_images/test*.jpg')
    straight_imgs = glob.glob('test_images/straight_lines*.jpg')
    all_imgs = straight_imgs + test_imgs
    image = mpimg.imread(all_imgs[0])

    # get video frame for debugging
    #

    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    # If pipeline to be run on image and not video
    if video == 0:
        # Perspective transform source and destination points
        image_x = image.shape[1]
        image_y = image.shape[0]

        src_bot_left = [int(image_x * 0.09), int(image_y * 0.93)]
        src_top_left = [int(image_x * 0.40), int(image_y * 0.66)]
        src_top_right = [int(image_x * 0.62), int(image_y * 0.66)]
        src_bot_right = [int(image_x * 0.99), int(image_y * 0.93)]

        src = np.float32([src_bot_left,
                          src_top_left,
                          src_top_right,
                          src_bot_right])

        dst_top_right = [int(image_x * 0.01), int(image_y * 0.99)]
        dst_bot_right = [int(image_x * 0.01), int(image_y * 0.01)]
        dst_bot_left = [int(image_x * 0.99), int(image_y * 0.01)]
        dst_top_left = [int(image_x * 0.99), int(image_y * 0.99)]

        dst = np.float32([dst_top_right,
                          dst_bot_right,
                          dst_bot_left,
                          dst_top_left])

        im = lanefinding_pipeline(image)
        plt.figure(figsize=(15, 10))
        plt.imshow(im)
        plt.axis('off')
        plt.show()

    # Pipeline run on project video
    if video == 1:
        # Perspective transform source and destination points
        image_x = image.shape[1]
        image_y = image.shape[0]

        src_bot_left = [int(image_x * 0.09), int(image_y * 0.93)]
        src_top_left = [int(image_x * 0.40), int(image_y * 0.66)]
        src_top_right = [int(image_x * 0.62), int(image_y * 0.66)]
        src_bot_right = [int(image_x * 0.99), int(image_y * 0.93)]

        src = np.float32([src_bot_left,
                          src_top_left,
                          src_top_right,
                          src_bot_right])

        # four destination points
        dst_top_right = [int(image_x * 0.01), int(image_y * 0.99)]
        dst_bot_right = [int(image_x * 0.01), int(image_y * 0.01)]
        dst_bot_left = [int(image_x * 0.99), int(image_y * 0.01)]
        dst_top_left = [int(image_x * 0.99), int(image_y * 0.99)]

        dst = np.float32([dst_top_right,
                          dst_bot_right,
                          dst_bot_left,
                          dst_top_left])

        if diagnostic_screen == True:
            diag_string = '_diagnostic'
        else:
            diag_string = ''

        challenge_video = 'videos/video_output/project_output' + diag_string + '.mp4'
        challenge_clip = VideoFileClip('videos/video_input/project_video.mp4')  # .subclip(0, 1)
        project_clip = challenge_clip.fl_image(lanefinding_pipeline)  # NOTE: this function expects color images!!
        project_clip.write_videofile(challenge_video, audio=False)

    # Pipeline run on challenge video
    if video == 2:
        # Perspective transform source and destination points
        image_x = image.shape[1]
        image_y = image.shape[0]

        src_bot_left = [int(image_x * 0.15), int(image_y * 0.93)]
        src_top_left = [int(image_x * 0.46), int(image_y * 0.66)]
        src_top_right = [int(image_x * 0.58), int(image_y * 0.66)]
        src_bot_right = [int(image_x * 0.93), int(image_y * 0.93)]

        src = np.float32([src_bot_left,
                          src_top_left,
                          src_top_right,
                          src_bot_right])

        # four destination points
        dst_top_right = [int(image_x * 0.01), int(image_y * 0.99)]
        dst_bot_right = [int(image_x * 0.01), int(image_y * 0.01)]
        dst_bot_left = [int(image_x * 0.99), int(image_y * 0.01)]
        dst_top_left = [int(image_x * 0.99), int(image_y * 0.99)]

        dst = np.float32([dst_top_right,
                          dst_bot_right,
                          dst_bot_left,
                          dst_top_left])

        if diagnostic_screen == True:
            diag_string = '_diagnostic'
        else:
            diag_string = ''

        challenge_video = 'videos/video_output/challenge_output' + diag_string + '.mp4'
        challenge_clip = VideoFileClip('videos/video_input/challenge_video.mp4')  # .subclip(0, 5)
        project_clip = challenge_clip.fl_image(lanefinding_pipeline)  # NOTE: this function expects color images!!
        project_clip.write_videofile(challenge_video, audio=False)

    # Pipeline for harder challenge video
    if video == 3:
        # Perspective transform source and destination points
        image_x = image.shape[1]
        image_y = image.shape[0]

        src_bot_left = [int(image_x * 0.15), int(image_y * 0.93)]
        src_top_left = [int(image_x * 0.40), int(image_y * 0.66)]
        src_top_right = [int(image_x * 0.55), int(image_y * 0.66)]
        src_bot_right = [int(image_x * 0.93), int(image_y * 0.93)]

        src = np.float32([src_bot_left,
                          src_top_left,
                          src_top_right,
                          src_bot_right])

        # four destination points
        dst_top_right = [int(image_x * 0.01), int(image_y * 0.99)]
        dst_bot_right = [int(image_x * 0.01), int(image_y * 0.01)]
        dst_bot_left = [int(image_x * 0.99), int(image_y * 0.01)]
        dst_top_left = [int(image_x * 0.99), int(image_y * 0.99)]

        dst = np.float32([dst_top_right,
                          dst_bot_right,
                          dst_bot_left,
                          dst_top_left])

        if diagnostic_screen == True:
            diag_string = '_diagnostic'
        else:
            diag_string = ''

        challenge_video = 'videos/video_output/harder_challenge_output' + diag_string + '.mp4'
        challenge_clip = VideoFileClip('videos/video_input/harder_challenge_video.mp4').subclip(0, 5)
        project_clip = challenge_clip.fl_image(lanefinding_pipeline)  # NOTE: this function expects color images!!
        project_clip.write_videofile(challenge_video, audio=False)

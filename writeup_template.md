## Advanced Lane Finding - Pablo Vela


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[Undist_vs_Dist]: ./write_up_images/Distorted_vs_Undistorted.PNG "Undistorted"
[test_undist_vs_dist]: ./write_up_images/testimg_undist_vs_dist.PNG "Test Undistort"
[final_mask]: ./write_up_images/final_mask.PNG "Binary Example"
[color_mask]: ./write_up_images/Color_mask.PNG "Color Mask"
[sobel_mask]: ./write_up_images/sobel_mask.PNG "Sobel Mask"
[persp_transform]: ./write_up_images/perspective_transform.PNG "Warp Example"
[fit_lines]: ./write_up_images/fitlines.png "Fit Visual"
[curvature_eq]: ./write_up_images/curvature_eq.PNG "Fit Visual"
[plotted_results]: ./write_up_images/plotted_results.PNG "Output"
[video1]: ./video_output/new_project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 0. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in both the python file `pipeline.py` as well as `helper_function.py`.

In `pipeline.py` I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0,
such that the object points are the same for each calibration image.
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
  I applied this distortion correction to the test image using the `cv2.undistort()` and obtained this result:

![alt text][Undist_vs_Dist]

These two functions can both be found in the helper function `undistort(img, objpoints, imgpoints)`

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As show above I use the helper function `undistort(img, objpoints, imgpoints)` to get the distortion coefficients and camera matrix and undistort the image. An example of the distorted vs undistorted can be see below :

![alt text][test_undist_vs_dist]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img)`, which appears under `helper_function.py`.
  The `perspective_transform(img)` function takes as input an image `(img)`.  I chose the hardcode the source and destination points in the following manner:

```python
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
    dst_top_right = [int(calib_img_x*(1-margin)), 0]
    dst_bot_right = [int(calib_img_x*(1-margin))-125, calib_img_y]
    dst_bot_left = [int(calib_img_x*margin)+125, calib_img_y]
    dst_top_left = [int(calib_img_x*margin), 0]
    dst = np.float32([dst_top_right,
            dst_bot_right,
            dst_bot_left,
            dst_top_left])
```

This resulted in the following source and destination points:

|Location       | Source        | Destination   |
|:-------------:|:-------------:|:-------------:|
|Top Right      | 750, 450      | 1216, 0       |
|Bottom Right   | 1200, 680     | 1066, 720     |
|Bottom Left    | 100, 680      | 189, 720      |
|Top Left       | 550, 450      | 64, 0         |

A picture comparing the warped image and the original image is shown below where we can see the lines end in a parallel position confirming a correct transform

![alt text][persp_transform]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image
(thresholding steps found under `helper_function.py` in  function `mask(img)` and `final_mask(img)`. I found that
using the RGB colorspace for the white lane line, HSV for the yellow lane line, and HLS for the sobel operator worked best.

Here's an example of my output for my color masking

![alt text][color_mask]

Here's an example of my output for my sobel masking only in the x directions

![alt text][sobel_mask]

and lastly of the two masks together

![alt text][final_mask]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Under `pipeline.py` you can find the function `sliding_window(binary_warped)`. In this function I identified lane-line pixels as follows
* Take a histogram of the bottom half of the image
* Select a midpoint to search for pixels using a sliding window search
* Choose the number of windows that will be stacked on top of each other and their height
* Identify all nonzero pixels in the x and y position of the image
* Choose the width of the windows and minimum pixels that need to be found to recenter the window
* Step through each window one by one and append found pixels to left lane and right lane lists

Once the corresponding left and right lane pixels are found then fit my lane lines with a 2nd order polynomial kinda like this:


![alt text][fit_lines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature I used the equation

![alt text][curvature_eq]

Where A is first term of the second order polynomial fit to the lane line, B is the second term. I did this  in lines 117 through 131 in my code in `pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 137 through 172 in my code in `pipeline.py` in the function `draw_lane()`.  Here is an example of my result on a test image:

![alt text][plotted_results]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./video_output/new_project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the major issues that I found in my implementation of the project is the sensitivity to color change that can be seen when my pipeline is applied to the challenge video and harder challenge video. The lane will jump around like crazy because some lanes
  switch colors or are half black half white. A way that this could be fixed is by having my algorithm take a running average and if there is a huge jump between where the next frames predicted lane line and the previous line is then I would just use the
  average to smooth out difficult frames.

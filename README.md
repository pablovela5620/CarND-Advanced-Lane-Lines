## Advanced Lane Finding - Pablo Vela


---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Use color transforms, gradients, etc., to create a thresholded binary image
* Detect lane pixels and fit to find the lane boundary
* Determine the curvature of the lane and vehicle position with respect to the center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

[//]: # (Image References)

[Undist_vs_Dist]: ./write_up_images/Distorted_vs_Undistorted.PNG "Undistorted"
[test_undist_vs_dist]: ./write_up_images/testimg_undist_vs_dist.PNG "Test Undistort"
[final_mask]: ./write_up_images/final_mask.PNG "Binary Example"
[color_mask]: ./write_up_images/Color_mask.PNG "Color Mask"
[sobel_mask]: ./write_up_images/sobel_mask.PNG "Sobel Mask"
[persp_transform]: ./write_up_images/perspective_transform.PNG "Warp Example"
[fit_lines]: ./write_up_images/fitlines.PNG "Fit Visual"
[curvature_eq]: ./write_up_images/curvature_eq.PNG "Fit Visual"
[plotted_results]: ./write_up_images/plotted_results.PNG "Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation:


### Writeup / README

#### 0. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in both the python file `pipeline.py` as well as `helper_function.py`.

In `helper_function.py` I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0,
such that the object points are the same for each calibration image.
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients (mtx,dist) using the `cv2.calibrateCamera()` function. Once these are found, mtx and dist
are saved in a pickle file to significantly increase speed of image distortion correction in images.
  Finally the distortion correction is applied to the image using `undistort_image(img, mtx, dist)` and obtained this result:

![alt text][Undist_vs_Dist]

These functions can both be found in `helper_function.py`

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

As show above `undistort(img, mtx, dist)` is used to correct any distortion in the image. An example of the distorted vs undistorted can be see below :

![alt text][test_undist_vs_dist]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img,src,dst)`, which appears under `helper_function.py`.
  The `perspective_transform(img)` function takes as input an image `img`, the source points you want to move `src`, and their destination points `dst`.  I chose the hardcode the source and destination points in the following manner:

```python
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

dst_top_right = [int(image_x * 0.01), int(image_y * 0.99)]
dst_bot_right = [int(image_x * 0.01), int(image_y * 0.01)]
dst_bot_left = [int(image_x * 0.99), int(image_y * 0.01)]
dst_top_left = [int(image_x * 0.99), int(image_y * 0.99)]

dst = np.float32([dst_top_right,
                  dst_bot_right,
                  dst_bot_left,
                  dst_top_left])
```

This resulted in the following source and destination points:

|Location       | Source        | Destination   |
|:-------------:|:-------------:|:-------------:|
|Bottom Left    | 192, 669      | 12, 712       |
|Top Left       | 512, 475      | 12, 7         |
|Top Right      | 704, 907      | 1267, 7       |
|Bottom Right   | 1190, 669     | 1267, 712     |

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

For this portion I was insipred by Vivek Yabdav's approach of using a computer vision  for the first frame of the video, and then falling back on using a polynomial fit based on the previously found mask to guess what the next frame would look like.
This approach allows for much faster processing of the video. In my case I was getting between 7 frames per second and 8 frames per second.

Under `helper_function.py` you can find the function `first_frame_detection(binary_warped)`. In this function I identified lane-line pixels as follows
* Take a histogram of the bottom half of the image
* Use a moving average to smooth any peaks found in the histogram
* Use scikit learns `find_peaks_cwt()` to determine how many peaks are in the image and their index
    * Depending on the number of peaks previously found peaks are used to help with bad frames.
    * Use this initial peaks to determine where next peaks will appear using a sliding window method
* Split the image into 8 windows
* Again using `find_peaks_cwt()` in the predeterimined sized window find the peaks within the window
* If a peak moves more then 60 pixels, reject as an outlier and use previously found peak
* Step through each window one by one until all peaks in each window has been found. Producing a mask of windows where the peaks were found
* After the first mask in the first frame of the video is found a second order polynomial is fit to the masks.
* This polynomial is saved to later be used in the upcoming frames to determine the coming masks by fitting new windows to the polynomial fit


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate the radius of curvature I used the equation

![alt text][curvature_eq]

Where A is first term of the second order polynomial fit to the lane line, B is the second term. This was done in the function `poly_curvature(fit_poly, y_point)`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `helper_function.py` in the function `draw_lane()`.  Here is an example of my result on a test image when running diagnostic:

![alt text][plotted_results]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Project Video

[![Project Video](http://img.youtube.com/vi/tDdDifBlFRw/0.jpg)](https://www.youtube.com/watch?v=tDdDifBlFRw)

Challenge Video

[![Project Video](http://img.youtube.com/vi/gTkqwcGMpfk/0.jpg)](https://www.youtube.com/watch?v=gTkqwcGMpfk)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the major issues that I found in my implementation of the project is the lack of accuracy if the polyfit deviates too much from the mask. This results in the lane being slightly too wide or too narrow as can be seen in the challenge
video.


### References
----------
- [Vivek Yadav - More robust lane finding using advanced computer vision techniques](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa)

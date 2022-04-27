from parameters import *
from classes import *


def read_img(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def show_image(image, title='Image', cmap_type='gray', bgr2rgb=False):
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.show()


def show_images(images, cols=2, cmap=None, bgr2rgb=False):

    rows = (len(images)+1)//cols

    plt.figure(figsize=(20, 22))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        try:
            cmap = 'gray' if len(image.shape) == 2 else cmap
        except AttributeError:
            cmap = None

        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def warper(img):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped


def unwarp(img):

    # Compute and apply inverse perpective transform
    img_size = (img.shape[1], img.shape[0])
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(
        img, Minv, img_size, flags=cv2.INTER_NEAREST)

    return unwarped


def calc_sobel(img, sx=False, sy=False, sobel_kernel=5, thresh=(25, 200)):

    # Convert to grayscale - sobel can only have one color channel
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the sobel gradient in x and y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    if sx:
        abs_sobel = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    elif sy:
        abs_sobel = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    else:
        # Calculate the magnitude
        mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))

    # Create a binary mask where mag thresholds are me
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def run_canny(img, kernel_size=5, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply a Gaussian Blur
    gausImage = gaussian_blur(gray, kernel_size)

    # Run the canny edge detection
    cannyImage = canny(gausImage, low_thresh, high_thresh)

    return cannyImage


def applyThreshold(channel, thresh):
    # Create an image of all zeros
    binary_output = np.zeros_like(channel)

    # Apply a threshold to the channel with inclusive thresholds
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output


def rgb_rthresh(img, thresh=(125, 255)):
    # Pull out the R channel - assuming that RGB was passed in
    channel = img[:, :, 0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)


def hls_sthresh(img, thresh=(125, 255)):
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Pull out the S channel
    channel = hls[:, :, 2]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)


def lab_bthresh(img, thresh=(125, 255)):
    # Convert to HLS
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Pull out the B channel
    channel = lab[:, :, 2]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)


def luv_lthresh(img, thresh=(125, 255)):
    # Convert to HLS
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # Pull out the L channel
    channel = luv[:, :, 0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)


def binaryPipeline(img, show_images=False,
                   sobel_kernel_size=7, sobel_thresh_low=35, sobel_thresh_high=50,
                   canny_kernel_size=5, canny_thresh_low=50, canny_thresh_high=150,
                   r_thresh_low=225, r_thresh_high=255,
                   s_thresh_low=220, s_thresh_high=250,
                   b_thresh_low=175, b_thresh_high=255,
                   l_thresh_low=215, l_thresh_high=255
                   ):

    warped = warper(img)

    # COLOR SELECTION
    r = rgb_rthresh(warped, thresh=(r_thresh_low, r_thresh_high))
    s = hls_sthresh(warped, thresh=(s_thresh_low, s_thresh_high))
    b = lab_bthresh(warped, thresh=(b_thresh_low, b_thresh_high))
    l = luv_lthresh(warped, thresh=(l_thresh_low, l_thresh_high))

    # EDGE DETECTION
    sobel = calc_sobel(warped, sx=True, sobel_kernel=sobel_kernel_size, thresh=(
        sobel_thresh_low, sobel_thresh_high))

    # Run canny edge detector
    # canny_ = run_canny(warped, kernel_size=canny_kernel_size, low_thresh=canny_thresh_low, high_thresh=canny_thresh_high)

    # Create plots if we want them
    if show_images:
        # f, (ax1, ax2, ax3, ax4, ax5 , ax6) = plt.subplots(1, 6, figsize=(16, 7))
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(16, 7))
        f.tight_layout()

        ax1.set_title('r', fontsize=10)
        ax1.axis('off')
        ax1.imshow(r, cmap='gray')

        ax2.set_title('s', fontsize=15)
        ax2.axis('off')
        ax2.imshow(s, cmap='gray')

        ax3.set_title('b', fontsize=15)
        ax3.axis('off')
        ax3.imshow(b, cmap='gray')

        ax4.set_title('l', fontsize=15)
        ax4.axis('off')
        ax4.imshow(l, cmap='gray')

        ax5.set_title('sobel', fontsize=15)
        ax5.axis('off')
        ax5.imshow(sobel, cmap='gray')

        # ax6.set_title('canny', fontsize=15)
        # ax6.axis('off')
        # ax6.imshow(canny_, cmap='gray')

    # combine these layers
    combined_binary = np.zeros_like(r)
    combined_binary[(r == 1) | (s == 1) | (b == 1)
                    | (l == 1) | (sobel == 1)] = 1

    return combined_binary




def calc_line_fits(img):

    # Settings
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    # plt.figure()
    # plt.plot(histogram)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
            nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

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

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, left_fit_m, right_fit_m, out_img


def draw_fits(img, left_fit, right_fit):
    # calculates left and right and ploty and generates an image from the three of them then plots it using plt.imshow()

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array(
        [np.transpose(np.vstack([left_fitx, ploty]))], dtype=int)
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=int)
    pts = np.hstack((pts_left, pts_right))

    cv2.polylines(color_warp, pts_right, False, (0, 0, 255), thickness=15)
    cv2.polylines(color_warp, pts_left, False, (255, 0, 0), thickness=15)

    return color_warp


def get_center_dist(leftLine, rightLine):

    # grab the x and y fits at px 700 (slightly above the bottom of the image)
    y = 700.
    image_center = 640. * xm_per_pix

    leftPos = leftLine.fit_px[0] * \
        (y**2) + leftLine.fit_px[1]*y + leftLine.fit_px[2]
    rightPos = rightLine.fit_px[0]*(y**2) + \
        rightLine.fit_px[1]*y + rightLine.fit_px[2]
    lane_middle = int((rightPos - leftPos)/2.)+leftPos
    lane_middle = lane_middle * xm_per_pix

    mag = lane_middle - image_center
    if (mag > 0):
        head = "Right"
    else:
        head = "Left"

    return head, mag


def combine_radii(leftLine, rightLine):

    left = leftLine.radius_of_curvature
    right = rightLine.radius_of_curvature

    return np.average([left, right])


def create_final_image(img, binary_warped, leftLine, rightLine, show_images=False):

    left_fit = leftLine.fit_px
    right_fit = rightLine.fit_px

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32(
        [pts_left]), isClosed=False, color=(255, 0, 255), thickness=20)
    cv2.polylines(color_warp, np.int32(
        [pts_right]), isClosed=False, color=(0, 255, 255), thickness=20)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)

    if show_images:
        plt.figure(figsize=(9, 9))
        plt.imshow(color_warp)

        plt.figure(figsize=(9, 9))
        plt.imshow(result)

    return result


def add_image_text(img, radius, head, center):

    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_DUPLEX

    text = 'Radius of curvature: ' + '{:04.0f}'.format(radius) + 'm'
    cv2.putText(img, text, (50, 100), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    text = '{:03.2f}'.format(abs(center)) + 'm ' + head + ' of center'
    cv2.putText(img, text, (50, 175), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    return img


def write_on_img(img, text, pos):
    # writes the text on the top center of the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, pos, font, 1.5, (255, 0, 0), 2, cv2.LINE_AA)


def final_pipeline(img, leftLine=Line(), rightLine=Line()):

    binary_warped = binaryPipeline(img)

    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(
        binary_warped)

    leftLine.add_new_fit(left_fit, left_fit_m)
    rightLine.add_new_fit(right_fit, left_fit_m)

    # get radius and center distance
    curve_rad = combine_radii(leftLine, rightLine)
    head, center = get_center_dist(leftLine, rightLine)

    # create the final image
    result = create_final_image(img, binary_warped, leftLine, rightLine)

    # add the text to the image
    result = add_image_text(result, curve_rad, head, center)

    return result


def final_debug_pipline(img, leftLine=Line(), rightLine=Line()):

    binary_warped = binaryPipeline(img)

    left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(
        binary_warped)

    leftLine.add_new_fit(left_fit, left_fit_m)
    rightLine.add_new_fit(right_fit, left_fit_m)

    # get radius and center distance
    curve_rad = combine_radii(leftLine, rightLine)
    head, center = get_center_dist(leftLine, rightLine)

    # create the final image
    result = create_final_image(img, binary_warped, leftLine, rightLine)

    binary_warped_3D = np.dstack(
        (binary_warped, binary_warped, binary_warped))*255
    lines_only = draw_fits(binary_warped, left_fit, right_fit)

    # img >>>>>>>> warper(img) >>>> binary_warped_3D >>>> lines_only  >>>> out_img >>> result

    # perspective transform
    texts = ['1-original image', '2-warped image', '3-binary ',
             '4-sliding windows', '5-detected lines only', '6-final image']

    row_1 = np.hstack((img, warper(img), binary_warped_3D))
    row_2 = np.hstack((result, lines_only, out_img))
    debug_out = np.vstack((row_1, row_2))

    # all images are of shape (720,1280)
    #                                   (x  ,  y)
    write_on_img(debug_out, texts[0], (500, 50))
    write_on_img(debug_out, texts[1], (1700, 50))
    write_on_img(debug_out, texts[2], (3000, 50))
    write_on_img(debug_out, texts[3], (3000, 800))
    write_on_img(debug_out, texts[4], (1700, 800))
    write_on_img(debug_out, texts[5], (500, 800))

    return debug_out


def create_output(input_path, debug=False):

    if input_path.endswith('.mp4'):
        clip1 = VideoFileClip(input_path)
        if debug:
            clip = clip1.fl_image(final_debug_pipline)
            clip.write_videofile('debug_video.mp4', audio=False)
            print(f'output saved to >> debug_video.mp4')
        else:
            clip = clip1.fl_image(final_pipeline)
            clip.write_videofile('output_video.mp4', audio=False)
            print(f'output saved to >> output_video.mp4')

    elif input_path.endswith('.jpg'):
        img = cv2.imread(input_path)
        if debug:
            result = final_debug_pipline(img)
            print(f'output saved to >> debug_img.jpg')
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('debug_img.jpg', result)
        else:
            result = final_pipeline(img)
            print(f'output saved to >> output_img.jpg')
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite('output_img.jpg', result )


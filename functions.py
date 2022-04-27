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

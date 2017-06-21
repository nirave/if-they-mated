""" Assignment 6 - Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available on T-square under references:

(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983

(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983

Notes
-----
    You may not use cv2.pyrUp or cv2.pyrDown anywhere in this assignment.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.

    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter.

    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel of parameter of 0.4
    and then reduce its width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT) and only keep the valid
    region (i.e., do NOT keep any pixels from the padded region) for the
    convolution. Subsampling must include the first row and column,
    skip the second, etc.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                          aabcdd
        abcd     Pad      aabcdd   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   eefghh   -------->  VUTS   -------->   RP
        ijkl    BORDER    iijkll     keep     RQPO               JH
        mnop   REFLECT    mmnopp     valid    NMLK
        qrst              qqrstt              JIHG
                          qqrstt

    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """

    # WRITE YOUR CODE HERE.
    image = image.astype(dtype=np.float64)
    output = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

    return output[::2, ::2].astype(dtype=np.float64)


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel of a=0.4.

    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after). For
    grading purposes, it is important that you use a reflected border (i.e.,
    padding equivalent to cv2.BORDER_REFLECT) and only keep the valid region
    (i.e., do NOT keep any pixels from the padded region) for the convolution.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                                          AA0B00
             Upsample   A0B0     Pad      AA0B00   Convolve   zyxw
        AB   ------->   0000   ------->   000000   ------->   vuts
        CD              C0D0    BORDER    CC0D00     keep     rqpo
        EF              0000   REFLECT    000000    valid     nmlk
                        E0F0              EE0F00              jihg
                        0000              000000              fedc
                                          000000

                NOTE: Remember to multiply the output by 4.

    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """

    # WRITE YOUR CODE HERE.
    expanded_image = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    expanded_image[::2, ::2] = image

    #kernel = generatingKernel(0.4)
    expanded_image = cv2.filter2D(expanded_image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    output = expanded_image * 4

    return output.astype(dtype=np.float64)


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels passed in by the input.

    You must use your reduce_layer() function to generate the pyramid.

    Parameters
    ----------
    image : numpy.ndarray(dtype=np.float)
        An image of dimension (r, c). The input is expected to contain floating
        point values.

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """

    # WRITE YOUR CODE HERE.
    output = [image.astype(dtype=np.float64)]

    last_image = image

    for i in range(0, levels):
        last_image = reduce_layer(last_image)
        output.append(last_image.astype(dtype=np.float64))

    return output

def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    You must use your expand_layer() function to generate the pyramid. The
    Gaussian Pyramid that is passed in is the output of your gaussPyramid
    function.

    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.

    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.

        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """

    # WRITE YOUR CODE HERE.
    output = []
    for i in range(0, len(gaussPyr) - 1):
        expanded_image = expand_layer(gaussPyr[i + 1])
        if gaussPyr[i].shape[0] % 2:
            expanded_image = expanded_image[0:-1, :]
        if gaussPyr[i].shape[1] % 2:
            expanded_image = expanded_image[:, 0:-1]
        output.append(gaussPyr[i] - expanded_image)

    output.append(gaussPyr[len(gaussPyr) - 1])

    return output

def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)

    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.

    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.

    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids

    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """

    # WRITE YOUR CODE HERE.
    blended_pyr = []

    for i in range(0, len(laplPyrBlack)):
        blended_pyr.append(gaussPyrMask[i] * laplPyrWhite[i] + (1 - gaussPyrMask[i]) * laplPyrBlack[i])

    return blended_pyr

def collapse(pyramid):
    """Collapse an input pyramid.

    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.

        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """

    # WRITE YOUR CODE HERE.
    output = None
    for i in range(len(pyramid) - 1, -1, -1):
        if output is None:
            output = pyramid[i]
        else:
            output = expand_layer(output)
            if output.shape[0] > pyramid[i].shape[0]:
                output = output[0:-1, :]
            if output.shape[1] > pyramid[i].shape[1]:
                output = output[:, 0:-1]

            output = pyramid[i] + output

    return output


def blender(black_img, white_img, mask_img):
    black_red = black_img[:, :, 0]
    black_blue = black_img[:, :, 1]
    black_green = black_img[:, :, 2]

    #mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY).astype(dtype=np.float64) / 255
    mask = mask_img

    white_red = white_img[:, :, 0]
    white_blue = white_img[:, :, 1]
    white_green = white_img[:, :, 2]

    black_gauss_red = gaussPyramid(black_red, 3)
    white_gauss_red = gaussPyramid(white_red, 3)
    mask_gauss = gaussPyramid(mask, 3)

    black_lap_red = laplPyramid(black_gauss_red)
    white_lap_red = laplPyramid(white_gauss_red)
    mask_lap = laplPyramid(mask_gauss)

    black_gauss_blue = gaussPyramid(black_blue, 3)
    white_gauss_blue = gaussPyramid(white_blue, 3)
    black_lap_blue = laplPyramid(black_gauss_blue)
    white_lap_blue = laplPyramid(white_gauss_blue)

    black_gauss_green = gaussPyramid(black_green, 3)
    white_gauss_green = gaussPyramid(white_green, 3)
    black_lap_green = laplPyramid(black_gauss_green)
    white_lap_green = laplPyramid(white_gauss_green)

    blended_red = blend(white_lap_red, black_lap_red, mask_gauss)
    blended_red = collapse(blended_red)
    blended_blue = blend(white_lap_blue, black_lap_blue, mask_gauss)
    blended_blue = collapse(blended_blue)
    blended_green = blend(white_lap_green, black_lap_green, mask_gauss)
    blended_green = collapse(blended_green)

    blended = np.copy(black_img)
    blended[:, :, 0] = blended_red
    blended[:, :, 1] = blended_blue
    blended[:, :, 2] = blended_green

    return blended

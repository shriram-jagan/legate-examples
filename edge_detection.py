import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve
import matplotlib

matplotlib.pyplot.switch_backend('Agg')

# RGB colors can vary between 0 and 255
pixel_min = 0.0
pixel_max = 255.0

def convolve_nd(array: ndarray, kernel: ndarray, mode: str = "same"):
    """
    array: ndarray
        Input array corresponding to a grayscale image
    kernel: ndarray
        Kernel to compute the gradient in x or y as per Sobel Edge Detector
    mode: str
        The default convolution mode. Note that cuNumeric only
        supports the convolution mode "same".

    Notes:
        Check https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
        for more information on Sobel Edge Detector

        The image was taken from:
        https://docs.nvidia.com/vpi/algo_canny_edge_detector.html
    """
    if np.__name__ == "cunumeric":
        return np.convolve(array, kernel, mode)
    return convolve(array, kernel, mode)


# Read the image and convert to grayscale
image = np.array(Image.open("image.png").convert("L"))

kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Apply the Sobel kernels
grad_x = convolve_nd(image, kernel_x, mode="same")
grad_y = convolve_nd(image, kernel_y, mode="same")

# Edges are computed from normalized absolute gradients
edges = np.sqrt(grad_x**2 + grad_y**2)
edges *= pixel_max / np.max(edges)
edges = edges.astype(int)


plt.subplot(121)
plt.imshow(image)
plt.title("Row of houses")
plt.xticks([]), plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap="gray")
plt.title("Edges on a row of houses")
plt.xticks([]), plt.yticks([])
plt.savefig("edges_houses.png")

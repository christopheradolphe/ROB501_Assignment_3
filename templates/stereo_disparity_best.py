import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """

    """
    Description of Algorithm Implemented:
    In my algorithm, I employed pre-processing using a Laplacian of Gaussian (LoG) 
    filter and post-processing using a median filter to improve upon the results 
    observed in part 1). It was observed that the algorithm from part 1) observed 
    greatest error around the edges in the image. To address this issue, a Laplacian 
    of Gaussian filter was applied to both the left and right stereo images. The 
    Gaussian component of the LoG filter reduces noise by smoothing variations in 
    image intensity, which helps to clarify edges, while the Laplace filter 
    identifies areas of intensity change, and thus better defines the edges in the 
    image. After applying this LoG filter it was observed that the disparity mapping 
    around the edges of objects in the image was improved. When printing out the final 
    disparity mapping from part 1), it was observed that the image appeared to have 
    holes with inconsistent colouring, which was most likely the result of noise in 
    the image. To mitigate this, a median filter was used to post process the disparity 
    mapping obtained from the SAD algorithm from part 1) to stabilize disparity values 
    in uniform regions. The standard deviation for the Laplacian of Gaussian filter and 
    window size for the median filter were tuned for optimal performance, and found 
    to be 1 and 7, respectively. 
    """
    
    # Define window size
    # Large window more robust to noise but may smooth out details
    window_size = 9 # 9x9 windows (found to be best from trial error)

    # Initialize Disparity Image 
    Id = np.zeros((Il.shape), dtype=np.uint8)

    # Apply LoG filter with a sigma of 1 to preprocess image
    sigma = 1
    Il_lg = gaussian_laplace(Il, sigma)
    Ir_lg = gaussian_laplace(Ir, sigma)

    # Pad boarders of image
    half_window = window_size // 2
    Il_padded = np.pad(Il_lg, half_window, mode='edge')
    Ir_padded = np.pad(Ir_lg, half_window, mode='edge')

    # Find bounding box corner in left image from values from bbox
    x_min, x_max = bbox[0]
    y_min, y_max = bbox[1]

    # Find disparity values for each pixel in bounding box
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
                # Initialize values to track minimum SAD and corresponding disparity
                y_padded = y + half_window
                x_padded = x + half_window

                # Variables to optimize
                min_sad = np.inf
                best_disparity = 0

                # Left image window centred at (x,y)
                left_image_window = Il_padded[y_padded - half_window:y_padded+half_window, x_padded-half_window:x_padded+half_window]

                # Loop over max_disparity for search
                for disparity in range(maxd+1):
                        # Ensure value is within bounds
                        if (x_padded - half_window - disparity) < 0:
                                continue

                        # Get the right image window
                        right_image_window = Ir_padded[y_padded - half_window:y_padded+half_window, x_padded-half_window-disparity:x_padded+half_window-disparity]


                        # Compute SAD similarity measure
                        sad = np.sum(np.abs(left_image_window - right_image_window))

                        if sad < min_sad:
                                min_sad = sad
                                best_disparity = disparity
                
                Id[y,x] = best_disparity
                

    Id = median_filter(Id, size=7)

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
import numpy as np
from scipy.ndimage.filters import *

def compute_zncc(left_patch, right_patch):
    left_mean = np.mean(left_patch)
    right_mean = np.mean(right_patch)
    left_std = np.std(left_patch)
    right_std = np.std(right_patch)
    denominator = left_std * right_std
    if denominator == 0:
        return 0  # Avoid division by zero
    numerator = np.sum((left_patch - left_mean) * (right_patch - right_mean))
    zncc_score = numerator / (denominator * left_patch.size)
    return zncc_score

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
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    # Define window size
    # Large window more robust to noise but may smooth out details
    window_size = 9 # 9x9 windows (found to be best from trial error)

    # Initialize Disparity Image 
    Id = np.zeros((Il.shape), dtype=np.uint8)

    # Pad boarders of image
    half_window = window_size // 2
    Il_padded = np.pad(Il, half_window, mode='edge')
    Ir_padded = np.pad(Ir, half_window, mode='edge')

    # Find bounding box corner in left image from values from bbox
    x_min, x_max = bbox[0]
    y_min, y_max = bbox[1]

#     # Store shifted right image in dictionary
#     shifted_image = {}
#     for disparity in range(maxd + 1):
#         shifted_image[disparity] = np.roll(Ir, -d, axis=1)
#         shifted_image[d][:, -d:] = 0  # Set wrapped-around values to 0

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
                

    # Note: Higher disparity means it is closer to image -> Darker

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

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

    # Notes:
    # basic, fixed window size matching routine
    # use sum of absolute difference similarity measure
    # correct matches is winner take all (lowest difference)

    """
    Steps:
    1. Define window size
    2. Initialize empty numpy array with shape of Il
    3. Pad the image borders of image
    4. Loop through each value in bounding box (row and columns)
    """

    # Define window size
    # Large window more robust to noise but may smooth out details
    window_size = 5 # 5x5 windows

    # Initialize Disparity Image 
    Id = np.zeros((Il.shape))

    # Pad boarders of image
    padding_size = window_size // 2
    Il_padded = np.pad(Il, padding_size, mode='edge')
    Ir_padded = np.pad(Ir, padding_size, mode='edge')

    # Find bounding box corner in left image from values from bbox
    x_min, x_max = bbox[0]
    y_min, y_max = bbox[1]

    # Find disparity values for each pixel in bounding box
    for y in range(y_min + padding_size, y_max + padding_size + 1):
        for x in range(x_min + padding_size, x_max + padding_size + 1):
                # Initialize values to track minimum SAD and corresponding disparity
                min_sad = np.inf
                best_disparity = 0

                # Left image window
                left_image_window = Il_padded[y:y+window_size, x: x+window_size]

                # Loop over max_disparity for search
                for disparity in range(maxd+1):
                        # Ensure value is within bounds
                        if (x - disparity) < 0:
                                continue

                        # Get the right image window
                        right_image_window = Ir_padded[y : y + window_size, (x - disparity) : (x - disparity) + window_size]


                        # Compute SAD similarity measure
                        sad = np.sum(np.abs(left_image_window - right_image_window))

                        if sad < min_sad:
                                min_sad = sad
                                best_disparity = disparity
                
                Id[y - padding_size,x - padding_size] = best_disparity
        print(y)
                

    # Note: Higher disparity means it is closer to image -> Darker

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
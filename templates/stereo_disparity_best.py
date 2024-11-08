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

    # Convert images to float
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    
    # Parameters
    window_size = 9
    half_window = window_size // 2
    P1 = 0.8  # Small disparity change penalty
    P2 = 8    # Large disparity change penalty
    
    # Initialize disparity map and cost volume
    Id = np.zeros(Il.shape, dtype=np.uint8)
    height, width = Il.shape
    cost_volume = np.zeros((height, width, maxd + 1), dtype=np.float32)
    
    # Compute ZNCC scores for each disparity
    for d in range(maxd + 1):
        # Shift right image
        shifted_Ir = np.roll(Ir, -d, axis=1)
        shifted_Ir[:, -d:] = 0  # Handle border pixels
        # Compute ZNCC scores
        for y in range(half_window, height - half_window):
            for x in range(half_window + d, width - half_window):
                left_patch = Il[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                right_patch = shifted_Ir[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                zncc_score = compute_zncc(left_patch, right_patch)
                cost_volume[y, x, d] = -zncc_score  # Negative for cost minimization
    
    # Apply smoothness constraint
    aggregated_cost = cost_volume.copy()
    for y in range(half_window, height - half_window):
        for x in range(half_window + maxd, width - half_window):
            for d in range(maxd + 1):
                if x == half_window + maxd:
                    continue  # Skip first column
                else:
                    min_prev_cost = aggregated_cost[y, x - 1, d]
                    min_prev_cost = min(
                        aggregated_cost[y, x - 1, d],
                        aggregated_cost[y, x - 1, d - 1] + P1 if d > 0 else np.inf,
                        aggregated_cost[y, x - 1, d + 1] + P1 if d < maxd else np.inf,
                        np.min(aggregated_cost[y, x - 1, :]) + P2
                    )
                    aggregated_cost[y, x, d] += min_prev_cost
    
    # Select disparities with minimum aggregated cost
    Id = np.argmin(aggregated_cost, axis=2)
    
    # Ensure correct output type and shape
    Id = Id.astype(np.uint8)

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
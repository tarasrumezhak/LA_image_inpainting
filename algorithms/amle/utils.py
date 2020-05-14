from matplotlib import image as mpimage
import numpy as np
import scipy

def create_image_and_mask(imagefilename, maskfilename):
    input_matrix = mpimage.imread(imagefilename)

    if input_matrix.ndim == 3:
        M, N, C = input_matrix.shape
    else:
        M, N = input_matrix.shape
        C = 1

    # import the mask of the inpainting domain
    # mask = 1 intact part
    # mask = 0 missing domain
    mask = np.float64((mpimage.imread(maskfilename) == 1))

    if (input_matrix.ndim == 3) and (mask.ndim < 3):
        mask = np.repeat(mask[:, :, np.newaxis], C, axis=2)

    if C == 1:
        input_matrix = np.expand_dims(input_matrix, axis=2)
        mask = np.expand_dims(mask, axis=2)

    # create the image with the missin domain:
    noise = scipy.rand(M, N, C)
    u = mask * input_matrix + (1 - mask) * noise

    return (u, mask)

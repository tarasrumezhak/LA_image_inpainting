import matplotlib.image as mpimg
import cv2 as cv
import scipy
import numpy as np


def harmonic_inpainting(damaged_file, mask_file, output_file, fidelity, tol, maxiter, dt): #damaged_file,mask_file,output_file,lambda,tol,maxiter,dt
    input = mpimg.imread(damaged_file)

    if input.ndim == 3:
        M, N, C = input.shape
    else:
        M, N = input.shape
        C = 1

    mask = scipy.float64((mpimg.imread(mask_file) == 1))
    if (input.ndim == 3) & (mask.ndim < 3):
        mask = np.repeat(mask[:, :, np.newaxis], C, axis=2)

    if C == 1:
        input = scipy.expand_dims(input, axis=2)
        mask = scipy.expand_dims(mask, axis=2)

    u = mask * input

    # u = input.copy()

    for c in range(0, C):

        for iter in range(0, maxiter):

            # COMPUTE NEW SOLUTION
            laplacian = cv.Laplacian(u[:, :, c], cv.CV_64F)
            unew = u[:, :, c] + dt * (laplacian + fidelity * mask[:, :, c] * (input[:, :, c] - u[:, :, c]))

            # exit condition
            diff_u = np.linalg.norm(unew.reshape(M * N, 1) - u[:, :, c].reshape(M * N, 1), 2) / np.linalg.norm(
                unew.reshape(M * N, 1), 2)

            # update
            u[:, :, c] = unew

            # test exit condition
            if diff_u < tol:
                break

    mpimg.imsave(output_file, u)

    return u

if __name__ == '__main__':
    fidelity = 10
    tol = 1e-5
    maxiter = 500
    dt = 0.1
    harmonic_inpainting('malone.jpg', 'mask_malone.png', 'result.png', fidelity, tol, maxiter, dt)

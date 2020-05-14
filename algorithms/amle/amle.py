import sys
import cv2 as cv
import matplotlib.image as mpimage
import numpy as np
from PIL import Image
from utils import create_image_and_mask
import scipy


def create_kernel_derivatives():

    # Kernels
    k_a_i = np.zeros((3, 3))
    k_a_i[1, 1] = -1
    k_a_i[2, 1] = 1
    k_a_j = np.zeros((3, 3))
    k_a_j[1, 1] = -1
    k_a_j[1, 2] = 1

    # backwards i,j
    k_b_i = np.zeros((3, 3))
    k_b_i[1, 1] = 1
    k_b_i[0, 1] = -1
    k_b_j = np.zeros((3, 3))
    k_b_j[1, 1] = 1
    k_b_j[1, 0] = -1

    # centred i,j
    k_a_i = np.zeros((3, 3))
    k_b_i[0, 1] = -0.5
    k_b_i[2, 1] = 0.5
    k_c_j = np.zeros((3, 3))
    k_c_j[1, 2] = 0.5
    k_c_j[1, 0] = -0.5

    return k_a_i, k_a_j, k_b_i, k_b_j, k_a_i, k_c_j


def amle(input_matrix, mask_matrix, fidelity, tol, maxiter, dt):

    if input_matrix.ndim == 3:
        M, N, C = input_matrix.shape
    else:
        M, N = input_matrix.shape
        C = 1

    Kfi, Kfj, Kbi, Kbj, Kci, Kcj = create_kernel_derivatives()

    u = input_matrix.copy()
    v = np.zeros((M, N, 2))

    for c in range(0, C):

        for iter in range(0, maxiter):

            ux = cv.filter2D(u[:, :, c], -1, Kfi)
            uy = cv.filter2D(u[:, :, c], -1, Kfj)

            uxx = cv.filter2D(ux, -1, Kbi)
            uxy = cv.filter2D(ux, -1, Kbj)
            uyx = cv.filter2D(uy, -1, Kbi)
            uyy = cv.filter2D(uy, -1, Kbj)

            v[:, :, 0] = cv.filter2D(u[:, :, c], -1, Kci)
            v[:, :, 1] = cv.filter2D(u[:, :, c], -1, Kcj)

            dennormal = scipy.sqrt(scipy.sum(v ** 2, axis=2) + 1e-15)
            v[:, :, 0] = v[:, :, 0] / dennormal
            v[:, :, 1] = v[:, :, 1] / dennormal

            unew = u[:, :, c] + dt * (
                uxx * v[:, :, 0] ** 2
                + uyy * v[:, :, 1] ** 2
                + (uxy + uyx) * (v[:, :, 0] * v[:, :, 1])
                + fidelity * mask_matrix[:, :, c] * (input_matrix[:, :, c] - u[:, :, c])
            )

            diff_u = np.linalg.norm(
                unew.reshape(M * N, 1) - u[:, :, c].reshape(M * N, 1), 2
            ) / np.linalg.norm(unew.reshape(M * N, 1), 2)

            u[:, :, c] = unew

            if diff_u < tol:
                break

    if C == 1:
        mpimage.imsave("./amle_output.png", u[:, :, 0], cmap="gray")
    elif C == 3:
        mpimage.imsave("./amle_output.png", u)

    return u


def main(image_filename, mask_filename):
    fidelity = 10 ^ 2
    tol = 1e-8
    maxiter = 40000
    dt = 0.01
    cleanfilename = "./dataset/amle_clean.png"
    maskfilename = "./dataset/amle_mask.png"
    input_image, mask = create_image_and_mask(cleanfilename, maskfilename)
    mpimage.imsave("./dataset/amle_input.png", input_image[:, :, 0], cmap="gray")
    u = amle(input_image, mask, fidelity, tol, maxiter, dt)


if __name__ == "__main__":
    image_filename, mask_filename = sys.argv[1:]
    main(image_filename, mask_filename)

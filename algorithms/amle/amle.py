from __future__ import division
import cv2 as cv
import matplotlib.image as mpimage
import numpy as np
from utils import create_image_and_mask



def create_kernel_derivatives():

    # forward i,j
    kernel_forward_i = np.zeros((3, 3))
    kernel_forward_i[1, 1] = -1
    kernel_forward_i[2, 1] = 1
    kernel_forward_j = np.zeros((3, 3))
    kernel_forward_j[1, 1] = -1
    kernel_forward_j[1, 2] = 1

    # backward i,j
    kernel_backwrds_i = np.zeros((3, 3))
    kernel_backwrds_i[1, 1] = 1
    kernel_backwrds_i[0, 1] = -1
    kernel_backwards_j = np.zeros((3, 3))
    kernel_backwards_j[1, 1] = 1
    kernel_backwards_j[1, 0] = -1

    # centred i,j
    kernel_centred_i = np.zeros((3, 3))
    kernel_centred_i[0, 1] = -0.5
    kernel_centred_i[2, 1] = 0.5
    kernel_cetnred_j = np.zeros((3, 3))
    kernel_cetnred_j[1, 2] = 0.5
    kernel_cetnred_j[1, 0] = -0.5

    return [
        kernel_forward_i,
        kernel_forward_j,
        kernel_backwrds_i,
        kernel_backwards_j,
        kernel_centred_i,
        kernel_cetnred_j,
    ]


def amle_inpainting(input_matrix, mask_matrix, fidelity, tolerance, maxiter, dt):

    if input_matrix.ndim == 3:
        M, N, C = input_matrix.shape
    else:
        M, N = input_matrix.shape
        C = 1

    (
        kernel_forward_i,
        kernel_forward_j,
        kernel_backwards_i,
        kernel_backwards_j,
        kernel_centred_i,
        kernel_centred_j,
    ) = create_kernel_derivatives()

    u = input_matrix.copy()
    vec = np.zeros((M, N, 2))

    for c in range(0, C):
        for _ in range(0, maxiter):

            u_x = cv.filter2D(u[:, :, c], -1, kernel_forward_i)
            u_y = cv.filter2D(u[:, :, c], -1, kernel_forward_j)

            u_xx = cv.filter2D(u_x, -1, kernel_backwards_i)
            u_xy = cv.filter2D(u_x, -1, kernel_backwards_j)
            u_yx = cv.filter2D(u_y, -1, kernel_backwards_i)
            u_yy = cv.filter2D(u_y, -1, kernel_backwards_j)

            vec[:, :, 0] = cv.filter2D(u[:, :, c], -1, kernel_centred_i)
            vec[:, :, 1] = cv.filter2D(u[:, :, c], -1, kernel_centred_j)

            dennormal = np.lib.scimath.sqrt(np.sum(vec ** 2, axis=2) + 1e-15)
            vec[:, :, 0] = vec[:, :, 0] / dennormal
            vec[:, :, 1] = vec[:, :, 1] / dennormal

            u_new = u[:, :, c] + dt * (
                u_xx * vec[:, :, 0] ** 2
                + u_yy * vec[:, :, 1] ** 2
                + (u_xy + u_yx) * (vec[:, :, 0] * vec[:, :, 1])
                + fidelity * mask_matrix[:, :, c] * (input_matrix[:, :, c] - u[:, :, c])
            )

            diff_u = np.linalg.norm(
                u_new.reshape(M * N, 1) - u[:, :, c].reshape(M * N, 1), 2
            ) / np.linalg.norm(u_new.reshape(M * N, 1), 2)

            u[:, :, c] = u_new

            if diff_u < tolerance:
                break

    if C == 1:
        mpimage.imsave("./amle_output.png", u[:, :, 0], cmap="gray")
    elif C == 3:
        mpimage.imsave("./amle_output.png", u)

    return u


def main():
    fidelity = 10 ^ 2
    tolrance = 1e-8
    max_iterations = 100000
    d_t = 0.01
    cleanfilename = "./dataset/amle_clean.png"
    maskfilename = "./dataset/amle_mask.png"
    input_image, mask = create_image_and_mask(cleanfilename, maskfilename)
    mpimage.imsave("./dataset/amle_input.png", input_image[:, :, 0], cmap="gray")
    result = amle_inpainting(input_image, mask, fidelity, tolrance, max_iterations, d_t)


if __name__ == "__main__":
    main()

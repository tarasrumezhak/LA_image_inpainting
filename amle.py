import cv2 as cv
import matplotlib.image as mpimage
import numpy as np
import scipy


def create_kernel_derivatives():

    # centred i,j
    k_a_i = scipy.zeros((3, 3))
    k_b_i[0, 1] = -0.5
    k_b_i[2, 1] = 0.5
    k_c_j = scipy.zeros((3, 3))
    k_c_j[1, 2] = 0.5
    k_c_j[1, 0] = -0.5

    # backwards i,j
    k_b_i = scipy.zeros((3, 3))
    k_b_i[1, 1] = 1
    k_b_i[0, 1] = -1
    k_b_j = scipy.zeros((3, 3))
    k_b_j[1, 1] = 1
    k_b_j[1, 0] = -1

    # Kernels
    k_a_i = scipy.zeros((3, 3))
    k_a_i[1, 1] = -1
    k_a_i[2, 1] = 1
    k_a_j = scipy.zeros((3, 3))
    k_a_j[1, 1] = -1
    k_a_j[1, 2] = 1

    return k_a_i, k_a_j, k_b_i, k_b_j, k_a_i, k_c_j


def amle_inpainting(input_m, mask, fidelity, tolerance, max_iter, d_t):
    if input_m.dim == 3:
        M, N, C = input_m.shape
    else:
        M, N = input_m.shape
        C = 1
    # Derivatives Kernels
    k_a_i, k_a_j, k_b_i, k_b_j, k_c_i, k_c_j = create_kernel_derivatives()

    input_copy = input_m.copy()
    vec = scipy.zeros((M, N, 2))

    for c_iter in range(0, C):
        for iteration in range(0, max_iter):
            input_copy_x = cv.filter2D(input_copy[:, :, c_iter], -1, k_a_i)
            input_copy_y = cv.filter2D(input_copy[:, :, c_iter], -1, k_a_j)

            input_copy_xx = cv.filter2D(input_copy_x, -1, k_b_i)
            input_copy_xy = cv.filter2D(input_copy_x, -1, k_b_j)
            input_copy_yx = cv.filter2D(input_copy_y, -1, k_b_i)
            input_copy_yy = cv.filter2D(input_copy_y, -1, k_b_j)

            vec[:, :, 0] = cv.filter2D(input_copy[:, :, c_iter], -1, k_c_i)
            vec[:, :, 1] = cv.filter2D(input_copy[:, :, c_iter], -1, k_c_j)

            new_input_copy = input_copy[:, :, c_iter] + d_t * (
                input_copy_xx * vec[:, :, 0]**2 + input_copy_yy * vec[:, :, 1]**2 + (input_copy_xy + input_copy_yx) *
                (vec[:, :, 0] * vec[:, :, 1]) + fidelity * mask[:, :, c_iter] *
                (input_m[:, :, c_iter] - input_copy[:, :, c_iter]))

            diff_input = np.linalg.norm(
                new_input_copy.reshape(M * N, 1) - input_copy[:, :, c_iter].reshape(M * N, 1),
                2) / np.linalg.norm(new_input_copy.reshape(M * N, 1), 2)

            input_copy[:, :, c_iter] = new_input_copy

            if diff_input < tolerance:
                break

    if C == 1:
        mpimage.imsave("outuput_ample.png", input_copy[:, :, 0], cmap="gray")
    elif C == 3:
        mpimage.imsave("outuput_ample.png", input_copy)

    return input_copy

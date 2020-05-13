import cv2 as cv
import matplotlib.image as mpimage
import numpy as np
import scipy


def create_kernel_derivatives():

    # Kernels
    k_a_i = scipy.zeros((3, 3))
    k_a_i[1, 1] = -1
    k_a_i[2, 1] = 1
    k_a_j = scipy.zeros((3, 3))
    k_a_j[1, 1] = -1
    k_a_j[1, 2] = 1

    # backwards i,j
    k_b_i = scipy.zeros((3, 3))
    k_b_i[1, 1] = 1
    k_b_i[0, 1] = -1
    k_b_j = scipy.zeros((3, 3))
    k_b_j[1, 1] = 1
    k_b_j[1, 0] = -1

    # centred i,j
    k_a_i = scipy.zeros((3, 3))
    k_b_i[0, 1] = -0.5
    k_b_i[2, 1] = 0.5
    k_c_j = scipy.zeros((3, 3))
    k_c_j[1, 2] = 0.5
    k_c_j[1, 0] = -0.5

    return k_a_i, k_a_j, k_b_i, k_b_j, k_a_i, k_c_j


def amle_inpainting(input_m, mask, fidelity, tolerance, max_iter, d_t):
    if input_m.dim == 3:
        M, N, C = input_m.shape
    else:
        M, N = input_m.shape
        C = 1
    # Derivatives Kernels
    k_a_i, k_a_j, k_b_i, k_b_j, k_c_i, k_c_j = create_kernel_derivatives()

    u = input_m.copy()
    vec = scipy.zeros((M, N, 2))

    for c_iter in range(0, C):
        for iteration in range(0, max_iter):
            u_x = cv.filter2D(u[:, :, c_iter], -1, k_a_i)
            u_y = cv.filter2D(u[:, :, c_iter], -1, k_a_j)

            u_xx = cv.filter2D(u_x, -1, k_b_i)
            u_xy = cv.filter2D(u_x, -1, k_b_j)
            u_yx = cv.filter2D(u_y, -1, k_b_i)
            u_yy = cv.filter2D(u_y, -1, k_b_j)

            vec[:, :, 0] = cv.filter2D(u[:, :, c_iter], -1, k_c_i)
            vec[:, :, 1] = cv.filter2D(u[:, :, c_iter], -1, k_c_j)

            new_u = u[:, :, c_iter] + d_t * (
                u_xx * vec[:, :, 0] ** 2
                + u_yy * vec[:, :, 1] ** 2
                + (u_xy + u_yx) * (vec[:, :, 0] * vec[:, :, 1])
                + fidelity
                * mask[:, :, c_iter]
                * (input_m[:, :, c_iter] - u[:, :, c_iter])
            )

            diff_input = np.linalg.norm(
                new_u.reshape(M * N, 1) - u[:, :, c_iter].reshape(M * N, 1), 2
            ) / np.linalg.norm(new_u.reshape(M * N, 1), 2)

            u[:, :, c_iter] = new_u

            if diff_input < tolerance:
                break

    if C == 1:
        mpimage.imsave("outuput_amle.png", u[:, :, 0], cmap="gray")
    elif C == 3:
        mpimage.imsave("outuput_amle.png", u)

    return u

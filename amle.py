import cv2 as cv
import matplotlib.image as mpimage
import numpy as np
import scipy


def create_kernel_derivatives():
    # Kernels
    k_fi = scipy.zeros((3, 3))
    k_fi[1, 1] = -1
    k_fi[2, 1] = 1
    k_fj = scipy.zeros((3, 3))
    k_fj[1, 1] = -1
    k_fj[1, 2] = 1

    # backwards i,j
    k_bi = scipy.zeros((3, 3))
    k_bi[1, 1] = 1
    k_bi[0, 1] = -1
    k_bj = scipy.zeros((3, 3))
    k_bj[1, 1] = 1
    k_bj[1, 0] = -1

    # centred i,j
    k_ci = scipy.zeros((3, 3))
    k_bi[0, 1] = -0.5
    k_bi[2, 1] = 0.5
    k_cj = scipy.zeros((3, 3))
    k_cj[1, 2] = 0.5
    k_cj[1, 0] = -0.5
    return k_fi, k_fj, k_bi, k_bj, k_ci, k_cj


def amle_inpainting(input_m, mask, fidelity, tolerance, max_iter, dt):
    if input_m.dim == 3:
        M, N, C = input_m.shape
    else:
        M, N = input_m.shape
        C = 1
    # Derivatives Kernels
    k_fi, k_fj, k_bi, k_bj, k_ci, k_cj = create_kernel_derivatives()

    u = input_m.copy()
    v = scipy.zeros((M, N, 2))

    for c_iter in range(0, C):
        for iteration in range(0, max_iter):
            ux = cv.filter2D(u[:, :, c_iter], -1, k_fi)
            uy = cv.filter2D(u[:, :, c_iter], -1, k_fj)

            uxx = cv.filter2D(ux, -1, k_bi)
            uxy = cv.filter2D(ux, -1, k_bj)
            uyx = cv.filter2D(uy, -1, k_bi)
            uyy = cv.filter2D(uy, -1, k_bj)

            v[:, :, 0] = cv.filter2D(u[:, :, c_iter], -1, k_ci)
            v[:, :, 1] = cv.filter2D(u[:, :, c_iter], -1, k_cj)

            unew = u[:, :, c_iter] + dt * (uxx * v[:, :, 0] ** 2 + uyy * v[:, :, 1] ** 2 + (uxy + uyx) * (
                    v[:, :, 0] * v[:, :, 1]) + fidelity * mask[:, :, c_iter] * (
                                                       input_m[:, :, c_iter] - u[:, :, c_iter]))

            diff_u = np.linalg.norm(unew.reshape(M * N, 1) - u[:, :, c_iter].reshape(M * N, 1), 2) / np.linalg.norm(
                unew.reshape(M * N, 1), 2)

            u[:, :, c_iter] = unew

            if diff_u < tolerance:
                break

    if C == 1:
        mpimage.imsave("outuput_ample.png", u[:, :, 0], cmap="gray")
    elif C == 3:
        mpimage.imsave("outuput_ample.png", u)

    return u

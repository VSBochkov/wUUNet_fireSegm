import cv2
import numpy as np
import torch


def get_ad_segmask(actual_data, imh, imw, weighted=False, gauss=False, sigma_div=4):
    win = actual_data.shape[3]
    hwin = int(win / 2)
    MAX_I = actual_data.shape[0] - 1
    MAX_J = actual_data.shape[1] - 1

    if gauss:
        # ORIGIN
        gauss_kern_x = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)
        gauss_kern_y = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)

        gauss_kern = np.dot(gauss_kern_x, gauss_kern_y.transpose())
        gauss_kern = gauss_kern / gauss_kern.max()

        for i in range(0, actual_data.shape[0]):
            for j in range(0, actual_data.shape[1]):
                ad = np.multiply(actual_data[i, j, :, : win - 1, : win - 1], gauss_kern)
                if i == 0:
                    tmp = np.zeros((3, win - 1, win - 1), dtype=np.float32)
                    if j == 0:
                        tmp[:, hwin: win - 1, hwin: win - 1] = actual_data[0, 0, :, : hwin - 1, : hwin - 1]
                        ad[:, : hwin - 1, : hwin - 1] += np.multiply(tmp, gauss_kern)[:, hwin: win - 1, hwin: win - 1]
                        tmp[:, hwin: win - 1, : win - 1] = actual_data[0, 0, :, : hwin - 1, : win - 1]
                        ad[:, : hwin - 1, : win - 1] += np.multiply(tmp, gauss_kern)[:, hwin: win - 1, : win - 1]
                        tmp[:, : win - 1, hwin: win - 1] = actual_data[0, 0, :, : win - 1, : hwin - 1]
                        ad[:, : win - 1, : hwin - 1] += np.multiply(tmp, gauss_kern)[:, : win - 1, hwin: win - 1]
                    else:
                        tmp[:, hwin: win - 1, :] = actual_data[0, j, :, : hwin - 1, : win - 1]
                        ad[:, : hwin - 1, : win - 1] += np.multiply(tmp, gauss_kern)[:, hwin: win - 1, : win - 1]
                        if j == MAX_J:
                            tmp[:, hwin: win - 1, : hwin - 1] = actual_data[0, MAX_J, :, : hwin - 1, hwin: win - 1]
                            ad[:, : hwin - 1, hwin: win - 1] += np.multiply(tmp, gauss_kern)[:, hwin: win - 1,
                                                                : hwin - 1]
                elif j == 0:
                    tmp = np.zeros((3, win - 1, win - 1), dtype=np.float32)
                    tmp[:, :, hwin: win - 1] = actual_data[i, 0, :, : win - 1, : hwin - 1]
                    ad[:, : win - 1, : hwin - 1] += np.multiply(tmp, gauss_kern)[:, : win - 1, hwin: win - 1]
                if j == MAX_J:
                    tmp = np.zeros((3, win - 1, win - 1), dtype=np.float32)
                    tmp[:, : win - 1, : hwin - 1] = actual_data[i, MAX_J, :, : win - 1, hwin: win - 1]
                    ad[:, : win - 1, hwin: win - 1] += np.multiply(tmp, gauss_kern)[:, : win - 1, : hwin - 1]

                actual_data[i, j, :, : win - 1, : win - 1] = ad.copy()
                actual_data[i, j, :, win - 1, : win - 2] = actual_data[i, j, :, win - 2, : win - 2]
                actual_data[i, j, :, : win - 2, win - 1] = actual_data[i, j, :, : win - 2, win - 2]
                actual_data[i, j, :, win - 1, win - 1] = actual_data[i, j, :, win - 2, win - 2]

    ad_segmask = np.zeros((actual_data.shape[2], imh, imw), dtype=np.float32)

    if weighted:
        single_node_weight = 4.
        two_node_weight = 2.
        four_node_weight = 1.
    else:
        single_node_weight = 1.
        two_node_weight = 1.
        four_node_weight = 1.

    ad_segmask[:, :hwin, :hwin] = actual_data[0, 0, :, :hwin, :hwin] * single_node_weight
    ad_segmask[:, hwin * (MAX_I + 1): imh, :hwin] = actual_data[MAX_I, 0, :, hwin: imh - hwin * MAX_I, :hwin] * single_node_weight
    ad_segmask[:, :hwin, hwin * (MAX_J + 1): imw] = actual_data[0, MAX_J, :, :hwin, hwin: imw - hwin * MAX_J] * single_node_weight
    ad_segmask[:, hwin * (MAX_I + 1): imh, hwin * (MAX_J + 1): imw] = actual_data[0, MAX_J, :,
                                                                      hwin:imh - hwin * MAX_I,
                                                                      hwin:imw - hwin * MAX_J] * single_node_weight

    for i in range(1, actual_data.shape[0]):
        ad_segmask[:, i * hwin: (i + 1) * hwin, (MAX_J + 1) * hwin: imw] = \
            (actual_data[i, MAX_J, :, :hwin, hwin: imw - MAX_J * hwin] + actual_data[i - 1, MAX_J, :, hwin:win, hwin: imw - MAX_J * hwin]) * two_node_weight
        ad_segmask[:, i * hwin: (i + 1) * hwin, :hwin] = \
            (actual_data[i, 0, :, :hwin, :hwin] + actual_data[i - 1, 0, :, hwin:win, :hwin]) * two_node_weight

    for j in range(1, actual_data.shape[1]):
        ad_segmask[:, (MAX_I + 1) * hwin: imh, j * hwin: (j + 1) * hwin] = \
            (actual_data[MAX_I, j, :, hwin: imh - MAX_I * hwin, :hwin] + actual_data[MAX_I, j - 1, :, hwin: imh - MAX_I * hwin, hwin:win]) * two_node_weight
        ad_segmask[:, :hwin, j * hwin: (j + 1) * hwin] = \
            (actual_data[0, j, :, :hwin, :hwin] + actual_data[0, j - 1, :, :hwin, hwin:win]) * two_node_weight

    for i in range(1, actual_data.shape[0]):
        for j in range(1, actual_data.shape[1]):
            ad_segmask[:, i * hwin: (i + 1) * hwin, j * hwin: (j + 1) * hwin] = \
                (actual_data[i - 1, j - 1, :, hwin:win, hwin:win] + actual_data[i - 1, j, :, hwin:win, :hwin] + actual_data[i, j - 1, :, :hwin, hwin:win] + actual_data[i, j, :, :hwin, :hwin]) * four_node_weight

    return ad_segmask


def get_gauss_kern_cuda(win, sigma_div):
    # ORIGIN
    gauss_kern_x = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)
    gauss_kern_y = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)

    gauss_kern = np.dot(gauss_kern_x, gauss_kern_y.transpose())
    gauss_kern = gauss_kern / gauss_kern.max()
    return torch.from_numpy(gauss_kern).cuda()


def get_ad_segmask_cuda(actual_data, imh, imw, weighted=False, gauss=False, sigma_div=4, gauss_kern=None):
    win = actual_data.shape[3]
    hwin = int(win / 2)
    MAX_I = actual_data.shape[0] - 1
    MAX_J = actual_data.shape[1] - 1

    if gauss:
        for i in range(0, actual_data.shape[0]):
            for j in range(0, actual_data.shape[1]):
                ad = torch.mul(actual_data[i, j, :, : win - 1, : win - 1], gauss_kern)
                if i == 0:
                    tmp = torch.zeros((3, win - 1, win - 1), dtype=torch.float32, device="cuda")
                    if j == 0:
                        tmp[:, hwin: win - 1, hwin: win - 1] = actual_data[0, 0, :, : hwin - 1, : hwin - 1]
                        ad[:, : hwin - 1, : hwin - 1] += torch.mul(tmp, gauss_kern)[:, hwin: win - 1, hwin: win - 1]
                        tmp[:, hwin: win - 1, : win - 1] = actual_data[0, 0, :, : hwin - 1, : win - 1]
                        ad[:, : hwin - 1, : win - 1] += torch.mul(tmp, gauss_kern)[:, hwin: win - 1, : win - 1]
                        tmp[:, : win - 1, hwin: win - 1] = actual_data[0, 0, :, : win - 1, : hwin - 1]
                        ad[:, : win - 1, : hwin - 1] += torch.mul(tmp, gauss_kern)[:, : win - 1, hwin: win - 1]
                    else:
                        tmp[:, hwin: win - 1, :] = actual_data[0, j, :, : hwin - 1, : win - 1]
                        ad[:, : hwin - 1, : win - 1] += torch.mul(tmp, gauss_kern)[:, hwin: win - 1, : win - 1]
                        if j == MAX_J:
                            tmp[:, hwin: win - 1, : hwin - 1] = actual_data[0, MAX_J, :, : hwin - 1, hwin: win - 1]
                            ad[:, : hwin - 1, hwin: win - 1] += torch.mul(tmp, gauss_kern)[:, hwin: win - 1, : hwin - 1]
                elif j == 0:
                    tmp = torch.zeros((3, win - 1, win - 1), dtype=torch.float32, device="cuda")
                    tmp[:, :, hwin: win - 1] = actual_data[i, 0, :, : win - 1, : hwin - 1]
                    ad[:, : win - 1, : hwin - 1] += torch.mul(tmp, gauss_kern)[:, : win - 1, hwin: win - 1]
                if j == MAX_J:
                    tmp = torch.zeros((3, win - 1, win - 1), dtype=torch.float32, device="cuda")
                    tmp[:, : win - 1, : hwin - 1] = actual_data[i, MAX_J, :, : win - 1, hwin: win - 1]
                    ad[:, : win - 1, hwin: win - 1] += torch.mul(tmp, gauss_kern)[:, : win - 1, : hwin - 1]

                actual_data[i, j, :, : win - 1, : win - 1] = ad.clone()
                actual_data[i, j, :, win - 1, : win - 2] = actual_data[i, j, :, win - 2, : win - 2]
                actual_data[i, j, :, : win - 2, win - 1] = actual_data[i, j, :, : win - 2, win - 2]
                actual_data[i, j, :, win - 1, win - 1] = actual_data[i, j, :, win - 2, win - 2]

    ad_segmask = torch.zeros((actual_data.shape[2], imh, imw), dtype=torch.float32, device="cuda")

    if weighted:
        single_node_weight = 4.
        two_node_weight = 2.
        four_node_weight = 1.
    else:
        single_node_weight = 1.
        two_node_weight = 1.
        four_node_weight = 1.

    ad_segmask[:, :hwin, :hwin] = actual_data[0, 0, :, :hwin, :hwin] * single_node_weight
    ad_segmask[:, hwin * (MAX_I + 1): imh, :hwin] = actual_data[MAX_I, 0, :, hwin: imh - hwin * MAX_I, :hwin] * single_node_weight
    ad_segmask[:, :hwin, hwin * (MAX_J + 1): imw] = actual_data[0, MAX_J, :, :hwin, hwin: imw - hwin * MAX_J] * single_node_weight
    ad_segmask[:, hwin * (MAX_I + 1): imh, hwin * (MAX_J + 1): imw] = actual_data[0, MAX_J, :,
                                                                      hwin:imh - hwin * MAX_I,
                                                                      hwin:imw - hwin * MAX_J] * single_node_weight

    for i in range(1, actual_data.shape[0]):
        ad_segmask[:, i * hwin: (i + 1) * hwin, (MAX_J + 1) * hwin: imw] = \
            (actual_data[i, MAX_J, :, :hwin, hwin: imw - MAX_J * hwin] + actual_data[i - 1, MAX_J, :, hwin:win, hwin: imw - MAX_J * hwin]) * two_node_weight
        ad_segmask[:, i * hwin: (i + 1) * hwin, :hwin] = \
            (actual_data[i, 0, :, :hwin, :hwin] + actual_data[i - 1, 0, :, hwin:win, :hwin]) * two_node_weight

    for j in range(1, actual_data.shape[1]):
        ad_segmask[:, (MAX_I + 1) * hwin: imh, j * hwin: (j + 1) * hwin] = \
            (actual_data[MAX_I, j, :, hwin: imh - MAX_I * hwin, :hwin] + actual_data[MAX_I, j - 1, :, hwin: imh - MAX_I * hwin, hwin:win]) * two_node_weight
        ad_segmask[:, :hwin, j * hwin: (j + 1) * hwin] = \
            (actual_data[0, j, :, :hwin, :hwin] + actual_data[0, j - 1, :, :hwin, hwin:win]) * two_node_weight

    for i in range(1, actual_data.shape[0]):
        for j in range(1, actual_data.shape[1]):
            ad_segmask[:, i * hwin: (i + 1) * hwin, j * hwin: (j + 1) * hwin] = \
                (actual_data[i - 1, j - 1, :, hwin:win, hwin:win] + actual_data[i - 1, j, :, hwin:win, :hwin] + actual_data[i, j - 1, :, :hwin, hwin:win] + actual_data[i, j, :, :hwin, :hwin]) * four_node_weight

    return ad_segmask


def get_ad_segmask_448(actual_data, imh, imw, weighted=False, gauss=False, sigma_div=4):
    win = actual_data.shape[2]
    win_step = int((imw - win) / (actual_data.shape[0] - 1))
    # ORIGIN
    gauss_kern_x = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)
    gauss_kern_y = cv2.getGaussianKernel(win - 1, float(win) / sigma_div)

    gauss_kern = np.dot(gauss_kern_x, gauss_kern_y.transpose())
    gauss_kern = gauss_kern / gauss_kern.max()
    ad_segmask = np.zeros((actual_data.shape[1], imh, imw), dtype=np.float32)
    if actual_data.shape[0] == 2:
        # win_step = 192
        if gauss:
            tmp = actual_data[0, :, : win - 1, 1: win]
            actual_data[0, :, : win - 1, win_step + 1: win] = np.multiply(tmp, gauss_kern)[:, :, win_step: win - 1].copy()
            tmp = actual_data[1, :, : win - 1, 0: win - 1]
            actual_data[1, :, : win - 1, 0: win - win_step - 1] = np.multiply(tmp, gauss_kern)[:, :, 0: win - win_step - 1].copy()

        if weighted:
            single_node_weight = 2.
            two_node_weight = 1.
        else:
            single_node_weight = 1.
            two_node_weight = 1.

        ad_segmask[:, :, 0: win_step] = single_node_weight * actual_data[0, :, :, : win_step]
        ad_segmask[:, :, win_step + 1: win] = two_node_weight * (actual_data[0, :, :, win_step + 1: win] +
                                                                 actual_data[1, :, :, 0: win - win_step - 1])
        ad_segmask[:, :, win + 1: imw] = single_node_weight * actual_data[1, :, :, win - win_step: win - 1]
    elif actual_data.shape[0] == 3:
        # win_step = 96
        if gauss:
            tmp = actual_data[0, :, : win - 1, 1: win]
            actual_data[0, :, : win - 1, win_step + 1: win] = np.multiply(tmp, gauss_kern)[:, :, win_step: win - 1].copy()
            tmp = actual_data[1, :, : win - 1, : win - 1]
            actual_data[1, :, : win - 1, : win - 1] = np.multiply(tmp, gauss_kern).copy()
            tmp = actual_data[2, :, : win - 1, : win - 1]
            actual_data[2, :, : win - 1, 0: win - win_step - 1] = np.multiply(tmp, gauss_kern)[:, :, 0: win - win_step - 1].copy()

        if weighted:
            single_node_weight = 6.
            two_node_weight = 3.
            three_node_weight = 2.
        else:
            single_node_weight = 1.
            two_node_weight = 1.
            three_node_weight = 1.

        ad_segmask[:, :, 0: win_step] = single_node_weight * actual_data[0, :, :, : win_step]
        ad_segmask[:, :, win_step + 1: 2 * win_step] = two_node_weight * (
                actual_data[0, :, :, win_step + 1: 2 * win_step] + actual_data[1, :, :, 0: win_step - 1])
        ad_segmask[:, :, 2 * win_step + 1: win] = three_node_weight * (
                actual_data[0, :, :, 2 * win_step + 1: win] + actual_data[1, :, :, win_step: win - win_step - 1]
                + actual_data[2, :, :, 0: win - 2 * win_step - 1])
        ad_segmask[:, :, win + 1: win + win_step] = two_node_weight * (
                actual_data[1, :, :, win - win_step: win - 1] + actual_data[2, :, :, win - 2 * win_step: win - win_step - 1])
        ad_segmask[:, :, win + win_step + 1: imw] = single_node_weight * actual_data[2, :, :, win - win_step: win - 1]

    return ad_segmask


def visualize_ad_segmask(image, ad_segmask):
    # HSV
    classes = np.array([[0xff, 0, 0], [0xff, 0xcc, 0], [0xff, 0xff, 0]], dtype=np.float32)
    print('image.shape = {}, ad_segmask.shape = {}'.format(image.shape, ad_segmask.shape))
    ad_segmask = torch.clamp(ad_segmask, 0.)
    maximals, argmaximals = torch.max(ad_segmask, 2, False)
    max, _ = torch.max(maximals, 0)
    max, _ = torch.max(max, 0)
    max = max.item()
    maximals = maximals.cpu().numpy()
    argmaximals = argmaximals.cpu().numpy()
    image = torch.from_numpy(image)
    overlay_float = torch.as_tensor(image, dtype=torch.float32, device=torch.device('cuda')) #/ 2
    overlay_float = overlay_float.cpu().numpy()
    ___mxmls = []
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if maximals[i, j] > 0.05 * max:
                ___mxmls.append(maximals[i, j])
                overlay_float[i, j] = maximals[i, j] / max * classes[argmaximals[i, j]] + (1 - maximals[i, j] / max) * overlay_float[i, j]

    print('max = {}, np.array(___mxmls).mean() = {}'.format(max, np.array(___mxmls, dtype=np.float32).mean()))
    overlay = np.asarray(overlay_float, dtype=np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

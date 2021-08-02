import json
import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from keras_dataset import Dataset
from keras_unet_models import wuunet, unet
import keras_utils


class Validation(object):
    def __init__(self, dataset_variant):
        self.activation_thrs = self.get_thresholds_array()
        self.mult_num_class = 3
        self.input_size = (224, 224)
        self.division_coeff = 1
        self.dataset_variant = dataset_variant

    @staticmethod
    def get_thresholds_array():
        begin, end, step = 0., 1., 1e-2
        nodes = int((end - begin) / step + 1)
        return np.linspace(begin, end, nodes)

    def get_model(self, model_name, snapshot_variant):
        dataset = Dataset('../dataset/ow_224/' + self.dataset_variant, batch_size=1, shuffle=False)
        n_class = 1 if model_name.find('_n1') != -1 else self.mult_num_class
        light = model_name.find('light') != -1
        main_metric = 'jaccard'
        if 'wuunet' in model_name:
            model = wuunet(n_class, batch_size=1, light=light)
        else:
            model = unet(n_class, batch_size=1, light=light)
        high_level_dir = 'output/keras'
        model.load_weights(join(join(join(high_level_dir, model_name), 'snapshots'), snapshot_variant))
        return model, n_class, main_metric, dataset

    def get_metrics(self, ad_mult, gt_mult):
        ad_bin = keras_utils.multiclass_to_binary_label_map(ad_mult, self.mult_num_class)
        gt_bin = keras_utils.multiclass_to_binary_label_map(gt_mult, self.mult_num_class)
        jaccard_mults = keras_utils.np_jaccard_acc(ad_mult, gt_mult)
        jaccard_bins = keras_utils.np_jaccard_acc(ad_bin, gt_bin)
        return jaccard_mults.mean(), jaccard_bins.mean()

    def get_metrics_for_all_samples(self, model_name, snapshot_variant, threshold, samples_num):
        model_metrics_dir = 'output/evaluation/ow_224/{}/{}/'.format(model_name, snapshot_variant)
        _jaccard_mult_list = np.zeros(samples_num, dtype=np.float32)
        _jaccard_bin_list = np.zeros(samples_num, dtype=np.float32)
        for j in range(0, samples_num):
            outputs = np.load(join(model_metrics_dir, 'ad_segmask{}.npy'.format(j)))
            mult_lbl = np.load(join(model_metrics_dir, 'gt_segmask{}.npy'.format(j)))
            outs = outputs - threshold
            outs = outs.clip(min=0)
            maximals = outs.max(axis=2, keepdims=True)
            argmaximals = np.expand_dims(outs.argmax(axis=2), axis=2)
            outs = argmaximals + np.asarray(maximals > 0, dtype=np.int64)
            outs = keras_utils.get_multiclass_label_map(np.squeeze(outs), self.mult_num_class)
            _jaccard_mult_list[j], _jaccard_bin_list[j] = self.get_metrics(outs, mult_lbl)
        return _jaccard_mult_list, _jaccard_bin_list

    def metrics(self, samples_num, threshold, model_name, snapshot_variant):
        _jaccard_mult_list, _jaccard_bin_list = self.get_metrics_for_all_samples(
            model_name,
            snapshot_variant,
            threshold,
            samples_num
        )
        jaccard_mult_mean = _jaccard_mult_list.mean()
        jaccard_mult_std = np.std(_jaccard_mult_list)
        jaccard_bin_mean = _jaccard_bin_list.mean()
        jaccard_bin_std = np.std(_jaccard_bin_list)
        return jaccard_mult_mean, jaccard_mult_std, jaccard_bin_mean, jaccard_bin_std

    def get_thresholded_metrics(self, model_name, snapshot_variant, samples_num):
        jaccard_mult_mean = np.zeros(len(self.activation_thrs), dtype=np.float32)
        jaccard_mult_std = np.zeros(len(self.activation_thrs), dtype=np.float32)
        jaccard_bin_mean = np.zeros(len(self.activation_thrs), dtype=np.float32)
        jaccard_bin_std = np.zeros(len(self.activation_thrs), dtype=np.float32)
        for i, threshold in enumerate(tqdm(self.activation_thrs)):
            jaccard_mult_mean[i], jaccard_mult_std[i], jaccard_bin_mean[i], jaccard_bin_std[i] = self.metrics(
                samples_num, threshold, model_name, snapshot_variant)
        return jaccard_mult_mean, jaccard_mult_std, jaccard_bin_mean, jaccard_bin_std, samples_num

    def get_all_metrics(self, model, model_name, snapshot_variant, dataset):
        model_metrics_dir = 'output/evaluation/ow_224/{}/{}'.format(model_name, snapshot_variant)
        if not os.path.exists(model_metrics_dir):
            os.makedirs(model_metrics_dir)
        for i in tqdm(range(0, len(dataset))):
            image, label = dataset[i]
            if model_name.find('wuunet') != -1:
                output_mul = model.predict(image)['out_mult']
            else:
                output_mul = model.predict(image)
            ad_segmask = output_mul[0].astype(np.float32)
            gt_segmask = label[0].astype(np.float32)
            np.save(join(model_metrics_dir, 'ad_segmask{}.npy'.format(i)), ad_segmask)  # -1,+1 but not binary
            np.save(join(model_metrics_dir, 'gt_segmask{}.npy'.format(i)), gt_segmask)
        return len(dataset)

    def autolabel(self, ax, rects, width):
        x = rects[0].get_x() - width / 10
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > 0.:
                ax.text(x, height, '{0:.4f}'.format(height), ha='right', va='bottom')

    def draw_barchart(self, model_metrics, step=4.5):
        jaccard_mult_means = [0]
        jaccard_mult_means.extend([model_metrics[model_name][3] for model_name in model_metrics])
        jaccard_mult_stds = [0]
        jaccard_mult_stds.extend([model_metrics[model_name][4] for model_name in model_metrics])
        jaccard_bin_means = [0]
        jaccard_bin_means.extend([model_metrics[model_name][7] for model_name in model_metrics])
        jaccard_bin_stds = [0]
        jaccard_bin_stds.extend([model_metrics[model_name][8] for model_name in model_metrics])

        ind = np.arange(len(model_metrics) + 1) * step  # the x locations for the groups
        width = step / 6  # the width of the bars

        fig, ax = plt.subplots(figsize=(len(model_metrics) * 4 + 6, 8))
        colors = ['#7f7f7f', '#9f9f9f', '#bfbfbf', '#dfdfdf']
        rects3 = ax.bar(ind + 2 * width + 1, jaccard_mult_means, width, yerr=jaccard_mult_stds,
                        color=colors[3], label='JACCARD MULTICLASS')

        for i in range(0, len(rects3)):
            self.autolabel(ax, [rects3[i]], width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Value')
        ax.set_title('Pixel-wise segmentation metrics')
        ax.set_xticks(ind + step)
        labels = ['']
        labels.extend(model_metrics.keys())
        ax.set_xticklabels(labels)
        ax.legend(loc='upper left')

        plt.show()

    def draw_max_metric(self, plt, metric_id, metric, metric_name, ytext):
        plt.annotate(metric_name + ' = {0:.3f}'.format(metric), xy=(self.activation_thrs[metric_id], metric), xytext=(self.activation_thrs[metric_id] + 0.01, ytext))#,
        plt.axvline(x=self.activation_thrs[metric_id], ymin=0, ymax=ytext, linestyle='--', color='#7f7f7f')

    def plot_thresholds(self, jaccard_mults, jaccard_bins):
        plt.figure(figsize=(15, 5))
        plt.plot(self.activation_thrs, jaccard_bins, linestyle='--', color='#7f7f7f')
        plt.plot(self.activation_thrs, jaccard_mults, linestyle='--', color='#afafaf')
        legend = ['BINARY JACCARD', 'MULTICLASS JACCARD']
        plt.grid(True)
        plt.legend(legend, loc='upper left', fontsize=14)
        plt.xlabel('threshold (distance)', fontsize=18)
        plt.ylabel('rate', fontsize=18)
        jba = np.argmax(jaccard_bins)
        jma = np.argmax(jaccard_mults)
        metrics = np.array([jaccard_mults[jma]])
        self.draw_max_metric(plt, jma, jaccard_mults[jma], 'MULT JACCARD MAX', metrics.max() + 0.25)
        self.draw_max_metric(plt, jba, jaccard_bins[jba], 'BIN JACCARD MAX', metrics.max() + 0.15)
        plt.ylim(top=metrics.max() + 0.30)
        plt.tick_params(labelsize=14)

    def calculate_thresholds(self, ml_model_name, snapshot_variant):
        model_metrics_dir = 'output/evaluation/ow_224/{}/{}'.format(ml_model_name, snapshot_variant)
        if os.path.exists(join(model_metrics_dir, 'jaccard_bin_mean.npy')):
            jm_mean = np.load(join(model_metrics_dir, 'jaccard_mult_mean.npy'))
            jm_std = np.load(join(model_metrics_dir, 'jaccard_mult_std.npy'))
            jb_mean = np.load(join(model_metrics_dir, 'jaccard_bin_mean.npy'))
            jb_std = np.load(join(model_metrics_dir, 'jaccard_bin_std.npy'))
        else:
            model, num_classes, main_metric, dl_test = self.get_model(ml_model_name, snapshot_variant)
            samples_num = self.get_all_metrics(model, ml_model_name, snapshot_variant, dl_test)
            jm_mean, jm_std, jb_mean, jb_std, samples_num = self.get_thresholded_metrics(
                ml_model_name, snapshot_variant, samples_num)
            if not os.path.exists(model_metrics_dir):
                os.makedirs(model_metrics_dir)
            np.save(join(model_metrics_dir, 'jaccard_mult_mean'), jm_mean)
            np.save(join(model_metrics_dir, 'jaccard_mult_std'), jm_std)
            np.save(join(model_metrics_dir, 'jaccard_bin_mean'), jb_mean)
            np.save(join(model_metrics_dir, 'jaccard_bin_std'), jb_std)
        best_thr_id = np.argmax(jm_mean)
        best_thr = self.activation_thrs[best_thr_id]
        print('[{}_{}, threshold = {}]:\n'
              'MULTICLASS JACCARD [val = {}, std = {}]\n'
              'BINARY JACCARD [val = {}, std = {}]\n'
                  .format(ml_model_name, snapshot_variant, best_thr,
                          jm_mean[best_thr_id], jm_std[best_thr_id],
                          jb_mean[best_thr_id], jb_std[best_thr_id]))
        #self.plot_thresholds(dm_mean, jm_mean, db_mean, jb_mean)
        return (
            best_thr, jm_mean[best_thr_id], jm_std[best_thr_id], jb_mean[best_thr_id], jb_std[best_thr_id]
        )

    def print_metrics_at(self, ml_model_name, snapshot_variant, thr):
        model_metrics_dir = 'output/evaluation/ow_224/{}/{}'.format(ml_model_name, snapshot_variant)
        if os.path.exists(model_metrics_dir):
            jm_mean = np.load(join(model_metrics_dir, 'jaccard_mult_mean.npy'))
            jm_std = np.load(join(model_metrics_dir, 'jaccard_mult_std.npy'))
            jb_mean = np.load(join(model_metrics_dir, 'jaccard_bin_mean.npy'))
            jb_std = np.load(join(model_metrics_dir, 'jaccard_bin_std.npy'))
            thr_id = self.activation_thrs.index(thr)
            print('[{}_{}, threshold = {}]:\n'
                  'MULTICLASS JACCARD [val = {}, std = {}]\n'
                  'BINARY JACCARD [val = {}, std = {}]\n'
                  .format(ml_model_name, snapshot_variant, thr, jm_mean[thr_id], jm_std[thr_id], jb_mean[thr_id], jb_std[thr_id]))
    #
    # ##############################
    # ######## VIZUALIZATION #######
    # ##############################
    # def onehot_to_label(self, onehot3d_matrix_chw):
    #     _, imh, imw = onehot3d_matrix_chw.shape
    #     onehot3d_matrix_hwc = onehot3d_matrix_chw.transpose(0, 1).transpose(1, 2)
    #     red_matrix_chw = torch.mul(
    #         torch.ones_like(onehot3d_matrix_hwc).cuda(), torch.tensor([1, 0, 0]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     orange_matrix_chw = torch.mul(
    #         torch.ones_like(onehot3d_matrix_hwc).cuda(), torch.tensor([0, 1, 0]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     yellow_matrix_chw = torch.mul(
    #         torch.ones_like(onehot3d_matrix_hwc).cuda(), torch.tensor([0, 0, 1]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     red_new_chw = torch.ones((1, imh, imw), dtype=torch.uint8).cuda() * 1
    #     orange_new_chw = torch.ones((1, imh, imw), dtype=torch.uint8).cuda() * 2
    #     yellow_new_chw = torch.ones((1, imh, imw), dtype=torch.uint8).cuda() * 3
    #     labelmap_chw = torch.zeros((1, imh, imw), dtype=torch.uint8).cuda()
    #     labelmap_chw = torch.where(onehot3d_matrix_chw[0] == 1, red_new_chw, labelmap_chw)
    #     labelmap_chw = torch.where(onehot3d_matrix_chw[1] == 1, orange_new_chw, labelmap_chw)
    #     labelmap_chw = torch.where(onehot3d_matrix_chw[2] == 1, yellow_new_chw, labelmap_chw)
    #     return labelmap_chw
    #
    # def rawonehot_to_onehot(self, raw_actual_data_oh, threshold):
    #     raw_actual_data_oh = raw_actual_data_oh.clone() - threshold
    #     raw_actual_data_oh = torch.clamp(raw_actual_data_oh, 0.)
    #     maximals, argmaximals = torch.max(raw_actual_data_oh, 0, True)
    #     actual_data_lbl = torch.as_tensor(
    #         argmaximals + torch.as_tensor(maximals > 0, dtype=torch.int64, device=torch.device('cuda')),
    #         dtype=torch.float32, device=torch.device('cuda')
    #     )
    #     return pytorch_utils.get_multiclass_label_map(actual_data_lbl.squeeze(), raw_actual_data_oh.shape[0])
    #
    # def get_overlay(self, overlay, labeled_img, imh, imw):
    #     red = np.multiply(np.ones((imh, imw, 3), dtype=np.uint8), np.array([0xff, 0x00, 0x00]))
    #     yellow = np.multiply(np.ones((imh, imw, 3), dtype=np.uint8), np.array([0xff, 0xff, 0x00]))
    #     orange = np.multiply(np.ones((imh, imw, 3), dtype=np.uint8), np.array([0xff, 0xcc, 0x00]))
    #     red = torch.as_tensor(np.transpose(red, (2, 0, 1)), dtype=torch.int64).cuda()
    #     yellow = torch.as_tensor(np.transpose(yellow, (2, 0, 1)), dtype=torch.int64).cuda()
    #     orange = torch.as_tensor(np.transpose(orange, (2, 0, 1)), dtype=torch.int64).cuda()
    #     overlay_cvted = cv2.cvtColor(overlay.transpose(0, 1).transpose(1, 2).data.cpu().numpy().copy(),
    #                                  cv2.COLOR_BGR2RGB)
    #     overlay_cvted = torch.as_tensor(overlay_cvted, dtype=torch.int64).cuda().transpose(2, 0).transpose(1, 2)
    #     labeled_img = torch.as_tensor(labeled_img, dtype=torch.int64).cuda()
    #     overlay = torch.as_tensor(overlay, dtype=torch.int64).cuda()
    #     overlay = torch.where(labeled_img == 1, (red + overlay_cvted) / 2, overlay)
    #     overlay = torch.where(labeled_img == 2, (orange + overlay_cvted) / 2, overlay)
    #     overlay = torch.where(labeled_img == 3, (yellow + overlay_cvted) / 2, overlay)
    #     # overlay = torch.where(labeled_img == 1, red, overlay)
    #     # overlay = torch.where(labeled_img == 2, orange, overlay)
    #     # overlay = torch.where(labeled_img == 3, yellow, overlay)
    #     overlay = torch.as_tensor(overlay, dtype=torch.uint8).cuda()
    #     return cv2.cvtColor(
    #         overlay.transpose(0, 1).transpose(1, 2).cpu().numpy(),
    #         cv2.COLOR_BGR2RGB
    #     )
    #
    # def get_diff(self, overlay, actual_data, ground_truth, imh, imw, window):
    #     red = torch.mul(
    #         torch.ones((imh, imw, 3), dtype=torch.float32).cuda(), torch.tensor([0, 0, 0xff]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     yellow = torch.mul(
    #         torch.ones((imh, imw, 3), dtype=torch.float32).cuda(), torch.tensor([0, 0xff, 0]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     orange = torch.mul(
    #         torch.ones((imh, imw, 3), dtype=torch.float32).cuda(), torch.tensor([0xff, 0, 0]).cuda()
    #     ).transpose(0, 2).transpose(1, 2)
    #     actual_data = torch.as_tensor(actual_data, dtype=torch.float32).cuda()
    #     ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32).cuda()
    #     overlay = torch.as_tensor(overlay, dtype=torch.float32).cuda()
    #     bkg_color = torch.tensor([0xff, 0xff, 0xff]).cuda()  # torch.tensor([0x7f, 0x7f, 0x7f]).cuda()
    #     ad_overlay = torch.mul(torch.ones((imh, imw, 3), dtype=torch.float32).cuda(), bkg_color)
    #     ad_overlay = ad_overlay.transpose(0, 2).transpose(1, 2)
    #     gt_overlay = ad_overlay.clone()
    #     ad_overlay = torch.where(actual_data == 1, red, ad_overlay)
    #     ad_overlay = torch.where(actual_data == 2, orange, ad_overlay)
    #     ad_overlay = torch.where(actual_data == 3, yellow, ad_overlay)
    #     gt_overlay = torch.where(ground_truth == 1, red, gt_overlay)
    #     gt_overlay = torch.where(ground_truth == 2, orange, gt_overlay)
    #     gt_overlay = torch.where(ground_truth == 3, yellow, gt_overlay)
    #     overlay = gt_overlay * window + (1 - window) * ad_overlay
    #     overlay = torch.as_tensor(overlay, dtype=torch.uint8).transpose(0, 1).transpose(1, 2)
    #     return overlay.data.cpu().numpy()
    #
    # def draw_image_label(self, overlay, data_label, string):
    #     def get_label_color(subzone):
    #         b = subzone[0].mean()
    #         g = subzone[1].mean()
    #         r = subzone[2].mean()
    #         if int(b) not in range(100, 150) and int(g) not in range(100, 150) and int(r) not in range(100, 150):
    #             return 255 - b, 255 - g, 255 - r
    #         else:
    #             return 255, 255, 255
    #
    #     left_up_cnt = 0
    #     right_up_cnt = 0
    #     if data_label is not None:
    #         ones = torch.ones(data_label.shape, dtype=torch.uint8).cuda()
    #         zeros = torch.zeros_like(ones).cuda()
    #         count_fire = torch.where(data_label > 0, ones, zeros)
    #         left_up_cnt = count_fire[0, :35, :200].sum()
    #         right_up_cnt = count_fire[0, :35, -200:].sum()
    #         left_down_cnt = count_fire[0, -35:, :200].sum()
    #         right_down_cnt = count_fire[0, -35:, -200:].sum()
    #     if left_up_cnt <= right_up_cnt:
    #         b, g, r = get_label_color(overlay[:35, :150])
    #         pos_x = 10
    #         pos_y = 35
    #     else:
    #         b, g, r = get_label_color(overlay[:35, -150:])
    #         pos_x = overlay.shape[1] - 200
    #         pos_y = 35
    #     cv2.putText(overlay, string, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (b, g, r), 2)
    #     return overlay
    #
    # def calc_hist(self, jaccard_mult, _range):
    #     X = np.linspace(0, 100, 100 / _range + 1)
    #     Y = np.zeros_like(X)
    #     for value in jaccard_mult:
    #         Y[int(np.round(value * 100 / _range))] += 1
    #     return X, Y
    #
    # def draw_graphics(self, imh, imw, styles_map, jaccard_mults, short_model_names, metrics, _range):
    #     fig = plt.figure()
    #     fig.add_subplot(111)
    #
    #     for i, (key, jaccard_mult) in enumerate(jaccard_mults):
    #         X, Y = self.calc_hist(jaccard_mult, _range)
    #         cur_X = int(np.round(metrics[key][1] * 100 / _range))
    #         cur_Y = Y[cur_X]
    #         spline = interp1d(X, Y, 'cubic')
    #         X = np.linspace(0, 100, 1001)
    #         Y = spline(X)
    #         plt.plot(X, Y, styles_map[key], label=short_model_names[key])
    #         plt.plot(cur_X * _range, cur_Y, styles_map[key] + 'o')
    #     plt.legend()
    #     plt.title('Current multiclass jaccard values on histograms')
    #     fig.canvas.draw()
    #
    #     # Now we can save it to a numpy array.
    #     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     return cv2.resize(data, (imw, imh))
    #
    # def visualize_frame_result(self, image_name, short_model_names, raw_actual_data_oh_map, thresholds_map, ground_truth_oh, styles_map):
    #     image = cv2.imread(image_name)
    #     imh, imw, _ = image.shape
    #     overlay = np.zeros((int(imh * 3 + 100), int(imw * len(raw_actual_data_oh_map)), 3), dtype=np.uint8)
    #     overlay[: imh, : imw] = image.copy()
    #     image = torch.as_tensor(image, dtype=torch.uint8).cuda()
    #     image = image.transpose(2, 0).transpose(1, 2)
    #     actual_data_oh_map = {
    #         MODEL: self.rawonehot_to_onehot(raw_actual_data_oh_map[MODEL], thresholds_map[MODEL])
    #         for MODEL in raw_actual_data_oh_map
    #     }
    #     actual_data_lbl_map = {MODEL: self.onehot_to_label(actual_data_oh_map[MODEL]) for MODEL in actual_data_oh_map}
    #     ground_truth_lbl = self.onehot_to_label(ground_truth_oh)
    #     overlay[: imh] = self.draw_image_label(
    #         overlay[: imh], None, "Original image"
    #     )
    #     overlay[: imh, imw: 2 * imw] = self.get_overlay(image, ground_truth_lbl, imh, imw)
    #     overlay[: imh, imw: 2 * imw] = self.draw_image_label(
    #         overlay[: imh, imw: 2 * imw], ground_truth_lbl, "Human labeled"
    #     )
    #     metrics = {}
    #     jaccard_mults = []
    #     over = {}
    #     for i, key in enumerate(actual_data_lbl_map):
    #         actual_data_lbl = actual_data_lbl_map[key]
    #         over[key] = self.get_overlay(image, actual_data_lbl, imh, imw)
    #         overlay[imh: imh * 2, i * imw: (i + 1) * imw] = over[key].copy()
    #         overlay[imh: imh * 2, i * imw: (i + 1) * imw] = self.draw_image_label(
    #             overlay[imh: imh * 2, i * imw: (i + 1) * imw], actual_data_lbl, short_model_names[key]  # "NN output"
    #         )
    #         overlay[imh * 2: imh * 3, i * imw: (i + 1) * imw] = self.get_diff(
    #             torch.ones((3, imh, imw), dtype=torch.uint8).cuda() * 127,
    #             actual_data_lbl, ground_truth_lbl, imh, imw, 0.5)
    #         overlay[imh * 2: imh * 3, i * imw: (i + 1) * imw] = self.draw_image_label(
    #             overlay[imh * 2: imh * 3, i * imw: (i + 1) * imw], actual_data_lbl, "Difference"
    #         )
    #         jaccard_mult, jaccard_bin = self.get_metrics(
    #             torch.unsqueeze(actual_data_oh_map[key], 0),
    #             torch.unsqueeze(ground_truth_oh, 0)
    #         )
    #         metrics[key] = (jaccard_mult.data.cpu().numpy(),
    #                         jaccard_bin.data.cpu().numpy())
    #         for j, metric_string in enumerate([
    #             "Accuracy:",
    #             "Binary:    [Jaccard = {}%]".format(float(int(jaccard_bin * 1000) / 10)),
    #             "Multiclass: [Jaccard = {}%]".format(float(int(jaccard_mult * 1000) / 10))
    #         ]):
    #             cv2.putText(
    #                 overlay[3 * imh:, i * imw: (i + 1) * imw],
    #                 metric_string,
    #                 (35, 20 + j * 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.8,
    #                 (0xff, 0xff, 0xff),
    #                 2
    #             )
    #         key_list = key.split('/')
    #         model_name, snapshot_variant = key_list[1], key_list[2]
    #         jaccard_mults.append(
    #             (key, np.load('output/evaluation/ow_224/{}/{}/'.format(model_name, snapshot_variant) + 'jaccard_mult.npy'))
    #         )
    #     graphic = self.draw_graphics(imh, imw, styles_map, jaccard_mults, short_model_names, metrics, 5)
    #     if len(actual_data_lbl_map) == 3:
    #         overlay[: imh, (len(actual_data_lbl_map) - 1) * imw:] = graphic
    #     overlay[3 * imh:] = 0xff - overlay[3 * imh:]
    #     return overlay, metrics, graphic, over
    #
    # def calc_metrics_arrays_to_graphic(self, model_name, snapshot_variant, threshold, gauss_div):
    #     model_metrics_dir = 'output/evaluation/ow_224/{}/{}'.format(model_name, snapshot_variant)
    #     if not os.path.exists(join(model_metrics_dir, 'jaccard_mult.npy')):
    #         self.gauss_sigma_div = gauss_div
    #         model, num_classes, main_metric, dl_test = self.get_model(model_name, snapshot_variant)
    #         samples_num = self.get_all_metrics(model, model_name, snapshot_variant, dl_test)
    #         _jaccard_mult_list, _jaccard_bin_list = self.get_metrics_for_all_samples(
    #             model_name, snapshot_variant, threshold, samples_num
    #         )
    #         if not os.path.exists(model_metrics_dir):
    #             os.makedirs(model_metrics_dir)
    #         np.save(join(model_metrics_dir, 'jaccard_mult'), _jaccard_mult_list.data.cpu().numpy())
    #         np.save(join(model_metrics_dir, 'jaccard_bin'), _jaccard_bin_list.data.cpu().numpy())
    #     else:
    #         jaccard_mult = np.load(join(model_metrics_dir, 'jaccard_mult.npy'))
    #         samples_num = jaccard_mult.shape[0]
    #     return samples_num

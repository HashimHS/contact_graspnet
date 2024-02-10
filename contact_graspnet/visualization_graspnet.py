#!/home/graspnetthesis/miniconda3/envs/graspnet/bin/python
# -*- coding: utf-8 -*-
from visualization_utils import show_image, visualize_grasps
import argparse
import numpy as np
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default='results/predictions_000000.npz', help='Path to npz file')
    parser.add_argument('--n', type=int, default=5, help='Number of grasps to return per object')
    FLAGS = parser.parse_args()

    result = np.load(FLAGS.p, allow_pickle=True)
    pred_grasps_cam = result.item()['pred_grasps_cam']
    scores = result.item()['scores']
    contact_pts = result.item()['contact_pts']
    pc_colors = result.item()['pc_colors']
    pc_full = result.item()['pc_full']
    rgb = result.item()['rgb']
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    segmap = result.item()['segmap']

    # Top n grasps for each object
    n = FLAGS.n
    for k, v in scores.items():
        if len(v) > n:
            top_idx = np.argsort(v)[-n:][::-1]
            # top_idx = [0]
            scores[k] = v[top_idx]
            pred_grasps_cam[k] = pred_grasps_cam[k][top_idx]

        grasp_idx = 1
    for k in scores.keys():
        pred_grasps_cam[k] = [pred_grasps_cam[k][grasp_idx-1]]
        scores[k] = [scores[k][grasp_idx-1]]

    # Visualize results
    show_image(rgb, segmap)
    visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)



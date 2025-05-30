# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import KITTIOdomDataset
import datasets
import networks


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs

def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2) # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10" or opt.eval_split == "endo" or opt.eval_split == "endo_2", \
        "eval_split should be either odom_9 or odom_10"

    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files_depth_S1.txt"))
    # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files_pose_B1.txt"))
    # sequence_id = int(opt.eval_split.split("_")[1])

    # filenames = readlines(
    #     os.path.join(os.path.dirname(__file__), "splits", "odom",
    #                  "test_files_{:02d}.txt".format(sequence_id)))

    # dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
    #                            [0, 1], 4, is_train=False)

    # 使用separate_resnet训练的模型加载数据集
    # dataset = datasets.EndoRAWDataset(opt.data_path, filenames,
    #                                        opt.height, opt.width,
    #                                        [0, 1], 4, is_train=False)
    
    # 使用posecnn训练的模型加载数据集
    dataset = datasets.EndoRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width,
                                           [0, 1], 4, is_train=False)
    
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    # 使用separate_resnet训练的模型读取
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    # 使用posecnn训练的模型读取
    # pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    
    # 使用separate_resnet训练的模型加载
    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    # 使用posecnn训练的模型加载
    # pose_decoder = networks.PoseCNN(2)
    # pose_decoder.load_state_dict(torch.load(pose_decoder_path))                

    # pose_decoder.cuda()
    # pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            # 使用separate_resnet训练
            features = [pose_encoder(all_color_aug)]
    
            out_mu, out_var = pose_decoder(features)
            
            axisangle = out_mu[..., :3]
            translation = out_mu[..., 3:]

            # axisangle = reparameterize(axisangle_mean, axisangle_logvar)
            # translation = reparameterize(translation_mean, translation_logvar)
            
            # axisangle = axisangle_mean
            # translation = translation_mean
            
            # 使用posecnn训练
            # axisangle, translation = pose_decoder(all_color_aug)
            # print('axisangle:',axisangle)
            # print('translation:',translation)
            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    translations = np.array([matrix[:3, 3] for matrix in pred_poses])
    translations_x = np.cumsum(translations[:, 0])
    translations_y = np.cumsum(translations[:, 1])
    translations_z = np.cumsum(translations[:, 2])
    # print(translations_z)
    # gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_poses_path = os.path.join(splits_dir, opt.eval_split, "SyntheticColon_I_S1.txt")
    # gt_poses_path = os.path.join(splits_dir, opt.eval_split, "SyntheticColon_II_B1_new.txt")
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]


    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join('/well/rittscher/users/ycr745/monodepth2_revised/pose', "predict_poses_S1_separate_vae_test_1.npy")
    # save_path = os.path.join('/well/rittscher/users/ycr745/monodepth2_revised/pose', "predict_poses_B1_separate_orginal_noautomasking.npy")
    
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())

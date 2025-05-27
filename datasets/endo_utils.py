from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def make_endo_split(path):
    list_frames = []
    list_all = os.listdir(path)
    for file in list_all:
        if os.path.isdir(os.path.join(path,file)):
            list_frames.append(file)
    list_frames.sort(key=lambda x:int(x.split('_')[1][1:]))
    train_split = list_frames[0:int(len(list_frames)*0.8)]
    val_split = list_frames[int(len(list_frames)*0.8):]
    write_in_txt(path,train_split,is_train = True)
    write_in_txt(path,val_split,is_train = False)

def make_endo_split_all(path):
    # list_frames_train = ['Frames_S1','Frames_S2','Frames_S3','Frames_S6','Frames_S7',
    #                      'Frames_S8','Frames_S11','Frames_S12','Frames_S13',
    #                      'Frames_B1','Frames_B2','Frames_B3','Frames_B6','Frames_B7',
    #                      'Frames_B8','Frames_B11','Frames_B12','Frames_B13',
    #                      'Frames_S5','Frames_S10','Frames_S15','Frames_B5','Frames_B10','Frames_B15']
    
    list_frames_train = ['Frames_S1','Frames_S2','Frames_S3','Frames_S4','Frames_S5','Frames_S6','Frames_S7',
                         'Frames_S8','Frames_S9','Frames_S10','Frames_S11','Frames_S12','Frames_S13','Frames_S14','Frames_S15',
                         'Frames_B1','Frames_B2','Frames_B3','Frames_B4','Frames_B5','Frames_B6','Frames_B7',
                         'Frames_B8','Frames_B9','Frames_B10','Frames_B11','Frames_B12','Frames_B13','Frames_B14','Frames_B15',
                         'Frames_O1','Frames_O2','Frames_O3']
    list_frames_val = [ 'Frames_S4', 'Frames_S9','Frames_S14',
                       'Frames_B4', 'Frames_B9','Frames_B14']
    # list_all = os.listdir(path)
    # for file in list_all:
    #     if os.path.isdir(os.path.join(path,file)):
    #         list_frames.append(file)
    # list_frames.sort(key=lambda x:int(x.split('_')[1][1:]))
    # train_split = list_frames[0:int(len(list_frames)*0.8)]
    # val_split = list_frames[int(len(list_frames)*0.8):]
    write_in_txt_all(path,list_frames_train,is_train = True)
    write_in_txt_all(path,list_frames_val,is_train = False)

def write_in_txt(path,split,is_train = True):

    for name in split:
        frames_name = os.listdir(os.path.join(path,name))
        FrameBuffer_id = []
        for i in frames_name:
            if i.startswith('FrameBuffer'):
                FrameBuffer_id.append(i)
        FrameBuffer_id.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        if is_train:
            file_write_obj = open("/well/rittscher/users/ycr745/monodepth2/splits/endo/train_files.txt", 'a')
        else:
            file_write_obj = open("/well/rittscher/users/ycr745/monodepth2/splits/endo/val_files.txt", 'a')    
        for var in FrameBuffer_id:
            info = [name,' ',var.split('_')[-1].split('.')[0],' ','l']
            file_write_obj.writelines(info)
            file_write_obj.write('\n')
        file_write_obj.close()

def write_in_txt_all(path,split,is_train = True):

    for name in split:
        frames_name = os.listdir(os.path.join(path,name))
        FrameBuffer_id = []
        for i in frames_name:
            if i.startswith('FrameBuffer'):
                FrameBuffer_id.append(i)
        FrameBuffer_id.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        if is_train:
            file_write_obj = open("/well/rittscher/users/ycr745/monodepth2/splits/endo/train_files_all.txt", 'a')
        else:
            file_write_obj = open("/well/rittscher/users/ycr745/monodepth2/splits/endo/val_files_all.txt", 'a')    
        for var in FrameBuffer_id[1:-1]:
            info = [name,' ',var.split('_')[-1].split('.')[0],' ','l']
            file_write_obj.writelines(info)
            file_write_obj.write('\n')
        file_write_obj.close()

def make_endo_test(path):
    list_frames_test = ['Frames_S1']
    # list_frames = []
    # list_all = os.listdir(path)
    # for file in list_all:
    #     if os.path.isdir(os.path.join(path,file)):
    #         list_frames.append(file)
    # list_frames.sort(key=lambda x:int(x.split('_')[1][1:]))
    write_in_txt_test(path,list_frames_test)

def write_in_txt_test(path,split):

    for name in split:
        frames_name = os.listdir(os.path.join(path,name))
        FrameBuffer_id = []
        for i in frames_name:
            if i.startswith('FrameBuffer'):
                FrameBuffer_id.append(i)
        FrameBuffer_id.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        file_write_obj = open("/well/rittscher/users/ycr745/monodepth2/splits/endo_2/test_files_depth_S1.txt", 'a')
        for var in FrameBuffer_id:
            info = [name,' ',var.split('_')[-1].split('.')[0],' ','l']
            file_write_obj.writelines(info)
            file_write_obj.write('\n')
        file_write_obj.close()    

    



endo_path_test = '/well/rittscher/projects/3d_ziang/dataset/SyntheticColon_III'
endo_path_train = '/well/rittscher/projects/3d_ziang/dataset/SyntheticColon_I'
endo_path_all = '/well/rittscher/projects/3d_ziang/dataset/data_all'
make_endo_split_all(endo_path_all)
# make_endo_test(endo_path_all)
# make_endo_split(endo_path_train)
# make_endo_test(endo_path_test)

# python train.py --model_name endo --png --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_I --split endo --dataset endo --height 480 --width 480
# python train.py --model_name endo_mono+stereo --png --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_I --split endo --dataset endo --height 480 --width 480 --frame_ids 0 -1 1 --use_stereo
# python export_gt_depth.py --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_III --split endo
# python evaluate_depth.py --load_weights_folder /well/rittscher/users/ycr745/monodepth2/tmp/endo/models/weights_19/ --eval_mono --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_III --eval_split endo
# python evaluate_depth.py --load_weights_folder /well/rittscher/users/ycr745/monodepth2/tmp/endo/models/weights_19/ --eval_mono --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_II --eval_split endo_2
# python evaluate_pose.py --eval_split endo_2 --load_weights_folder /well/rittscher/users/ycr745/monodepth2/tmp/endo/models/weights_19/ --eval_mono --data_path /well/rittscher/projects/3d_ziang/dataset/SyntheticColon_II --eval_mono

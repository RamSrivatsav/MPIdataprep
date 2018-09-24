# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:09:22 2018

@author: vatsa
"""

from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc
TAG_FLOAT = 202021.25

class MPI_loader(object):
    def __init__(self, dataset_dir, split, cam_dir, img_height=256, img_width=256, seq_length=5):
#        dir_path = os.path.dirname(os.path.realpath(__file__))
#        static_frames_file = dir_path + '/static_frames.txt'
#        test_scene_file = dir_path + '/test_scenes_' + split + '.txt'
#        with open(test_scene_file, 'r') as f:
#            test_scenes = f.readlines()
#        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = dataset_dir
        self.cam_dir = cam_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
#        self.collect_static_frames(static_frames_file)
        self.collect_train_frames()

#    def collect_static_frames(self, static_frames_file):
#        with open(static_frames_file, 'r') as f:
#            frames = f.readlines()
#        self.static_frames = []
#        for fr in frames:
#            if fr == '\n':
#                continue
#            date, drive, frame_id = fr.split(' ')
#            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
#            for cid in self.cam_ids:
#                self.static_frames.append(drive + ' ' + cid + ' ' + curr_fid)
        
    def collect_train_frames(self):
        all_frames = []
        typelist=os.listdir(self.dataset_dir)
        for fold in typelist:
            data_set = os.listdir(self.dataset_dir+ '/' + fold)
            N = len(data_set)
            for n in data_set:
#                frame_id = '%.10d' % n
                all_frames.append(fold + ' ' + n)
                        

#        for s in self.static_frames:
#            try:
#                all_frames.remove(s)
#                # print('removed static frame from training: %s' % s)
#            except:
#                pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        cid, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_cid, _ = frames[min_src_idx].split(' ')
        max_src_cid, _ = frames[max_src_idx].split(' ')
        if cid == min_src_cid and cid == max_src_cid:
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_cid, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_cid, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw(tgt_cid, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_cid + '/'
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, cid, frame_id):
#        date = drive[:10]
        img_file = os.path.join(self.dataset_dir, cid, frame_id)
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics_raw(self, cid, frame_id):
#        date = drive[:10]
        calib_file = os.path.join(self.cam_dir, cid, 'frame_0001.cam')

        intrinsics = self.read_raw_calib_file(calib_file)
        return intrinsics

    def read_raw_calib_file(self,filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        file1=open(filepath,'r')
        check = np.fromfile(file1,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        M1 = np.fromfile(file1,dtype='float64',count=9).reshape((3,3))
        N1 = np.fromfile(file1,dtype='float64',count=12).reshape((3,4))
        return M1

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out
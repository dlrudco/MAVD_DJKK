import torch.utils.data as data
import numpy as np
import utils.utils as utils

CLASSES = ['A', 'B1', 'B2', 'B4', 'B5', 'B6', 'G']   # index 0 = normal


def parse_label(path):
    # '..._label_B2-G-0__3.npy' -> multi-hot over CLASSES ('0' is an empty slot, normal is '_label_A')
    tags = path.split('_label_')[-1].split('__')[0].split('-')
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for t in tags:
        if t != '0':
            y[CLASSES.index(t)] = 1.0
    return y


class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, return_name=False):
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
            self.flow_list_file = args.test_flow_list
            self.clip_list_file = getattr(args, 'test_clip_list', None)
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list
            self.flow_list_file = args.flow_list
            self.clip_list_file = getattr(args, 'clip_list', None)
        self.max_seqlen = args.max_seqlen
        self.transform = transform
        self.test_mode = test_mode
        self.return_name = return_name
        self.normal_flag = '_label_A'
        self.multi_class = getattr(args, 'mode', None) == 'vfa_cls'
        self.clip = self.multi_class and getattr(args, 'clip', 0) == 1
        if self.clip:
            self.clip_flip = args.clip_flip
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file, encoding='utf-8'))
        self.audio_list = list(open(self.audio_list_file, encoding='utf-8'))
        self.flow_list = list(open(self.flow_list_file, encoding='utf-8'))
        if self.clip:
            self.clip_list = list(open(self.clip_list_file, encoding='utf-8'))

    def load_clip(self, index, length):
        # CLIP row k = snippet k (frames 16k~16k+15) 의 첫 프레임 1장, 영상의 ~10% 는 rgb 보다 1 줄 더 김 (끝에 붙음)
        # -> rgb 길이로 자른 뒤 process_feat 에 넣어야 uniform_extract 가 rgb 와 같은 snippet 을 뽑음
        path = self.clip_list[index].strip('\n')
        if self.clip_flip and not self.test_mode and np.random.rand() < 0.5:
            head, crop = path.rsplit('__', 1)
            path = '{}__{}.npy'.format(head, int(crop[:-4]) + 5)   # crop k -> 좌우반전 crop k+5
        f_c = np.array(np.load(path), dtype=np.float32)              # 저장은 대부분 float16
        return f_c[:length]

    def video_labels(self):
        # (num_videos, 7) multi-hot; the audio list has one line per video (rgb/flow have 5 crops each)
        return np.stack([parse_label(l.strip('\n')) for l in self.audio_list])

    def __getitem__(self, index):
        if self.multi_class:
            label = parse_label(self.list[index].strip('\n'))
        elif self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0
        f_v = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        f_f = np.array(np.load(self.flow_list[index].strip('\n')), dtype=np.float32)
        f_a = np.array(np.load(self.audio_list[index//5].strip('\n')), dtype=np.float32)
        if self.clip:
            f_c = self.load_clip(index, len(f_v))   # --clip: 4번째 feature, rgb 길이에 맞춰 자름
        if self.transform is not None:
            f_v = self.transform(f_v)
            f_f = self.transform(f_f)
            f_a = self.transform(f_a)
            if self.clip:
                f_c = self.transform(f_c)
        if self.test_mode:
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return f_v, f_a, f_f, file_name
            if self.clip:
                return f_v, f_a, f_f, f_c
            return f_v, f_a, f_f
        else:
            f_v = utils.process_feat(f_v, self.max_seqlen, is_random=False)
            f_a = utils.process_feat(f_a, self.max_seqlen, is_random=False)
            f_f = utils.process_feat(f_f, self.max_seqlen, is_random=False)
            if self.clip:
                f_c = utils.process_feat(f_c, self.max_seqlen, is_random=False)
            if self.return_name == True:
                file_name = self.list[index].strip('\n').split('/')[-1][:-7]
                return f_v, f_a, f_f, file_name
            if self.clip:
                return f_v, f_a, f_f, f_c, label
            return f_v, f_a, f_f, label

    def __len__(self):
        return len(self.list)

    


import argparse
from random import seed
import os

def init_args(args):
    args.output_path = os.path.join(args.base_path, f'{args.mode}', 'outputs')
    args.model_path = os.path.join(args.base_path, f'{args.mode}', 'saved_models')
    args.log_path = os.path.join(args.base_path, f'{args.mode}', 'logs')
    args.root_dir = os.path.join(args.base_path, f'{args.mode}', 'root_dir')
    args.init_path = os.path.join(args.base_path, f'{args.mode}', 'init_models')
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args

descript = 'Pytorch Implementation of VAF'
parser = argparse.ArgumentParser(description = descript)

parser.add_argument('--model_file', type = str, default = "model_{}.pkl".format(seed), help = 'the path of pre-trained model file')

parser.add_argument('--base_path', type = str, default = 'outputs/', help = 'base path for saving outputs')
parser.add_argument('--wandb_entity', type = str, default = 'ActionTeam', help = 'wandb team/user to log to (the account has no default entity)')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rates')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--seed', type = int, default = 500, help = 'random seed (-1 for no manual seed)')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--num_steps', default=1000, help='number of epochs to train for')

parser.add_argument('--rgb-list', default='./list/video_train.list', help='list of rgb features')
parser.add_argument('--flow-list', default='./list/flow_train.list', help='list of flow features')
parser.add_argument('--audio-list', default='./list/audio_train.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='./list/video_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='./list/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='./list/audio_test.list', help='list of test audio features')

parser.add_argument('--max_seqlen', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--gt', default='./list/gt.npy', help='file of ground truth ')

parser.add_argument('--lamda1', type = float, default = 10.0)
parser.add_argument('--lamda2', type = float, default = 10.0)
parser.add_argument('--lamda3', type = float, default = 0.001)

parser.add_argument('--mode', type=str, choices=['vf','v','f', 'vf2', 'vf3', 'vfa', 'vfa_cls'])

# vfa_cls: vfa + 7-way class head (normal + 6 violence classes) on the fusion features
parser.add_argument('--lamda_cls', type = float, default = 1.0, help = 'weight of the class loss (CLASM)')
parser.add_argument('--cls_act', type = str, default = 'softmax', choices = ['softmax', 'sigmoid'], help = 'softmax: CE with multi-hot / #labels, sigmoid: per-class BCE with multi-hot')

# vfa / vfa_cls: cat 직전에 MBT 스타일 bottleneck fusion (model/bottleneck.py)
parser.add_argument('--fusion', type = str, default = 'none', choices = ['none', 'bottleneck'], help = 'none: original cat, bottleneck: bottleneck fusion before the cat')
parser.add_argument('--n_bottleneck', type = int, default = 4, help = 'bottleneck tokens per modality')
parser.add_argument('--bn_layers', type = int, default = 1, help = 'number of bottleneck exchanges: A -> (X -> A) x bn_layers')
parser.add_argument('--bn_dim', type = int, default = 64, help = 'common dim inside the fusion')
parser.add_argument('--bn_heads', type = int, default = 4, help = 'attention heads (dim_head = bn_dim // 2)')
parser.add_argument('--bn_residual', type = int, default = 1, choices = [0, 1], help = '1: out = in + fusion (zero-init), 0: out = fusion')
parser.add_argument('--bn_tsa', type = int, default = 1, choices = [0, 1], help = '1: time tokens self-attend over time (full), 0: time tokens see only themselves + bottleneck (no temporal SA)')

# vfa_cls: 4번째 modality 로 CLIP ViT-B/16 feature (XDData/VadCLIP, VadCLIP 공개 feature), clip crop k <-> rgb crop k
parser.add_argument('--clip', type = int, default = 0, choices = [0, 1], help = 'vfa_cls only: 1 adds the CLIP modality (c_net -> vc_net -> cat / bottleneck)')
parser.add_argument('--clip-list', default='./list/clip_train.list', help='list of CLIP features (same order as the rgb list)')
parser.add_argument('--test-clip-list', default='./list/clip_test.list', help='list of test CLIP features')
parser.add_argument('--clip_flip', type = int, default = 0, choices = [0, 1], help = 'train only: 1 swaps crop k for its flipped crop k+5 with prob 0.5')
parser.add_argument('--clip_detach', type = int, default = 1, choices = [0, 1], help = '1: c_net output detached before the fusion like the other modalities, 0: fusion / class gradients also train c_net')
parser.add_argument('--lamda_clip_mil', type = float, default = 1.0, help = 'weight of the CLIP unimodal binary MIL loss in U_MIL (v / a / f use 1, 0 = off)')
parser.add_argument('--lamda_clip_cls', type = float, default = 0.0, help = 'weight of a CLASM class loss on c_net itself (0 = off)')




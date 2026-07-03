import os
import torch
import numpy as np
import torch.utils.data as data
import config.options as options

from train_test.train import *
from train_test.test import *
from utils.utils import *
from losses.losses import * 
from model.unimodal import *
from model.multimodal import *
from model.projection import *
from dataset.dataset_loader import *

import wandb 

if __name__ == "__main__":

    args = options.parser.parse_args()
    args.save_model_path = f'saved_models_vfvanilla/seed_{args.seed}/'
    args = options.init_args(args)
    set_seed(args.seed)
    
    lamda1, lamda2, lamda3= args.lamda1, args.lamda2, args.lamda3

    train_loader = data.DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = data.DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128)
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64)
    merger = nn.Sequential(nn.Linear(128+64, 64), nn.ReLU(),
                                       nn.Linear(64, 32),
                                       nn.Linear(32, 1), nn.Sigmoid())
    
    optimizer = torch.optim.AdamW(list(v_net.parameters())+list(f_net.parameters())+list(merger.parameters()), 
                                 lr = args.lr, betas = (0.9, 0.999), weight_decay = 0.0005)

    criterion = AD_Loss_v_only()
    criterion_disl = None

    v_net = v_net.cuda()
    a_net = None
    f_net = f_net.cuda()
    va_net = None
    vf_net = None
    vaf_net = None
    merger = merger.cuda()

    best_ap = 0.0
    test_info = {"iteration": [], "m_ap":[]}

    gt = np.load(args.gt)
    step = 0
    # for step in tqdm(range(1, args.num_steps + 1), leave=True, desc="Training"):
    wandb.init(project="MAVD", name=f"vf_vanilla_seqlen20_seed_{args.seed}", config=args, reinit=True)
    for epoch in tqdm(range(10), leave=True, desc="Training Epochs"):
        for batch_data in tqdm(train_loader, leave=False, desc="Training Batches"):
            # if (step-1) % len(train_loader) == 0:
            #     train_loader_iter = iter(train_loader)
            step += 1
            loss_dict_list = train_vf_vanilla(v_net, f_net, merger,
                        batch_data, 
                        optimizer,
                        criterion, criterion_disl, step,
                        lamda1, lamda2, lamda3)
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            if step % 10 == 0:
                # print(f'Step: {step}, '
                # f'U_MIL_loss: {loss_dict_list["U_MIL_loss"]:.6f}, '
                # f'MA_loss: {loss_dict_list_disl["MA_loss"]:.6f}, '
                # f'M_MIL_loss: {loss_dict_list_disl["M_MIL_loss"]:.6f}, '
                # f'Triplet_loss: {loss_dict_list_disl["Triplet_loss"]:.6f}, '
                # f'LR: {current_lr:.6f} '
                # )
                wandb.log({
                    "U_MIL_loss": loss_dict_list["U_MIL_loss"],
                    "LR": current_lr,
                    "step": step,
                
                }, commit=False)
                test_vf_vanilla(v_net, f_net, merger,
                                test_loader, gt, 
                                test_info, step)
                # print(test_info)
                
                if test_info["m_ap"][-1] > best_ap:
                    best_ap = test_info["m_ap"][-1]
                    os.makedirs(os.path.join('vf_vanilla', args.save_model_path), exist_ok=True)
                    os.makedirs(os.path.join('vf_vanilla', args.output_path), exist_ok=True)
                    utils.save_best_record(test_info, 
                        os.path.join('vf_vanilla', args.output_path, "best_record_{}.txt".format(args.seed)))
                    torch.save(v_net.state_dict(), os.path.join('vf_vanilla', args.save_model_path, "v_model.pth"))
                    torch.save(f_net.state_dict(), os.path.join('vf_vanilla', args.save_model_path, "f_model.pth"))
                    torch.save(merger.state_dict(), os.path.join('vf_vanilla', args.save_model_path, "merger.pth"))
            
                wandb.log({
                    "mAP": test_info["m_ap"][-1],
                    'best_mAP': best_ap,
                    "step": step,
                }, commit=True)
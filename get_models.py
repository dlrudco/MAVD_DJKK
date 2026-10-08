import os
import time
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
from model.bottleneck import BottleneckFusion
from dataset.dataset_loader import *

import wandb 

# parser.add_argument('--output_path', type = str, default = 'outputs/')
# parser.add_argument('--root_dir', type = str, default = 'outputs/')
# parser.add_argument('--log_path', type = str, default = 'logs/')
# parser.add_argument('--model_path', type = str, default = 'saved_models/')
# parser.add_argument('--init_path', type = str, default = 'saved_models/init_models/')

if __name__ == "__main__":

    args = options.parser.parse_args()
    assert args.fusion == 'none' or args.mode in ['vfa', 'vfa_cls'], 'bottleneck fusion needs the 3 modalities (vfa / vfa_cls)'
    assert not args.clip or args.mode == 'vfa_cls', 'the CLIP modality is implemented for vfa_cls only'
    assert not args.clip or args.lamda_clip_mil > 0 or args.lamda_clip_cls > 0 or not args.clip_detach, 'c_net gets no gradient: set --lamda_clip_mil / --lamda_clip_cls > 0 or --clip_detach 0'
    # one folder per run so checkpoints from different runs never overwrite each other
    tags = ("sig_" if args.mode == 'vfa_cls' and args.cls_act == 'sigmoid' else "") + (f"bn{args.n_bottleneck}x{args.bn_layers}{'' if args.bn_residual else 'nores'}{'' if args.bn_tsa else 'notsa'}_" if args.fusion == 'bottleneck' else "") \
        + (f"clip128{'f' if args.clip_flip else ''}{'' if args.clip_detach else 'nodet'}{f'cm{args.lamda_clip_mil:g}' if args.lamda_clip_mil != 1 else ''}{f'cc{args.lamda_clip_cls:g}' if args.lamda_clip_cls > 0 else ''}_" if args.clip else "")
    args.run_name = f'seqlen{args.max_seqlen}_seed{args.seed}_{tags}{time.strftime("%Y%m%d-%H%M%S")}'
    args.save_model_path = os.path.join(args.base_path, f'{args.mode}', 'saved_models', args.run_name)
    args = options.init_args(args)
    set_seed(args.seed)

    
    lamda1, lamda2, lamda3= args.lamda1, args.lamda2, args.lamda3

    train_loader = data.DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = data.DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    if args.mode == 'vf3':
        v_net = Unimodal(input_size=2048, h_dim=128, feature_dim=128)
    else:
        v_net = Unimodal(input_size=1024, h_dim=128, feature_dim=128)
    a_net = Unimodal(input_size=128, h_dim=64, feature_dim=32)
    f_net = Unimodal(input_size=1024, h_dim=128, feature_dim=64)
    

    va_net = Projection(32, 32, 32)
    vf_net = Projection(64, 64, 64)
    

    # --clip (vfa_cls): CLIP ViT-B/16 512-d -> c_net -> vc_net, 다른 modality 와 같은 구조 (c_net 의 class head 는 --lamda_clip_cls > 0 일 때만)
    c_net, vc_net = None, None
    clip_dims = [128] if args.clip else []
    if args.clip:
        c_net = Unimodal(input_size=512, h_dim=128, feature_dim=128, num_classes=len(CLASSES) if args.lamda_clip_cls > 0 else 0)
        vc_net = Projection(128, 128, 128)

    if args.mode == 'vfa':
        vaf_net = Multimodal(input_size=128+32+64, h_dim=128, feature_dim=64)
    elif args.mode == 'vfa_cls':
        vaf_net = Multimodal(input_size=128+32+64+sum(clip_dims), h_dim=128, feature_dim=64, num_classes=len(CLASSES))
    else:
        vaf_net = Multimodal(input_size=128+64, h_dim=128, feature_dim=64)

    # v / a / f feature (128 / 32 / 64) 를 cat 직전에 bottleneck 으로 교류, 출력 shape 은 그대로
    bn_net = None
    if args.fusion == 'bottleneck':
        bn_net = BottleneckFusion([128, 32, 64] + clip_dims, dim=args.bn_dim, n_bottleneck=args.n_bottleneck,
                                  layers=args.bn_layers, heads=args.bn_heads, residual=bool(args.bn_residual),
                                  temporal_sa=bool(args.bn_tsa))


    optimizer = torch.optim.AdamW(list(v_net.parameters())+list(a_net.parameters())+list(f_net.parameters())+list(va_net.parameters())+list(vf_net.parameters())+list(vaf_net.parameters())+(list(bn_net.parameters()) if bn_net is not None else [])+(list(c_net.parameters())+list(vc_net.parameters()) if c_net is not None else []), 
                                 lr = args.lr, betas = (0.9, 0.999), weight_decay = 0.0005)

    if args.mode == 'vf':
        criterion = AD_Loss_vf()
    elif args.mode in ['v', 'f', 'vf2', 'vf3']:
        criterion = AD_Loss_single()
    elif args.mode in ['vfa', 'vfa_cls']:
        criterion = AD_Loss()

    if args.mode == 'vfa':
        criterion_disl = DISL_Loss_vfa()
    elif args.mode == 'vfa_cls':
        criterion_disl = DISL_Loss_vfa_cls(args.cls_act)
    else:
        criterion_disl = DISL_Loss_vf()

    v_net = v_net.cuda()
    a_net = a_net.cuda()
    f_net = f_net.cuda()
    va_net = va_net.cuda()
    vf_net = vf_net.cuda()
    vaf_net = vaf_net.cuda()
    if bn_net is not None:
        bn_net = bn_net.cuda()
    if c_net is not None:
        c_net = c_net.cuda()
        vc_net = vc_net.cuda()

    best_ap = 0.0
    best_cls_map = 0.0
    test_info = {"iteration": [], "m_ap":[]}

    gt = np.load(args.gt)
    if args.mode == 'vfa_cls':
        video_labels = test_loader.dataset.video_labels()
    step = 0
    # for step in tqdm(range(1, args.num_steps + 1), leave=True, desc="Training"):
    wandb.init(entity=args.wandb_entity, project="MAVD", name=f"{args.mode}_loss3_adamw_{args.run_name}", config=args, reinit=True)
    # test_vf(v_net, a_net, f_net, va_net, vf_net, vaf_net,
    #                             test_loader, gt, 
    #                             test_info, step)
    for epoch in tqdm(range(4), leave=True, desc="Training Epochs"):
        for batch_data in tqdm(train_loader, leave=False, desc="Training Batches"):
            # if (step-1) % len(train_loader) == 0:
            #     train_loader_iter = iter(train_loader)
            step += 1
            if args.mode == 'vf':
                loss_dict_list, loss_dict_list_disl = train_vf(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data, 
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3)
            elif args.mode == 'vfa':
                loss_dict_list, loss_dict_list_disl = train_vfa(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data,
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3, bn_net=bn_net)
            elif args.mode == 'vfa_cls':
                loss_dict_list, loss_dict_list_disl = train_vfa_cls(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data,
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3, args.lamda_cls, bn_net=bn_net,
                            c_net=c_net, vc_net=vc_net, clip_detach=bool(args.clip_detach), lamda_clip_mil=args.lamda_clip_mil, lamda_clip_cls=args.lamda_clip_cls)
            elif args.mode == 'vf2':
                loss_dict_list, loss_dict_list_disl = train_vf2(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data, 
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3)
            elif args.mode == 'vf3':
                loss_dict_list, loss_dict_list_disl = train_vf3(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data, 
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3)
            elif args.mode == 'v':
                loss_dict_list, loss_dict_list_disl = train_v(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data, 
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3)
            elif args.mode == 'f':
                loss_dict_list, loss_dict_list_disl = train_f(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                            batch_data, 
                            optimizer,
                            criterion, criterion_disl, step,
                            lamda1, lamda2, lamda3)
            else:
                raise ValueError("Invalid mode selected. Choose from 'vf', 'v', or 'f'.")
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
                if args.mode == 'vf':
                    wandb.log({
                        "U_MIL_loss": loss_dict_list["U_MIL_loss"],
                        "MA_loss": loss_dict_list_disl["MA_loss"],
                        "M_MIL_loss": loss_dict_list_disl["M_MIL_loss"],
                        "Triplet_loss": loss_dict_list_disl["Triplet_loss"],
                        "LR": current_lr,
                        "step": step,
                    
                    }, commit=False)
                elif args.mode == 'vfa':
                    wandb.log({
                        "U_MIL_loss": loss_dict_list["U_MIL_loss"],
                        "MA_loss": loss_dict_list_disl["MA_loss"],
                        "M_MIL_loss": loss_dict_list_disl["M_MIL_loss"],
                        "Triplet_loss": loss_dict_list_disl["Triplet_loss"],
                        "LR": current_lr,
                        "step": step,

                    }, commit=False)
                elif args.mode == 'vfa_cls':
                    wandb.log({
                        "U_MIL_loss": loss_dict_list["U_MIL_loss"],
                        "M_MIL_loss": loss_dict_list_disl["M_MIL_loss"],
                        "CLS_loss": loss_dict_list_disl["CLS_loss"],
                        **({"C_MIL_loss": loss_dict_list["C_MIL_loss"]} if c_net is not None else {}),
                        **({"CLIP_CLS_loss": loss_dict_list_disl["CLIP_CLS_loss"]} if args.lamda_clip_cls > 0 else {}),
                        "LR": current_lr,
                        "step": step,

                    }, commit=False)
                elif args.mode in ['v', 'f']:
                    wandb.log({
                        "U_MIL_loss": loss_dict_list["U_MIL_loss"],
                        "LR": current_lr,
                        "step": step,

                    }, commit=False)
                if bn_net is not None:
                    # 마지막 train batch 의 fusion 보정량 크기 ||proj_out(h)|| / ||in||, test 전에 찍어야 train 값
                    wandb.log({f"bn_delta_ratio_{m}": r.item() for m, r in zip(['v', 'a', 'f', 'c'], bn_net.delta_ratio)}, commit=False)
                if args.mode == 'vf':
                    test_vf(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, 
                                    test_info, step)
                elif args.mode == 'vfa':
                    test_vfa(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt,
                                    test_info, step, bn_net=bn_net)
                elif args.mode == 'vfa_cls':
                    test_vfa_cls(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, video_labels,
                                    test_info, step, args.cls_act, bn_net=bn_net, c_net=c_net, vc_net=vc_net)
                elif args.mode == 'vf2':
                    test_vf2(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, 
                                    test_info, step)
                elif args.mode == 'vf3':
                    test_vf3(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, 
                                    test_info, step)
                elif args.mode == 'v':
                    test_v(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, 
                                    test_info, step)
                elif args.mode == 'f':
                    test_f(v_net, a_net, f_net, va_net, vf_net, vaf_net,
                                    test_loader, gt, 
                                    test_info, step)
                # print(test_info)
                
                if test_info["m_ap"][-1] > best_ap:
                    best_ap = test_info["m_ap"][-1]
                    utils.save_best_record(test_info, 
                        os.path.join(args.save_model_path, "best_record_{}.txt".format(args.seed)))
                    torch.save(v_net.state_dict(), os.path.join(args.save_model_path, "v_model.pth"))
                    torch.save(f_net.state_dict(), os.path.join(args.save_model_path, "f_model.pth"))
                    torch.save(vf_net.state_dict(), os.path.join(args.save_model_path, "vf_model.pth"))
                    torch.save(vaf_net.state_dict(), os.path.join(args.save_model_path, "vaf_model.pth"))
                    if args.mode in ['vfa', 'vfa_cls']:
                        torch.save(a_net.state_dict(), os.path.join(args.save_model_path, "a_model.pth"))
                        torch.save(va_net.state_dict(), os.path.join(args.save_model_path, "va_model.pth"))
                    if bn_net is not None:
                        torch.save(bn_net.state_dict(), os.path.join(args.save_model_path, "bn_model.pth"))
                    if c_net is not None:
                        torch.save(c_net.state_dict(), os.path.join(args.save_model_path, "c_model.pth"))
                        torch.save(vc_net.state_dict(), os.path.join(args.save_model_path, "vc_model.pth"))
                    if args.mode == 'vfa_cls':
                        np.savez(os.path.join(args.save_model_path, "preds.npz"), **test_info["last_pred"])

                if args.mode == 'vfa_cls':
                    # second best checkpoint, selected by the 7-class video mAP, in its own sub-folder
                    if test_info["cls_map7"][-1] > best_cls_map:
                        best_cls_map = test_info["cls_map7"][-1]
                        cls_dir = os.path.join(args.save_model_path, "best_cls")
                        os.makedirs(cls_dir, exist_ok=True)
                        utils.save_cls_record(test_info, os.path.join(cls_dir, "best_record_{}.txt".format(args.seed)))
                        for name, net in [('v', v_net), ('a', a_net), ('f', f_net), ('va', va_net), ('vf', vf_net), ('vaf', vaf_net)] + ([('bn', bn_net)] if bn_net is not None else []) + ([('c', c_net), ('vc', vc_net)] if c_net is not None else []):
                            torch.save(net.state_dict(), os.path.join(cls_dir, f"{name}_model.pth"))
                        np.savez(os.path.join(cls_dir, "preds.npz"), **test_info["last_pred"])
                    wandb.log({
                        "cls_mAP7": test_info["cls_map7"][-1],
                        "best_cls_mAP7": best_cls_map,
                        **{f"AP_{c}": ap for c, ap in test_info["cls_ap"][-1].items()},
                    }, commit=False)

                wandb.log({
                    "mAP": test_info["m_ap"][-1],
                    'best_mAP': best_ap,
                    "step": step,
                }, commit=True)
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score
from tqdm import tqdm
from losses.losses import cls_topk_pool
from dataset.dataset_loader import CLASSES
import warnings
warnings.filterwarnings("ignore")

def test(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        a_net.eval()
        f_net.eval()
        va_net.eval()
        vf_net.eval()
        vaf_net.eval()
        
        m_pred = torch.zeros(0).cuda()

        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)

            mix_f = torch.cat([v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"])], dim=-1)
            m_out = vaf_net(mix_f)
            
            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))

        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def test_vf(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        f_net.eval()
        vf_net.eval()
        vaf_net.eval()
        
        m_pred = torch.zeros(0).cuda()
        # breakpoint()
        seq_len = []
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            v_data = f_v.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            f_res = f_net(f_data)

            mix_f = torch.cat([v_res["satt_f"], vf_net(f_res["satt_f"])], dim=-1)
            m_out = vaf_net(mix_f)
            
            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
            seq_len.append(f_data.shape[1])
        # breakpoint()
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def test_vfa(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch, bn_net=None):

    with torch.no_grad():

        v_net.eval()
        a_net.eval()
        f_net.eval()
        va_net.eval()
        vf_net.eval()
        vaf_net.eval()
        if bn_net is not None:
            bn_net.eval()

        m_pred = torch.zeros(0).cuda()
        seq_len = []
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):

            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)

            mix_f = [v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"])]
            if bn_net is not None:
                mix_f = bn_net(mix_f)
            m_out = vaf_net(torch.cat(mix_f, dim=-1))

            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
            seq_len.append(f_data.shape[1])
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def video_cls_metrics(video_probs, video_labels):
    # video_probs, video_labels: (num_videos, 7), column 0 = normal
    # cls_map7 = macro AP over all 7 classes on all videos (per-class APs kept for analysis)
    ap_all = [average_precision_score(video_labels[:, c], video_probs[:, c]) for c in range(len(CLASSES))]
    return {"cls_map7": np.mean(ap_all),
            "cls_ap": dict(zip(CLASSES, ap_all))}

def test_vfa_cls(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, video_labels, test_info, epoch, cls_act='softmax', bn_net=None,
                 c_net=None, vc_net=None):
    # test_vfa + video-level class metrics; one batch = the 5 crops of one video
    act = torch.sigmoid if cls_act == 'sigmoid' else (lambda x: F.softmax(x, dim=-1))

    with torch.no_grad():

        v_net.eval()
        a_net.eval()
        f_net.eval()
        va_net.eval()
        vf_net.eval()
        vaf_net.eval()
        if bn_net is not None:
            bn_net.eval()
        if c_net is not None:
            c_net.eval()
            vc_net.eval()

        m_pred = torch.zeros(0).cuda()
        snippet_probs, video_probs, seq_len = [], [], []
        for i, data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            if c_net is not None:
                f_v, f_a, f_f, f_c = data
            else:
                f_v, f_a, f_f = data

            v_data = f_v.cuda()
            a_data = f_a.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            a_res = a_net(a_data)
            f_res = f_net(f_data)

            mix_f = [v_res["satt_f"], va_net(a_res["satt_f"]), vf_net(f_res["satt_f"])]
            if c_net is not None:
                c_data = f_c.cuda()
                c_res = c_net(c_data)
                mix_f.append(vc_net(c_res["satt_f"]))
            if bn_net is not None:
                mix_f = bn_net(mix_f)
            m_out = vaf_net(torch.cat(mix_f, dim=-1))

            m_pred = torch.cat((m_pred, torch.mean(m_out["output"], 0)))
            # class: same pooling as training (per-class top-k -> softmax / sigmoid), then average over crops
            cls_logits = m_out["cls_logits"]                              # (5, T, 7)
            snippet_probs.append(act(cls_logits).mean(0))                 # (T, 7)
            video_probs.append(act(cls_topk_pool(cls_logits)).mean(0))    # (7,)
            seq_len.append(f_data.shape[1])
        m_pred = m_pred.cpu().numpy()
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        video_probs = torch.stack(video_probs).cpu().numpy()
        cls_res = video_cls_metrics(video_probs, video_labels)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)
        for k, v in cls_res.items():
            test_info.setdefault(k, []).append(v)
        # predictions of this test pass (overwritten each time), saved next to the best checkpoints
        test_info["last_pred"] = {"binary": m_pred,                                            # (N_snippets,)
                                  "snippet_probs": torch.cat(snippet_probs).cpu().numpy(),     # (N_snippets, 7)
                                  "video_probs": video_probs,                                  # (N_videos, 7)
                                  "video_labels": video_labels,                                # (N_videos, 7)
                                  "seq_len": np.array(seq_len)}                                # (N_videos,)

def test_vf2(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        f_net.eval()
        vf_net.eval()
        
        m_pred = torch.zeros(0).cuda()
        # breakpoint()
        seq_len = []
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            v_data = f_v.cuda()
            f_data = f_f.cuda()

            v_res = v_net(v_data)
            f_res = f_net(f_data)
            
            m_out = {'output': (v_res['output'] + f_res['output']) / 2}
            
            m_out = torch.mean(m_out["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
            seq_len.append(f_data.shape[1])
        # breakpoint()
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def test_vf3(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        
        m_pred = torch.zeros(0).cuda()
        # breakpoint()
        seq_len = []
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            v_data = f_v.cuda()
            f_data = f_f.cuda()

            v_data = torch.cat([v_data, f_data], dim=-1)
            v_res = v_net(v_data)
            
            m_out = torch.mean(v_res["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
            seq_len.append(f_data.shape[1])
        # breakpoint()
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def test_v(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        v_net.eval()
        
        m_pred = torch.zeros(0).cuda()
        # breakpoint()
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            v_data = f_v.cuda()
            v_res = v_net(v_data)
            
            m_out = torch.mean(v_res["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
        # breakpoint()
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)

def test_f(v_net, a_net, f_net, va_net, vf_net, vaf_net, test_loader, gt, test_info, epoch):
    
    with torch.no_grad():

        f_net.eval()
        
        m_pred = torch.zeros(0).cuda()
        # breakpoint()
        for i, (f_v, f_a, f_f) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            
            f_data = f_f.cuda()
            f_res = f_net(f_data)
            m_out = torch.mean(f_res["output"], 0)
            m_pred = torch.cat((m_pred, m_out))
        # breakpoint()
        m_pred = list(m_pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(m_pred, 16))
        m_ap = auc(recall, precision)

        test_info["iteration"].append(epoch)
        test_info["m_ap"].append(m_ap)
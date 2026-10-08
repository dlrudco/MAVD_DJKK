import torch
from tqdm import tqdm

def train(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()
        a_net.train()
        f_net.train()
        va_net.train()
        vf_net.train()
        vaf_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_a = f_a[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        a_data = f_a.cuda()
        f_data = f_f.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)
        a_predict = a_net(a_data, seq_len)
        f_predict = f_net(f_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, a_predict, f_predict, label)

        v_input = v_predict["satt_f"].detach().clone()
        a_input = a_predict["satt_f"].detach().clone()
        f_input = f_predict["satt_f"].detach().clone()
        a_input = va_net(a_input)
        f_input = vf_net(f_input)

        vaf_input = torch.cat([v_input, a_input, f_input], dim=-1)

        va_output = a_net(a_input, seq_len, em_flag=False)
        vf_output = f_net(f_input, seq_len, em_flag=False)
        vaf_output = vaf_net(vaf_input, seq_len)

        total_loss_disl, loss_dict_list_disl = criterion_disl(v_predict, va_output, vf_output, vaf_output, label, seq_len.cuda(), lamda1, lamda2, lamda3)

        total_loss = total_loss + total_loss_disl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, loss_dict_list_disl
    
def train_vf(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()
        f_net.train()
        vf_net.train()
        vaf_net.train()
        # breakpoint()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        f_data = f_f.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)
        f_predict = f_net(f_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, f_predict, label)

        v_input = v_predict["satt_f"].detach().clone()
        f_input = f_predict["satt_f"].detach().clone()
        f_input = vf_net(f_input)

        vaf_input = torch.cat([v_input, f_input], dim=-1)

        vf_output = f_net(f_input, seq_len, em_flag=False)
        vaf_output = vaf_net(vaf_input, seq_len)

        total_loss_disl, loss_dict_list_disl = criterion_disl(v_predict, vf_output, vaf_output, label, seq_len.cuda(), lamda1, lamda2, lamda3)

        total_loss = total_loss + total_loss_disl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, loss_dict_list_disl

def train_vfa(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3, bn_net=None):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()
        a_net.train()
        f_net.train()
        va_net.train()
        vf_net.train()
        vaf_net.train()
        if bn_net is not None:
            bn_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_a = f_a[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        a_data = f_a.cuda()
        f_data = f_f.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)
        a_predict = a_net(a_data, seq_len)
        f_predict = f_net(f_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, a_predict, f_predict, label)

        v_input = v_predict["satt_f"].detach().clone()
        a_input = a_predict["satt_f"].detach().clone()
        f_input = f_predict["satt_f"].detach().clone()
        a_input = va_net(a_input)
        f_input = vf_net(f_input)

        # bottleneck fusion 은 cat 경로에만, va_output / vf_output (alignment) 은 fusion 이전 feature 그대로
        vaf_feats = [v_input, a_input, f_input]
        if bn_net is not None:
            vaf_feats = bn_net(vaf_feats, seq_len)
        vaf_input = torch.cat(vaf_feats, dim=-1)

        va_output = a_net(a_input, seq_len, em_flag=False)
        vf_output = f_net(f_input, seq_len, em_flag=False)
        vaf_output = vaf_net(vaf_input, seq_len)

        total_loss_disl, loss_dict_list_disl = criterion_disl(v_predict, va_output, vf_output, vaf_output, label, seq_len.cuda(), lamda1, lamda2, lamda3)

        total_loss = total_loss + total_loss_disl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return loss_dict_list, loss_dict_list_disl

def train_vfa_cls(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3, lamda_cls, bn_net=None,
                  c_net=None, vc_net=None, clip_detach=True, lamda_clip_mil=1.0, lamda_clip_cls=0.0):
    # same as train_vfa, but label is (B, 7) multi-hot: binary losses use 1 - normal, the class head uses all 7
    # c_net / vc_net (--clip): CLIP 을 4번째 modality 로, unimodal MIL + projection + cat / bottleneck 은 다른 modality 와 동일

    with torch.set_grad_enabled(True):

        if c_net is not None:
            f_v, f_a, f_f, f_c, label  = data
        else:
            f_v, f_a, f_f, label  = data
        v_net.train()
        a_net.train()
        f_net.train()
        va_net.train()
        vf_net.train()
        vaf_net.train()
        if bn_net is not None:
            bn_net.train()
        if c_net is not None:
            c_net.train()
            vc_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_a = f_a[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]
        if c_net is not None:
            f_c = f_c[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        a_data = f_a.cuda()
        f_data = f_f.cuda()
        label = label.cuda()
        bin_label = 1 - label[:, 0]

        v_predict = v_net(v_data, seq_len)
        a_predict = a_net(a_data, seq_len)
        f_predict = f_net(f_data, seq_len)
        c_predict = None
        if c_net is not None:
            c_data = f_c.cuda()
            c_predict = c_net(c_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, a_predict, f_predict, bin_label, c_predict, lamda_clip_mil)

        v_input = v_predict["satt_f"].detach().clone()
        a_input = a_predict["satt_f"].detach().clone()
        f_input = f_predict["satt_f"].detach().clone()
        a_input = va_net(a_input)
        f_input = vf_net(f_input)

        # bottleneck fusion 은 cat 경로에만, va_output / vf_output (alignment) 은 fusion 이전 feature 그대로
        vaf_feats = [v_input, a_input, f_input]
        if c_net is not None:
            # clip_detach=False 면 fusion / class loss gradient 가 c_net 까지 감
            c_input = c_predict["satt_f"].detach().clone() if clip_detach else c_predict["satt_f"]
            vaf_feats.append(vc_net(c_input))
        if bn_net is not None:
            vaf_feats = bn_net(vaf_feats, seq_len)
        vaf_input = torch.cat(vaf_feats, dim=-1)

        va_output = a_net(a_input, seq_len, em_flag=False)
        vf_output = f_net(f_input, seq_len, em_flag=False)
        vaf_output = vaf_net(vaf_input, seq_len)

        total_loss_disl, loss_dict_list_disl = criterion_disl(v_predict, va_output, vf_output, vaf_output, label, seq_len.cuda(), lamda1, lamda2, lamda3, lamda_cls,
                                                             c_predict, lamda_clip_cls)

        total_loss = total_loss + total_loss_disl

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return loss_dict_list, loss_dict_list_disl

def train_vf2(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()
        f_net.train()
        # breakpoint()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        f_data = f_f.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)
        f_predict = f_net(f_data, seq_len)

        fused = {'output': (v_predict['output'] + f_predict['output']) / 2}

        total_loss, loss_dict_list = criterion(fused, label)


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, None

def train_vf3(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()
        # breakpoint()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]
        f_f = f_f[:, :torch.max(seq_len), :]

        v_data = f_v.cuda()
        f_data = f_f.cuda()

        v_data = torch.cat([v_data, f_data], dim=-1)
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)


        total_loss, loss_dict_list = criterion(v_predict, label)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, None
        
def train_v(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        v_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)

        f_v = f_v[:, :torch.max(seq_len), :]


        v_data = f_v.cuda()
        label = label.cuda()

        v_predict = v_net(v_data, seq_len)

        total_loss, loss_dict_list = criterion(v_predict, label)

        total_loss = total_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, None
    
def train_f(v_net, a_net, f_net, va_net, vf_net, vaf_net, data, optimizer, criterion, criterion_disl, index, lamda1, lamda2, lamda3):

    with torch.set_grad_enabled(True):

        # f_v, f_a, f_f, label  = next(dataloader)
        f_v, f_a, f_f, label  = data
        f_net.train()

        seq_len = torch.sum(torch.max(torch.abs(f_f), dim=2)[0] > 0, 1)
        f_f = f_f[:, :torch.max(seq_len), :]

        f_data = f_f.cuda()
        label = label.cuda()

        f_predict = f_net(f_data, seq_len)

        total_loss, loss_dict_list = criterion(f_predict, label)


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return loss_dict_list, None
    
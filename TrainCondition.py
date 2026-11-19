import torch
import torch.nn.functional as F
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from models.InduxTFS_Diff import InduxTFS_Diff
from models.Diff_MTS import Diff_MTS
from models.MetaIndux_TS import MetaIndux_TS
from models.DiT import DiT
from models.SSSD import SSSD
from models.DiffWave import DiffWave
from models.WaveStitch import WaveStitch
from models.TimeGAN import TimeGAN

import data.CMAPSSDataset as CMAPSSDataset
import wandb
from GaussianDiffusion import GaussianDiffusion1D_cls_free, GradualWarmupScheduler


def build_model(args, device):
    if args.model_name == 'InduxTFS_Diff':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='lta+td+fre_parallel_adaptive',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'InduxTFS_NoTSME':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='ablation_no_tsme',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'InduxTFS_FixedFusion':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='ablation_fixed_fusion',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'InduxTFS_NoLTA':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='ablation_no_lta',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'InduxTFS_NoFreq':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='ablation_no_fre',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'InduxTFS_NoWavelet':
        net_model = InduxTFS_Diff(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.2,
                                  channels=args.input_size, arch='ablation_no_wavelet',
                                  input_length=args.window_size).to(device)
    if args.model_name == 'Diff_MTS':
        net_model = Diff_MTS(dim=32, dim_mults=(1, 2, 2), cond_drop_prob=0.5,
                             channels=args.input_size).to(device)
    if args.model_name == 'MetaIndux_TS':
        net_model = MetaIndux_TS(dim=32, dim_mults=(1, 2), cond_drop_prob=0.2,
                                 channels=args.input_size, length=args.window_size).to(device)
    if args.model_name == 'DiT':
        net_model = DiT(input_size=args.input_size, hidden_size=128, num_heads=1).to(device)
    if args.model_name == 'DiffWave':
        net_model = DiffWave(seq_length=args.window_size)
    if args.model_name == 'SSSD':
        net_model = SSSD(seq_length=args.window_size, s4_lmax=args.window_size)
    if args.model_name == 'WaveStitch':
        net_model = WaveStitch(
            in_channels=args.input_size + 1, res_channels=64, skip_channels=64,
            out_channels=args.input_size, num_res_layers=4,
            diffusion_step_embed_dim_in=32, diffusion_step_embed_dim_mid=64,
            diffusion_step_embed_dim_out=64,
            s4_lmax=args.window_size, s4_d_state=64, s4_dropout=0.0,
            s4_bidirectional=True, s4_layernorm=True).to(device)
    if args.model_name == 'TimeGAN':
        net_model = TimeGAN(seq_length=args.window_size, in_dim=args.input_size).to(device)
    return net_model


def train(args, train_data, train_label):
    device = args.device
    best_loss = 9999

    train_dataset = TensorDataset(train_data.permute(0, 2, 1).to(device), train_label.to(device))
    dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

    net_model = build_model(args, device)

    if args.model_name == 'TimeGAN':
        x = train_data.permute(0, 2, 1).to(device)
        net_model.fit(x, args.epoch)
        torch.save(net_model.state_dict(), args.model_path)
        return

    if args.model_name == 'WaveStitch':
        lr = 1e-4
    else:
        lr = args.lr
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=lr, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier,
                                             warm_epoch=args.epoch // 10 + 1, after_scheduler=cosineScheduler)
    print(args.T)
    if args.model_name == 'WaveStitch':
        trainer = None
    else:
        trainer = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, channels=args.input_size, timesteps=args.T, objective='pred_noise').to(device)

    if args.model_name == 'WaveStitch':
        _betas = np.linspace(1e-4, 0.02, args.T).reshape(-1)
        _alphas = 1 - _betas
        _alpha_bars = torch.from_numpy(np.cumprod(_alphas)).float().to(device)

    for e in range(args.epoch):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            loss_list = []
            for images, labels in tqdmDataLoader:
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)
                labels = labels.to(device)
                if args.model_name == 'WaveStitch':
                    x_lc = x_0.permute(0, 2, 1)
                    rul_2d = (labels / 125.0).expand(b, args.window_size)
                    xw = torch.cat([x_lc, rul_2d.unsqueeze(-1)], dim=-1)
                    t = torch.randint(0, args.T, (b,), device=device)
                    noise = torch.randn_like(xw)
                    ab = _alpha_bars[t].view(b, 1, 1)
                    xw_noised = ab.sqrt() * xw + (1 - ab).sqrt() * noise
                    xw_noised[:, :, -1] = rul_2d
                    pred = net_model(xw_noised, t.view(b, 1))
                    target_noise = noise[:, :, :14].permute(0, 2, 1)
                    loss = F.mse_loss(pred, target_noise)
                else:
                    if np.random.rand() < 0.1:
                        labels = torch.zeros_like(labels).to(device)
                    loss = trainer(x_0, classes=labels).sum()
                loss_list.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), args.grad_clip)
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": sum(loss_list) / len(loss_list),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        current_loss = sum(loss_list) / len(loss_list)
        wandb.log({"Diffusion_Loss": current_loss})
        if e > 5 and current_loss < best_loss:
            torch.save(net_model.state_dict(), args.model_path)
            print('*******imporove!!!********')


def sample(args, train_label=None):
    if train_label == None:
        if args.dataset == 'FEMTO':
            from data.FEMTODataset import FEMTODataset
            datasets = FEMTODataset(sequence_length=args.window_size)
            train_data = datasets.get_train_data()
            train_label = datasets.get_label_slice(train_data)
        else:
            datasets = CMAPSSDataset.CMAPSSDataset(fd_number=args.dataset, sequence_length=args.window_size, deleted_engine=[1000])
            train_data = datasets.get_train_data()
            train_label = datasets.get_label_slice(train_data)
    device = args.device
    with torch.no_grad():
        net_model = build_model(args, device)
        ckpt = torch.load(args.model_path)
        net_model.load_state_dict(ckpt)
        print("model load weight done.")
        net_model.eval()
        if args.model_name == 'TimeGAN':
            sampledata = net_model.generate(train_label.shape[0])
        elif args.model_name == 'WaveStitch':
            _betas = torch.from_numpy(np.linspace(1e-4, 0.02, args.T)).float().to(device)
            _alphas = 1 - _betas
            _alpha_bars = torch.cumprod(_alphas, dim=0)
            total = train_label.shape[0]
            chunks = []
            start_time = time.time()
            bs = 1000
            for i in range(0, total, bs):
                lab = train_label[i:i + bs].to(device)
                b = lab.shape[0]
                rul_2d = (lab / 125.0).expand(b, args.window_size)
                x = torch.randn(b, args.window_size, args.input_size + 1, device=device)
                x[:, :, -1] = rul_2d
                for step in range(args.T - 1, -1, -1):
                    t = torch.full((b, 1), step, device=device, dtype=torch.long)
                    eps_pred = net_model(x, t).permute(0, 2, 1)
                    alpha_bar_t = _alpha_bars[step]
                    alpha_bar_t_1 = _alpha_bars[step - 1] if step > 0 else torch.ones((), device=device)
                    alpha_t = _alphas[step]
                    beta_t = _betas[step]
                    x[:, :, :14] = (x[:, :, :14] - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)
                    if step > 0:
                        sigma = torch.sqrt(beta_t * (1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.randn_like(x[:, :, :14])
                        x[:, :, :14] = x[:, :, :14] + sigma
                    x[:, :, -1] = rul_2d
                chunks.append(x[:, :, :14].cpu())
            sampledata = torch.cat(chunks, dim=0)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"经过的时间: {elapsed_time} 秒")
        else:
            diffusion = GaussianDiffusion1D_cls_free(net_model, seq_length=args.window_size, channels=args.input_size, timesteps=args.T, objective='pred_noise').to(device)
            start_time = time.time()
            sampledata = diffusion.sample(classes=train_label.to(device)).permute(0, 2, 1)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"经过的时间: {elapsed_time} 秒")

        sampledata = sampledata.cpu().numpy()
        np.savez(args.syndata_path, data=sampledata, label=train_label.cpu().numpy())
    return sampledata

"""PyTorch implementation of TimeGAN.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layer):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=embed_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.tanh = nn.Tanh()
        self.apply(_weights_init)

    def forward(self, input, tanh=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if tanh:
            H = self.tanh(H)
        return H


class Recovery(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layer):
        super(Recovery, self).__init__()
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=in_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(in_dim, in_dim)
        self.tanh = nn.Tanh()
        self.apply(_weights_init)

    def forward(self, input, tanh=False):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if tanh:
            X_tilde = self.tanh(X_tilde)
        return X_tilde


class Generator(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layer):
        super(Generator, self).__init__()
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.tanh = nn.Tanh()
        self.apply(_weights_init)

    def forward(self, input, tanh=True):
        g_outputs, _ = self.rnn(input)
        E = self.fc(g_outputs)
        if tanh:
            E = self.tanh(E)
        return E


class Supervisor(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layer):
        super(Supervisor, self).__init__()
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.tanh = nn.Tanh()
        self.apply(_weights_init)

    def forward(self, input, tanh=True):
        s_outputs, _ = self.rnn(input)
        S = self.fc(s_outputs)
        if tanh:
            S = self.tanh(S)
        return S


class Discriminator(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layer):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        d_outputs, _ = self.rnn(input)
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat


class TimeGAN(nn.Module):
    def __init__(self, seq_length, in_dim, embed_dim=None, num_layer=3, lr=1e-3, beta1=0.9, batch_size=256):
        super(TimeGAN, self).__init__()
        self.seq_length = seq_length
        self.in_dim = in_dim
        self.embed_dim = embed_dim if embed_dim is not None else max(in_dim - 1, 1)
        self.num_layer = num_layer
        self.batch_size = batch_size

        self.nete = Encoder(in_dim, self.embed_dim, num_layer)
        self.netr = Recovery(in_dim, self.embed_dim, num_layer)
        self.netg = Generator(in_dim, self.embed_dim, num_layer)
        self.netd = Discriminator(in_dim, self.embed_dim, num_layer)
        self.nets = Supervisor(in_dim, self.embed_dim, num_layer)

        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=lr, betas=(beta1, 0.999))

    def _make_loader(self, x):
        dataset = TensorDataset(x)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def fit(self, x, epochs):
        device = next(self.parameters()).device
        loader = self._make_loader(x)

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                embed = self.nete(batch)
                recovered = self.netr(embed)
                self.optimizer_e.zero_grad()
                self.optimizer_r.zero_grad()
                loss = self.l_mse(recovered, batch)
                loss.backward(retain_graph=True)
                self.optimizer_e.step()
                self.optimizer_r.step()
                total_loss += loss.detach().cpu().numpy()
            print(f'TimeGAN AUTOENCODER EPOCH {epoch}, LOSS: {total_loss}')

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                embed = self.nete(batch)
                out = self.nets(embed)
                self.optimizer_s.zero_grad()
                loss = self.l_mse(out[:, :-1, :], embed[:, 1:, :])
                loss.backward(retain_graph=True)
                self.optimizer_s.step()
                total_loss += loss.detach().cpu().numpy()
            print(f'TimeGAN SUPERVISOR EPOCH {epoch}, LOSS: {total_loss}')

        for epoch in range(epochs):
            total_loss_g, total_loss_er, total_loss_d = 0.0, 0.0, 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                z = torch.normal(0, 1, (batch.shape[0], batch.shape[1], self.embed_dim), device=device)
                out_e = self.nete(batch)
                out_se = self.nets(out_e)
                out_g = self.netg(z)
                out_sg = self.nets(out_g)
                out_rsg = self.netr(out_sg)
                y_fake = self.netd(out_sg)
                y_fake_e = self.netd(out_g)
                self.optimizer_g.zero_grad()
                self.optimizer_s.zero_grad()
                err_g_U = self.l_bce(y_fake, torch.ones_like(y_fake))
                err_g_U_e = self.l_bce(y_fake_e, torch.ones_like(y_fake_e))
                err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(out_rsg, [0])[1] + 1e-6) - torch.sqrt(torch.std(batch, [0])[1] + 1e-6)))
                err_g_V2 = torch.mean(torch.abs((torch.mean(out_rsg, [0])[0]) - (torch.mean(batch, [0])[0])))
                err_s = self.l_mse(out_se[:, :-1, :], out_e[:, 1:, :])
                loss_g = err_g_U + err_g_U_e * 1 + err_g_V1 * 100 + err_g_V2 * 100 + torch.sqrt(err_s)
                loss_g.backward(retain_graph=True)
                total_loss_g += loss_g.detach().cpu().numpy()
                self.optimizer_s.step()
                self.optimizer_g.step()

                out_er = self.netr(out_e)
                out_se = self.nets(out_e)
                self.optimizer_e.zero_grad()
                self.optimizer_r.zero_grad()
                err_er_ = self.l_mse(out_er, batch)
                err_s = self.l_mse(out_se[:, :-1, :], out_e[:, 1:, :])
                loss_er = 10 * torch.sqrt(err_er_) + 0.1 * err_s
                loss_er.backward(retain_graph=True)
                self.optimizer_e.step()
                self.optimizer_r.step()
                total_loss_er += loss_er.detach().cpu().numpy()

                out_e = self.nete(batch)
                z = torch.normal(0, 1, (batch.shape[0], batch.shape[1], self.embed_dim), device=device)
                out_g = self.netg(z)
                out_sg = self.nets(out_g)
                y_real = self.netd(out_e)
                y_fake = self.netd(out_sg)
                y_fake_e = self.netd(out_g)
                err_d_real = self.l_bce(y_real, torch.ones_like(y_real))
                err_d_fake = self.l_bce(y_fake, torch.zeros_like(y_fake))
                err_d_fake_e = self.l_bce(y_fake_e, torch.zeros_like(y_fake_e))
                loss_d = err_d_real + err_d_fake + err_d_fake_e * 1
                self.optimizer_d.zero_grad()
                if loss_d > 0.15:
                    loss_d.backward(retain_graph=True)
                self.optimizer_d.step()
                total_loss_d = loss_d.detach().cpu().numpy()

            print(f'TimeGAN JOINT EPOCH {epoch}, LOSS_G: {total_loss_g}, LOSS_ER: {total_loss_er}, LOSS_D: {total_loss_d}')

    @torch.no_grad()
    def generate(self, n_samples):
        device = next(self.parameters()).device
        z = torch.normal(0, 1, (n_samples, self.seq_length, self.embed_dim), device=device)
        out_g = self.netg(z)
        out_r = self.netr(out_g)
        return out_r

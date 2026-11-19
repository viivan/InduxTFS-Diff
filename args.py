import argparse

import torch

arg_parser = argparse.ArgumentParser(description='Industrial time-series generation')

arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--model_name', type=str, default='InduxTFS_Diff')
arch_group.add_argument('--embedding', type=int, default=48)
arch_group.add_argument('--hidden', type=int, default=64)
arch_group.add_argument('--num_layers', type=int, default=1)

optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--epoch', default=50, type=int, metavar='N')
optim_group.add_argument('--eva_epoch', default=35, type=int, metavar='N')
optim_group.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
optim_group.add_argument('--optimizer', default='adam', choices=['sgd', 'rmsprop', 'adam'], metavar='N')
optim_group.add_argument('--lr', '--learning_rate', default=2e-3, type=float, metavar='LR')
optim_group.add_argument('--lr_type', default='multistep', choices=['multistep', 'cosine', 'warmup'], metavar='LR')
optim_group.add_argument('--grad_clip', default=1., type=float)
optim_group.add_argument('--seed', default=3, type=int)
optim_group.add_argument('--multiplier', default=2.5, type=float)
optim_group.add_argument('--schedule_name', default='linear', type=str)
optim_group.add_argument('--sample_type', default='ddpm', choices=['ddim', 'ddpm'], type=str)

arg_parser.add_argument('--input_size', default=14, type=int)
arg_parser.add_argument('--window_size', default=48, type=int)
arg_parser.add_argument('--dataset', default='FD001', type=str)
arg_parser.add_argument('--state', default='train', type=str)
arg_parser.add_argument('--T', default=500, type=int)
arg_parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str)
arg_parser.add_argument('--model_path', default='./outputs/temp.pth', type=str)
arg_parser.add_argument('--save_path', default='./outputs/lstm_temp.pth', type=str)
arg_parser.add_argument('--syndata_path', default='./outputs/syn_data/temp.npz', type=str)

args = arg_parser.parse_args()

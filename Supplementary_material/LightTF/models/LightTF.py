import torch
import torch.nn as nn
from torch.nn import init
import math
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # Get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.patch_size
        self.cut_freq = configs.cut_freq
        self.down_sampling = configs.M
        self.num_sampling = self.period_len // self.down_sampling
        self.group = 1
        self.in_group_freq = self.cut_freq // self.group
        self.individual = configs.individual
        self.flinear_individual = False if self.individual == 0 else True
        self.flinear_sparse_num = configs.K
        self.in_sparse_freq = self.cut_freq // self.flinear_sparse_num
        self.linear_individual = False

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        if self.linear_individual:
            self.linear_weight = nn.Parameter(torch.randn(self.enc_in, self.group, self.seg_num_y, self.seg_num_x, dtype=torch.cfloat))
        else:
            self.linear_weight = nn.Parameter(torch.randn(self.group, self.seg_num_y, self.seg_num_x, dtype=torch.cfloat))
            init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))

        if self.flinear_individual:
            self.flinear_weight = nn.Parameter(torch.randn(self.enc_in, self.flinear_sparse_num, self.in_sparse_freq, self.in_sparse_freq, dtype=torch.cfloat))
        else:
            self.flinear_weight = nn.Parameter(torch.randn(self.flinear_sparse_num, self.in_sparse_freq, self.in_sparse_freq, dtype=torch.cfloat), requires_grad=True)
            init.kaiming_uniform_(self.flinear_weight, a=math.sqrt(5))

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x, skip=None):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = ((x - seq_mean)).permute(0, 2, 1)

        # patch division, down-sampling, and rfft
        x = x.reshape(-1, self.seg_num_x, self.num_sampling, self.down_sampling).permute(0, 1, 3, 2) # bc,n,period,samp
        x = torch.fft.rfft(x, dim=3)[:, :, :, :self.cut_freq]
        x = x.reshape(-1, self.enc_in, self.seg_num_x, self.down_sampling, self.cut_freq)

        # Sparse Frequency Mixer
        if self.flinear_individual:
            x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.down_sampling, self.flinear_sparse_num, self.in_sparse_freq)
            x = torch.einsum('bcsdft,cfet->bcsdfe', x, self.flinear_weight) + x
        else:
            x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.down_sampling, self.flinear_sparse_num, self.in_sparse_freq)
            x = torch.einsum('bcsdft,fet->bcsdfe', x, self.flinear_weight) + x

        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.down_sampling, self.cut_freq)
        x = x.permute(0, 1, 3, 4, 2)

        # patch predictor
        if self.linear_individual:
            x = x.reshape(batch_size, self.enc_in, self.down_sampling, self.group, self.in_group_freq, self.seg_num_x)
            tmp = torch.einsum('bcfgkn,cgyn->bcfgky', x, self.linear_weight)
        else:
            x = x.reshape(batch_size, self.enc_in, self.down_sampling, self.group, self.in_group_freq, self.seg_num_x)
            tmp = torch.einsum('bcfgkn,gyn->bcfgky', x, self.linear_weight)

        x = tmp.reshape(batch_size, self.enc_in, self.down_sampling, self.cut_freq, self.seg_num_y)
        tmp2 = torch.zeros([x.size(0), x.size(1), x.size(2), self.num_sampling // 2 + 1, x.size(4)], dtype=x.dtype).to(x.device)
        tmp2[:, :, :, :self.cut_freq, :] = x


        y = tmp2.permute(0, 1, 4, 2, 3)
        y = torch.fft.irfft(y, dim=4).permute(0, 1, 2, 4, 3)
        y = y.reshape(batch_size, self.enc_in, self.pred_len)

        y = (y.permute(0, 2, 1)) + seq_mean

        return y

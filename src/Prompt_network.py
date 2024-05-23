import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

import copy


class Memory(nn.Module):
    """ Memory prompt
    """
    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.args = args

        self.num_memory = num_memory
        self.memory_dim = memory_dim

        self.memMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C
        self.keyMatrix = nn.Parameter(torch.zeros(num_memory, memory_dim))  # M,C

        self.x_proj = nn.Linear(memory_dim, memory_dim)
        
        self.initialize_weights()

        print("model initialized memory")


    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.memMatrix, std=0.02)
        torch.nn.init.trunc_normal_(self.keyMatrix, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,x,Type='',shape=None):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # dot product

        assert x.shape[-1]==self.memMatrix.shape[-1]==self.keyMatrix.shape[-1], "dimension mismatch"

        x_query = torch.tanh(self.x_proj(x))

        att_weight = F.linear(input=x_query, weight=self.keyMatrix)  # [N,C] by [M,C]^T --> [N,M]

        att_weight = F.softmax(att_weight, dim=-1)  # NxM

        out = F.linear(att_weight, self.memMatrix.permute(1, 0))  # [N,M] by [M,C]  --> [N,C]
        loss_top = 0.0

        return dict(out=out, att_weight=att_weight, loss=loss_top)


class Sptial_prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, num_memory, memory_dim, in_channels, conv_num, args=None):
        super().__init__()

        self.conv = [nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=1, padding='same')]
        for i in range(conv_num-1):
            self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=memory_dim, kernel_size=2**(i+1)+1, padding='same'))

        self.conv = nn.Sequential(*self.conv)

        self.spatial_memory = Memory(num_memory, memory_dim, args=args)

        for conv in self.conv:
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.args = args

    def forward(self,x):
        """
        :param x: query features with size [N, C, H, W]
        :return: multiscale spatial_prompt
        """
        out = []
        loss = 0.0
        if self.args.mode=='case_study':
            np.save(self.args.model_path+'{}_X_spatial_origin.npy'.format(self.args.case_study_data),x.detach().numpy(),x)
        for i in range(len(self.conv)):
            t = self.conv[i](x).permute(0,2,3,1)
            shape = x.shape
            t = t.reshape(t.shape[0],-1,t.shape[-1])
            output = self.spatial_memory(t, Type='s_conv_{}'.format(i),shape=shape)
            out.append(output['out'].reshape(x.shape[0],x.shape[2],x.shape[3],t.shape[-1]).permute(0,3,1,2))
            loss += output['loss']
        return dict(out=out, loss=loss)



class Temporal_prompt(nn.Module):
    """ closeness and period
    """
    def __init__(self, num_memory, memory_dim, args=None):
        super().__init__()

        self.temporal_memory = Memory(num_memory, memory_dim, args=args)

        encdoer_layer = nn.TransformerEncoderLayer(d_model=memory_dim, nhead=4, dim_feedforward=memory_dim,batch_first = True)

        self.c_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)

        self.p_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)

        self.memory_dim = memory_dim

        self.args = args
        


    def forward(self,x_c, x_p):
        """
        :param x_c, x_p: query features with size [N, T, dim]
        :return: closeness and period
        """
        # closeness
        hc = self.c_encoder(x_c).mean(dim=1)
        shape_c = hc.shape

        # period
        hp = self.p_encoder(x_p).mean(dim=1)
        shape_p = hp.shape

        
        hc_output = self.temporal_memory(hc,Type='closeness',shape=shape_c)
        hp_output = self.temporal_memory(hp,Type='period',shape=shape_p)
        
        hc, loss_c = hc_output['out'], hc_output['loss']
        hp, loss_p = hp_output['out'], hp_output['loss']

        return dict(hc=hc, hp=hp, loss = loss_c+loss_p)
    


class Prompt_ST(nn.Module):
    """
    Prompt ST with spatial prompt and temporal prompt
    spatial prompt: multiscale convolutional kernels
    temporal prompt: closeness and period
    """
    def __init__(self, num_memory_spatial, num_memory_temporal, memory_dim, in_channels, conv_num, args=None):
        super().__init__()

        self.spatial_prompt = Sptial_prompt(num_memory_spatial, memory_dim, in_channels, conv_num, args=args)
        self.temporal_prompt = Temporal_prompt(num_memory_temporal, memory_dim, args=args)


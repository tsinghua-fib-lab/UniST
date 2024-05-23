from functools import partial

import torch
import torch.nn as nn
import math
import numpy as np
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp
from ConvLSTM import ConvLSTM

from Embed import DataEmbedding, TokenEmbedding, SpatialPatchEmb, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_with_resolution, get_1d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid_with_resolution
from mask_strategy import *
import copy

from Prompt_network import Prompt, Prompt_ST

def mae_vit_ST(args, **kwargs):
    if args.size == 'small':
        model = MaskedAutoencoderViT(
            embed_dim=128,
            depth=3,
            decoder_embed_dim = 128,
            decoder_depth=3,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '1':
        model = MaskedAutoencoderViT(
            embed_dim=64,
            depth=2,
            decoder_embed_dim = 64,
            decoder_depth=2,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '2':
        model = MaskedAutoencoderViT(
            embed_dim=64,
            depth=6,
            decoder_embed_dim = 64,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '3':
        model = MaskedAutoencoderViT(
            embed_dim=128,
            depth=4,
            decoder_embed_dim = 128,
            decoder_depth=3,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '4':
        model = MaskedAutoencoderViT(
            embed_dim=128,
            depth=8,
            decoder_embed_dim = 128,
            decoder_depth=8,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '5':
        model = MaskedAutoencoderViT(
            embed_dim=256,
            depth=4,
            decoder_embed_dim = 256,
            decoder_depth=4,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '6':
        model = MaskedAutoencoderViT(
            embed_dim=256,
            depth=8,
            decoder_embed_dim = 256,
            decoder_depth=6,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model

    elif args.size == '7':
        model = MaskedAutoencoderViT(
            embed_dim=256,
            depth=12,
            decoder_embed_dim = 256,
            decoder_depth=10,
            num_heads=4,
            decoder_num_heads=2,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = args.no_qkv_bias,
            args = args,
            **kwargs,
        )
        return model


    elif args.size == 'middle': # 6.7M
        model = MaskedAutoencoderViT(
            embed_dim=128,
            depth=6,
            decoder_embed_dim = 128,
            decoder_depth=4,
            num_heads=8,
            decoder_num_heads=4,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model
    
    elif args.size == 'middle2': # 6.7M
        model = MaskedAutoencoderViT(
            embed_dim=128,
            depth=6,
            decoder_embed_dim = 128,
            decoder_depth=6,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model

    elif args.size == 'large': # 28.9M
        model = MaskedAutoencoderViT(
            embed_dim=384,
            depth=6,
            decoder_embed_dim = 384,
            decoder_depth=6,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model

    elif args.size == 'SuperLarge':
        model = MaskedAutoencoderViT(
            embed_dim=384,
            depth=8,
            decoder_embed_dim = 384,
            decoder_depth=8,
            num_heads=8,
            decoder_num_heads=8,
            mlp_ratio=2,
            t_patch_size = args.t_patch_size,
            patch_size = args.patch_size,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            pos_emb = args.pos_emb,
            no_qkv_bias = bool(args.no_qkv_bias),
            args = args,
            **kwargs,
        )
        return model


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim, bias= qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, patch_size=1, in_chans=1,
                 embed_dim=1024, decoder_embed_dim=512, depth=24, decoder_depth=4, num_heads=16,  decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, t_patch_size=1,
                 no_qkv_bias=False, pos_emb = 'trivial', args=None, ):
        super().__init__()

        self.args = args

        self.pos_emb = pos_emb

        self.Embedding = DataEmbedding(1, embed_dim, args=args)

        #if 'TDrive' in args.dataset or 'BikeNYC2' in args.dataset:
        self.Embedding_24 = DataEmbedding(1, embed_dim, args=args, size1=24, size2 = 7)

        if args.prompt_ST != 0:
            self.st_prompt = Prompt_ST(args.num_memory_spatial, args.num_memory_temporal, embed_dim, self.args.his_len, args.conv_num, args=args)
            self.spatial_patch = SpatialPatchEmb(embed_dim, embed_dim, self.args.patch_size)

        # mask

        self.t_patch_size = t_patch_size
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        

        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50, embed_dim)
        )

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, 1024, decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, 50,  decoder_embed_dim)
        )

        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.no_qkv_bias = no_qkv_bias
        self.norm_layer = norm_layer


        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #self.mask_token2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Sequential(*[
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias= not self.args.no_qkv_bias),
            nn.GELU(),
            nn.Linear(decoder_embed_dim,self.t_patch_size * patch_size**2 * in_chans, bias= not self.args.no_qkv_bias)
        ])

        self.initialize_weights_trivial()

        print("model initialized MAE")

    def init_prompt_memory(self):
        self.prompt_memory = Prompt(
            num_memory1 = self.args.num_memory1,
            num_memory2 = self.args.num_memory2,
            memory_dim = self.embed_dim,
            args = self.args,
        )


    def init_prompt(self):
        self.blocks_prompt = nn.ModuleList(
            [
                Block(
                    self.embed_dim,
                    self.num_heads//2,
                    self.mlp_ratio,
                    qkv_bias=not self.no_qkv_bias,
                    qk_scale=None,
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth//2)
            ]
        )
        norm_layer = nn.LayerNorm
        self.norm_prompt = norm_layer(self.embed_dim)
        self.blocks_prompt.apply(self._init_weights)
        self.norm_prompt.apply(self._init_weights)

    def init_emb(self):
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        #torch.nn.init.normal_(self.mask_token2, std=0.02)

    def init_pos_emb(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)
        w = self.Embedding.value_embedding.tokenConv.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        #torch.nn.init.normal_(self.mask_token2, std=0.02)
        

    def init_head(self):
        self.decoder_norm.apply(self._init_weights)
        self.decoder_pred.apply(self._init_weights)


    def get_weights_sincos(self, num_t_patch, num_patch_1, num_patch_2):
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed_spatial.shape[-1],
            grid_size1 = num_patch_1,
            grid_size2 = num_patch_2
        )

        pos_embed_spatial = nn.Parameter(
                torch.zeros(1, num_patch_1 * num_patch_2, self.embed_dim)
            )
        pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_t_patch, self.embed_dim)
        )

        pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_temporal_emb = get_1d_sincos_pos_embed_from_grid(pos_embed_temporal.shape[-1], np.arange(num_t_patch, dtype=np.float32))

        pos_embed_temporal.data.copy_(torch.from_numpy(pos_temporal_emb).float().unsqueeze(0))

        pos_embed_spatial.requires_grad = False
        pos_embed_temporal.requires_grad = False

        return pos_embed_spatial, pos_embed_temporal, copy.deepcopy(pos_embed_spatial), copy.deepcopy(pos_embed_temporal)

    def initialize_weights_trivial(self):
        torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.hour_embed.weight.data, std=0.02)
        torch.nn.init.trunc_normal_(self.Embedding.temporal_embedding.weekday_embed.weight.data, std=0.02)

        w = self.Embedding.value_embedding.tokenConv.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        #torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.args.patch_size
        u = self.args.t_patch_size
        assert H % p == 0 and W % p == 0 and T % u == 0
        h = H // p
        w = W // p
        t = T // u
        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x


    def unpatchify(self, imgs):
        """
        imgs: (N, L, patch_size**2 *1)
        x: (N, 1, T, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info
        imgs = imgs.reshape(shape=(N, t, h, w, u, p, p))
        imgs = torch.einsum("nthwupq->ntuhpwq", imgs)
        imgs = imgs.reshape(shape=(N, T, H, W))
        return imgs


    def pos_embed_enc(self, ids_keep, batch, input_size, res1, res2):

        if self.pos_emb == 'trivial':
            pos_embed = self.pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            pos_embed_spatial, pos_embed_temporal, _, _ = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])
            pos_embed = pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.to(ids_keep.device)

        pos_embed = pos_embed.expand(batch, -1, -1)

        pos_embed_sort = torch.gather(
            pos_embed,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
        )

        return pos_embed_sort

    def pos_embed_dec(self, ids_keep, batch, input_size, res1, res2):

        if self.pos_emb == 'trivial':
            decoder_pos_embed = self.decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

        elif self.pos_emb == 'SinCos':
            _, _, decoder_pos_embed_spatial, decoder_pos_embed_temporal  = self.get_weights_sincos(input_size[0], input_size[1], input_size[2])

            decoder_pos_embed = decoder_pos_embed_spatial[:,:input_size[1]*input_size[2]].repeat(
                1, input_size[0], 1
            ) + torch.repeat_interleave(
                decoder_pos_embed_temporal[:,:input_size[0]],
                input_size[1] * input_size[2],
                dim=1,
            )

            decoder_pos_embed = decoder_pos_embed.to(ids_keep.device)

        decoder_pos_embed = decoder_pos_embed.expand(batch, -1, -1)

        return decoder_pos_embed



    def forward_encoder(self, x, x_mark, mask_ratio, mask_strategy, seed=None, data=None, res1 = 1, res2 = 1, mode='backward'):
        # embed patches
        N, _, T, H, W = x.shape

        if 'TDrive' in data or 'BikeNYC2' in data  or 'DC' in data or 'Porto' in data or 'Ausin' in data:
            x, TimeEmb = self.Embedding_24(x, x_mark, is_time = True)
        elif data is not None:
            x, TimeEmb = self.Embedding(x, x_mark, is_time = True)
        _, L, C = x.shape

        T = T // self.args.t_patch_size

        assert mode in ['backward','forward']

        if mode=='backward':

            if mask_strategy == 'random':
                x, mask, ids_restore, ids_keep = random_masking(x, mask_ratio)

            elif mask_strategy == 'tube':
                x, mask, ids_restore, ids_keep = tube_masking(x, mask_ratio, T=T)

            elif mask_strategy == 'tube_block':
                x, mask, ids_restore, ids_keep = tube_block_masking(x, mask_ratio, T=T)

            elif mask_strategy in ['frame','causal']:
                x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        elif mode == 'forward': # just evaluate
            if mask_strategy == 'random':
                x, mask, ids_restore, ids_keep = random_masking_evaluate(x, mask_ratio)

            elif mask_strategy == 'tube':
                x, mask, ids_restore, ids_keep = tube_masking_evaluate(x, mask_ratio, T=T)

            elif mask_strategy == 'tube_block':
                x, mask, ids_restore, ids_keep = tube_block_masking_evaluate(x, mask_ratio, T=T)

            elif mask_strategy in ['frame','causal']:
                x, mask, ids_restore, ids_keep = causal_masking(x, mask_ratio, T=T, mask_strategy=mask_strategy)

        input_size = (T, H//self.patch_size, W//self.patch_size)
        pos_embed_sort = self.pos_embed_enc(ids_keep, N, input_size, res1, res2)

        assert x.shape == pos_embed_sort.shape

        x_attn = x + pos_embed_sort

        # apply Transformer blocks
        for index, blk in enumerate(self.blocks):
            x_attn = blk(x_attn)

        x_attn = self.norm(x_attn)
        return x_attn, mask, ids_restore, input_size, TimeEmb
    

    def prompt_generate(self, shape, x_period, x_closeness, x, data, pos):
        P = x_period.shape[1]

        HW = x_closeness.shape[2]

        x_period = x_period.unsqueeze(2).reshape(-1,1,x_period.shape[-3],x_period.shape[-2],x_period.shape[-1]) 

        x_closeness = x_closeness.permute(0,2,1,3).reshape(-1, x_closeness.shape[1], x_closeness.shape[-1]) 

        if 'TDrive' in data or 'BikeNYC2' in data:
            x_period = self.Embedding_24.value_embedding(x_period).reshape(shape[0], P, -1, self.embed_dim)
           
        else:
            x_period = self.Embedding.value_embedding(x_period).reshape(shape[0], P, -1, self.embed_dim)

        x_period = x_period.permute(0,2,1,3).reshape(-1,x_period.shape[1],x_period.shape[-1])

        prompt_t = self.st_prompt.temporal_prompt(x_closeness, x_period)

        prompt_c = prompt_t['hc'].reshape(shape[0], -1, prompt_t['hc'].shape[-1])
        prompt_p = prompt_t['hp'].reshape(shape[0], -1, prompt_t['hp'].shape[-1]) 

        prompt_c = prompt_c.unsqueeze(dim=1).repeat(1,self.args.pred_len//self.args.t_patch_size,1,1)
        
        pos_t = pos.reshape(shape[0],(self.args.his_len + self.args.pred_len) // self.t_patch_size, HW, self.embed_dim)[:,-self.args.pred_len // self.t_patch_size:]

        assert prompt_c.shape == pos_t.shape
        prompt_c = (prompt_c+pos_t).reshape(shape[0],-1,self.embed_dim)

        t_loss = prompt_t['loss']

        out_s = self.st_prompt.spatial_prompt(x)

        out_s, s_loss = out_s['out'], out_s['loss']
        out_s = [self.spatial_patch(i).unsqueeze(dim=1).repeat(1,self.args.pred_len//self.args.t_patch_size,1,1).reshape(i.shape[0],-1,self.embed_dim).unsqueeze(dim=0) for i in out_s]

        out_s = torch.mean(torch.cat(out_s,dim=0),dim=0)

        return dict(tc = prompt_c, tp = prompt_p, s = out_s, loss = t_loss + s_loss)


    def forward_decoder(self, x, x_period, x_origin, ids_restore, mask_strategy, TimeEmb, input_size=None, res1=1, res2 = 1, data=None):
        N = x.shape[0]
        T, H, W = input_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        if mask_strategy == 'random':
            x = random_restore(x, ids_restore, N, T,  H, W, C, self.mask_token)

        elif mask_strategy in ['tube','tube_block']:
            x = tube_restore(x, ids_restore, N, T, H,  W,  C, self.mask_token)

        elif mask_strategy in ['frame','causal']:
            x = causal_restore(x, ids_restore, N, T, H,  W, C, self.mask_token)

        decoder_pos_embed = self.pos_embed_dec(ids_restore, N, input_size, res1, res2)

        # add pos embed
        assert x.shape == decoder_pos_embed.shape == TimeEmb.shape

        x_attn = x + decoder_pos_embed + TimeEmb

        if self.args.prompt_ST == 1:

            prompt = self.prompt_generate(x_attn.shape, x_period, x_attn.reshape(N, T, H*W, x_attn.shape[-1]), x_origin, data, pos = decoder_pos_embed + TimeEmb)

            if self.args.prompt_content == 's_p':
                token_prompt = prompt['tp'] + prompt['s']

            elif self.args.prompt_content == 'p_c':
                token_prompt = prompt['tp'] + prompt['tc']

            elif self.args.prompt_content == 's_c':
                token_prompt = prompt['s'] + prompt['tc']

            elif self.args.prompt_content == 's':
                token_prompt = prompt['s']

            elif self.args.prompt_content == 'p':
                token_prompt = prompt['tp']

            elif self.args.prompt_content == 'c':
                token_prompt = prompt['tc']

            elif self.args.prompt_content == 's_p_c':
                token_prompt = prompt['tc'] + prompt['s'] + prompt['tp']

            x_attn[:,-self.args.pred_len // self.args.t_patch_size * H * W:] += token_prompt

            loss_top = prompt['loss']


        # apply Transformer blocks
        for index, blk in enumerate(self.decoder_blocks):
            x_attn = blk(x_attn)
        x_attn = self.decoder_norm(x_attn)

        return x_attn, loss_top

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, T, H, W]
        pred: [N, t*h*w, u*p*p*1]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """

        target = self.patchify(imgs)

        assert pred.shape == target.shape

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss1 = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss2 = (loss * (1-mask)).sum() / (1-mask).sum()
        return loss1, loss2, target


    def forward(self, imgs, mask_ratio=0.75, mask_strategy='causal',seed=None, data='none', res=[1,1], mode='backward'):
        imgs, imgs_mark, imgs_period = imgs

        imgs_period = imgs_period[:,:,self.args.his_len:]

        T, H, W = imgs.shape[2:]
        latent, mask, ids_restore, input_size, TimeEmb = self.forward_encoder(imgs, imgs_mark, mask_ratio, mask_strategy, seed=seed, data=data, res1 = res[0], res2 = res[1], mode=mode)

        pred, _ = self.forward_decoder(latent, imgs_period,  imgs[:,:,:self.args.his_len].squeeze(dim=1).clone(),  ids_restore, mask_strategy, TimeEmb, input_size = input_size, res1 = res[0], res2 = res[1], data = data)  # [N, L, p*p*1]
        L = pred.shape[1]

        pred = self.decoder_pred(pred)

        loss1, loss2, target = self.forward_loss(imgs, pred, mask)
        
        return loss1, loss2, pred, target, mask

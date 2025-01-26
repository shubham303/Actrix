from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import drop_path, trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from actrix.models import registry
from actrix.models.base import ModelConfig, Model


@dataclass
class MVDVisionTransformerConfig(ModelConfig):
    name:str = "mvd_vit_small"
    embed_dim: int = 384
    patch_size: tuple[int, int] = (16, 16)
    num_frames: int = 16
    tublet_size: int = 2
    scale_t: int | None = None
    use_cls_token: bool = False
    drop_path_prob: float = 0.01
    num_heads: int = 6
    mlp_ratio: float = 4.0
    activation_layer: str = "gelu"
    normalization_layer: str = "layer_norm"
    qkv_bias: bool = False
    qk_scale: float | None = None
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    attn_head_dim: int | None = None
    mlp_dropout: float = 0.0
    in_chans: int = 3 
    init_values:float = 0.0
    tubelet_size: int = 2
    depth: int = 12    
    fc_drop_rate: float = 0.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    init_scale: float = 0.0
    use_checkpoint: bool = False
    use_mean_pooling: bool = True

class MVDVisionTransformerConfigSmall(MVDVisionTransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(name = "mvd_vit_small", **kwargs)
        self.embed_dim = 384
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4
        self.qkv_bias = True
        

class MVDVisionTransformerConfigBase(MVDVisionTransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(name = "mvd_vit_base", **kwargs)
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.mlp_ratio = 4
        self.qkv_bias = True

class MVDVisionTransformerConfigLarge(MVDVisionTransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(name = "mvd_vit_large", **kwargs)
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 4
        self.qkv_bias = True

class MVDVisionTransformerConfigHuge(MVDVisionTransformerConfig):
    def __init__(self, **kwargs):
        super().__init__(name = "mvd_vit_huge", **kwargs)
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.mlp_ratio = 4
        self.qkv_bias = True
        


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert (
        embed_dim % 2 == 0
    ), " For 2d pos embed, embed_dim must be divisible by 2"  # noqa: S101

    # use half of dimensions to encode grid_h
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: np.ndarray, scale: int | None = None
):
    assert embed_dim % 2 == 0  # noqa: S101
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
        
    pos = pos.reshape(-1)  # (M,)
    if scale is not None:
        pos = pos * scale
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def get_3d_sincos_pos_embed(config: MVDVisionTransformerConfig) -> torch.FloatTensor:
    """Get the 3D sine-cosine positional embedding.

    Args:
        config (VisionTransformerConfig): The configuration for the Vision Transformer.

    Returns:
        torch.FloatTensor: The 3D positional embedding tensor.

    """
    embed_dim = config.embed_dim
    grid_size = config.img_size[0] // config.patch_size[0]
    t_size = config.num_frames // config.tublet_size
    scale_t = config.scale_t

    assert (
        embed_dim % 4 == 0
    ), " For 3d pos embed, embed_dim must be divisible by 4"  # noqa: S101

    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = _get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = _get_1d_sincos_pos_embed_from_grid(
        embed_dim_temporal, grid_t, scale=scale_t
    )

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(
        pos_embed_temporal, grid_size**2, axis=1
    )  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(
        pos_embed_spatial, t_size, axis=0
    )  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    if config.use_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return torch.FloatTensor(pos_embed).unsqueeze(0)  # type: ignore


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)



class Mlp(nn.Module):  # noqa: D101
    def __init__(
        self,
        config: MVDVisionTransformerConfig,
    ) -> None:
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        out_features = config.embed_dim
        act_layer = nn.GELU if config.activation_layer == "gelu" else nn.ReLU
        drop = config.mlp_dropout

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the module.

        Args:x (torch.Tensor): Input tensor.

        Returns:torch.Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(self, config: MVDVisionTransformerConfig):
        super().__init__()
        dim = config.embed_dim
        num_heads = config.num_heads
        qkv_bias = config.qkv_bias
        qk_scale = config.qk_scale
        attn_dropout = config.attn_dropout
        proj_dropout = config.proj_dropout
        attn_head_dim = config.attn_head_dim

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads if attn_head_dim is None else attn_head_dim

        self.all_head_dim = self.head_dim * self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, self.all_head_dim*3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_dim))
        else:
            self.q_bias = self.v_bias = None


        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Dropout
        self.attention_dropout = attn_dropout
        self.projection_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # Project and split into Q, K, V
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot product attention
        x = F.scaled_dot_product_attention(
            q,  # (B, H, N, D)
            k,  # (B, H, N, D)
            v,  # (B, H, N, D)
            dropout_p=self.attention_dropout,
            scale=self.scale,
        )

        # Reshape and project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.projection_dropout(x)

        return x

class Block(nn.Module):
    def __init__(self, config: MVDVisionTransformerConfig, drop_path_rate:float=0.0):
        super().__init__()
        # Map normalization layer string to class
        if config.normalization_layer == "layer_norm":
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {config.normalization_layer}")
        
        # Initialize components using config
        self.norm1 = norm_layer(config.embed_dim)
        self.attn = Attention(config)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = norm_layer(config.embed_dim)
        
        # MLP layer (now fully config-driven)
        self.mlp = Mlp(config)

        # Initialize gamma parameters if needed
        if config.init_values > 0:
            self.gamma_1 = nn.Parameter(
                config.init_values * torch.ones(config.embed_dim),
                requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                config.init_values * torch.ones(config.embed_dim),
                requires_grad=True
            )
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass remains unchanged
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, config: MVDVisionTransformerConfig):
        super().__init__()
        # Extract parameters from configto_2tuple
        img_size = config.img_size
        patch_size = config.patch_size
        in_chans = 3  # Default for RGB images
        embed_dim = config.embed_dim
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        # Compute derived values
        self.tubelet_size = int(tubelet_size)
        self.num_patches_w = img_size[1] // patch_size[1]
        self.num_patches_h = img_size[0] // patch_size[0]
        self.num_patches_t = num_frames // self.tubelet_size
        self.num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames // self.tubelet_size)
        )

        # Store sizes
        self.img_size = img_size
        self.patch_size = patch_size

        # Projection layer
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for PatchEmbed.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Sinusoid position encoding table."""
    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

@registry.register_model(MVDVisionTransformerConfigSmall)
@registry.register_model(MVDVisionTransformerConfigBase)
@registry.register_model(MVDVisionTransformerConfigLarge)
@registry.register_model(MVDVisionTransformerConfigHuge)
class MVDVisionTransformer(Model , nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self, config: MVDVisionTransformerConfig):
        Model.__init__(self, config=config)
        nn.Module.__init__(self)
        # Extract parameters from config
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim
        self.num_features = self.embed_dim  # For consistency with other models
        self.tubelet_size = config.tubelet_size
        self.patch_size = config.patch_size
        self.use_cls_token = config.use_cls_token
        self.use_checkpoint = config.use_checkpoint
        self.use_mean_pooling = config.use_mean_pooling

        # Patch embedding
        self.patch_embed = PatchEmbed(config)

        # Class token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.cls_token = None

        # Positional embedding
        self.pos_embed = get_3d_sincos_pos_embed(config)

        # Dropout
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(config, dpr[i]) for i in range(config.depth)
        ])

        # Normalization
        self.norm = nn.Identity() if self.use_mean_pooling else nn.LayerNorm(self.embed_dim)
        self.fc_norm = nn.LayerNorm(self.embed_dim) if self.use_mean_pooling else None

        # Dropout for final classification head
        self.fc_dropout = nn.Dropout(p=config.fc_drop_rate) if config.fc_drop_rate > 0 else nn.Identity()

        # Classification head
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        # Initialize weights
        if self.use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)

        if self.num_classes > 0:
            trunc_normal_(self.head.weight, std=0.02)
            self.head.weight.data.mul_(config.init_scale)
            self.head.bias.data.mul_(config.init_scale)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        """Get the number of transformer blocks."""
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Return parameters that should not have weight decay applied."""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self) -> nn.Module:
        """Get the classification head."""
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str = ''):
        """Reset the classification head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature extraction."""
        x = self.patch_embed(x)
        B, _, _ = x.size()

        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        # Add class token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        # Apply transformer blocks
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)

        # Apply normalization
        x = self.norm(x)
        if self.fc_norm is not None:
            if self.use_cls_token:
                x = x[:, 1:]
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Vision Transformer."""
        x = self.forward_features(x)
        x = self.head(self.fc_dropout(x))
        return x
    

    










if __name__ == "__main__":
    
    config = MVDVisionTransformerConfig(img_size=(64,64))
    model = MVDVisionTransformer(config)


    # create x of shape  (B, C, T, H, W).
    x = torch.rand(size=(1,3,16,64,64), dtype=torch.float32)
    output = model(x)
    print(output.shape)
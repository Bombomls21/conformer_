import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_
# from supression import SelectDropMAX


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    '''
    Block trong Transformer
    Norm --> Multihead self-attention --> norm --> MLP
    lưu ý có Residual connection
    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        '''Basic convluation block used in Conformer.

        This block includes three convluation modules, and supports three new
        functions:
        1. Returns the output of both the final layers and the second convluation
        module.
        2. Fuses the input of the second convluation module with an extra input
        feature map.
        3. Supports to add an extra convluation module to the identity connection.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): The stride of the second convluation module.
                Defaults to 1.
            groups (int): The groups of the second convluation module.
                Defaults to 1.
            drop_path_rate (float): The rate of the DropPath layer. Defaults to 0.
            with_residual_conv (bool): Whether to add an extra convluation module
                to the identity connection. Defaults to False.
            norm_cfg (dict): The config of normalization layers.
                Defaults to ``dict(type='BN', eps=1e-6)``.
            act_cfg (dict): The config of activative functions.
                Defaults to ``dict(type='ReLU', inplace=True))``.
            init_cfg (dict, optional): The extra config to initialize the module.
                Defaults to None.
        '''
        super(ConvBlock, self).__init__()

        expansion = 4  # hệ số mở rộng số kênh
        med_planes = outplanes // expansion

        # 1X1Conv-BN ~ conv1, bn1, act1
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        # 3X3Conv-BN ~ conv2, bn2, act2
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1,
                               bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        # 1X1Conv-BN ~ conv3, bn3, act3
        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        # Residual Connection
        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        # Khởi tạo bn3 với weight = 0
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        # giảm chiều
        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        # Tăng chiều
        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        # Trả vê x2 nếu cần feature map ở giữa
        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """
    Từ feature map của CNN --> token cho Transformer
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        '''
        :param inplanes:
        :param outplanes:
        :param dw_stride:
        :param act_layer:
        :param norm_layer:
        '''
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        # cân bằng kênh
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        # downsample ~ cân bằng không gian
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)
        # LayerNorm
        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        '''
        :param x: Feature map của CNN # [32, 64, 56, 56]
        :param x_t: Token của Trans # [32, 197, 768]
        :return: token mới, đầu vào cho Transformer
        '''
        x = self.conv_project(x)  # [N, C, H, W]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        # Sum với cls_token của x_t theo chiều num_tokens
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """
    Token của Transformer --> feature map của CNN
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        '''
        :param x: token của Transformer # [batch_size, num_tokens, embed_dim] #  [32, 197, 768]
        :param H, W: height và width mong muốn của feature map đầu ra
        :return: feature map upsampled
        '''
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)  #
        x_r = self.act(self.bn(self.conv_project(x_r)))
        # Upsample lên kích thước mong muốn bằng interpolate
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """

    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):
        '''
        :param inplanes: số kênh đầu vào
        :param act_layer: hàm activation
        :param groups: nhóm convolution
        :param norm_layer: lớp normalization
        :param drop_block: lớp dropblock
        :param drop_path: lớp droppath
        '''
        super(Med_ConvBlock, self).__init__()
        # hệ số mở rộng kênh và số kênh trung gian
        expansion = 4
        med_planes = inplanes // expansion

        # 1x1 conv giảm chiều kênh, batch norm, activation
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        # 3x3 conv
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        # 1x1 conv tăng chiều kênh về ban đầu
        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """Basic module for Conformer.
    This module is a fusion of CNN block transformer encoder block.
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        # Convolutional
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride,
                                   groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True,
                                          groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        #
        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        # Giảm chiều từ CNN xuống token
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        # Tăng chiều từ token lên feature map CNN
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        # khối Transformer
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        '''
        Input:
            x: feature map CNN [32, 256, 56, 56]
            x_t: embedding từ Transformer trước đó [32, 197, 768]

        Qua cnn_block, trích xuất thêm đặc trưng, lấy x2 là feature map ở giữa
        x2 được giảm kích thước xuống embedding bằng squeeze_block -> cộng vào x_t
        Cho qua trans_block để mã hóa thông tin embedding -> x_t (embedding mới từ Transformer)
        x_t được tăng kích thước lên bằng expand_block -> x_t_r (feature map từ Transformer)
        Fusion x_t_r với x (feature map CNN) để kết hợp thông tin -> x (feature map mới đã kết hợp từ CNN và Transformer)

        Return:
            x_t: embedding chứa thông tin từ feature map CNN
            x: feature map chứa thông tin từ embedding Transformer
        '''
        x, x2 = self.cnn_block(x)  # CNN
        # [32, 256, 56, 56], [32, 64, 56, 56], 3
        # [32, 512, 28, 28], [32, 128, 28, 28], 4
        # [32, 1024, 14, 14], [32, 256, 14, 14], 4
        _, _, H, W = x2.shape
        # CNN xuống token
        x_st = self.squeeze_block(x2, x_t)  # [32, 197, 768]

        # transformer
        x_t = self.trans_block(x_st + x_t)  # [32, 197, 768]

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride,
                                  W // self.dw_stride)  # bs, (64, 56, 56), (128, 28, 28), (256, 14, 14)
        x = self.fusion_block(x, x_t_r, return_x_2=False)
        # x: bs, (256, 56, 56), (512, 28, 28), (1024, 14, 14), (1024, 7, 7)
        # x_t: [bs, 197, 768]
        return x, x_t


class Conformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, num_classes=40, base_channel=64, channel_ratio=6, num_med_block=0,
                 embed_dim=576, depth=12, num_heads=9, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        '''
        patch_size: kích thước mỗi patch ảnh đầu vào cho Transformer
        in_chans: kênh input ảnh
        num_classes: số lớp dự đoán
        base_channel: số kênh cơ sở cho các khối CNN
        channel_ratio: tỷ lệ điều chỉnh số kênh theo từng stage
        num_med_block: số khối CNN bổ sung sau mỗi ConvTransBlock
        embed_dim: chiều embedding cho Transformer
        depth: số khối ConvTransBlock
        num_heads: số heads trong Transformer
        mlp_ratio: tỷ lệ chiều ẩn trong MLP của Transformer
        qkv_bias: có dùng bias cho qkv hay không
        qk_scale: factor nhân với qk
        các tham số drop_rate: tỷ lệ dropout
        '''

        # thuộc tính cho lớp
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # 768 num_features for consistency with other models
        assert depth % 3 == 0

        # cls token và mảng chứa tỷ lệ dropout theo từng khối.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)  # LayerNorm cho transformer output
        self.trans_cls_head = nn.Linear(embed_dim,
                                        num_classes) if num_classes > 0 else nn.Identity()  # linear projection từ token của Transformer
        self.pooling = nn.AdaptiveAvgPool2d(1)  # adaptive avg pooling 2D từ CNN output
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)  # linear projection từ CNN output

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # 7x7 conv, stride = 2 --> 3x3 max pool, stride = 2 --> initial local feature
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # ===================================   stage 1  ===================================
        # Block CNN
        stage_1_channel = int(base_channel * channel_ratio)  # 256
        trans_dw_stride = patch_size // 4  # 4,4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)

        # embedding cho transformer
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        # Block Transformer
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        # ===================================   stage 2  ===================================
        # 2 ~ 4
        init_stage = 2
        fin_stage = depth // 3 + 1  # 5
        for i in range(init_stage, fin_stage):  # 2 ~ 4
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )
        # print("Các ConvTransBlock:")
        # for i in range(init_stage, fin_stage):
        #     block = getattr(self, 'conv_trans_' + str(i))
        #     print(block)

        # ===================================   stage 3  ===================================
        stage_2_channel = int(base_channel * channel_ratio * 2)  # 512
        # 5~8
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        # ===================================   stage 4  ===================================
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4,
                                embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block, last_fusion=last_fusion
                            )
                            )
        self.fin_stage = fin_stage
        
        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        # dùng trong Transformer để biểu diễn cả bức ảnh.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [32, 1, 768]

        # pdb.set_trace()

        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # stage 2
        x = self.conv_1(x_base, return_x_2=False)  # Input CNN
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)  # Embed
        x_t = torch.cat([cls_tokens, x_t], dim=1)  # Nối cls_token với embed
        x_t = self.trans_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

        # x : [bs, 1024, 7, 7]
        # x_t : [bs, 197, 768]
        
        # conv classification
        x_p = self.pooling(x).flatten(1)  # [bs, 1024]
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)  # [32, 197, 768]
        tran_cls = self.trans_cls_head(x_t[:, 0])

        
        return [conv_cls, tran_cls]  # [bs, classes]

    def change_size(self, x):
        B, L, C = x.shape
        H, W = int(L ** 0.5), int(L ** 0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    x = torch.rand(16, 3, 224, 224)
    # model = Conformer()
    model = Conformer(num_classes=40)
    model.cuda()
    y = model(x.cuda())
    print(f'Total trainable parameters: {count_parameters(model)}')

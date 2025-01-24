# Referenced from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=cfCcEYjg7y46

import torch
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch import nn
from torchvision.models import resnet50
from PIL import Image

torch.set_grad_enabled(False)

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
            self, 
            num_classes, 
            hidden_dim=256, 
            nheads=8,
            num_encoder_layers=6, 
            num_decoder_layers=6
        ):
        super().__init__()

        # 1. Resnet Backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # 2. Conv2d
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # 3. Transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, 
            num_encoder_layers, num_decoder_layers
        )

        # 4. FCNN (img class & bbox)
        box_feature_dim = 4 # x y w h
        self.linear_class = nn.Linear(hidden_dim, num_classes+1)
        self.linear_box = nn.Linear(hidden_dim, box_feature_dim)

        # 5. object queries
        obj_query_dim = 100
        self.query_pos = nn.Parameter(torch.rand(obj_query_dim, hidden_dim))

        # 6. positional encodings (col & row)
        # (50 128)
        self.row_embed = nn.Parameter(torch.rand(obj_query_dim // 2, hidden_dim // 2)) 
        self.col_embed = nn.Parameter(torch.rand(obj_query_dim // 2, hidden_dim // 2))

    def forward(self, inputs):
        # Walking through Backdone
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Conv2d for shaping
        h = self.conv(x) # (2048 H W) > (256 H W)
        print(f"shape of h: {h.shape}")

        # Construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            # (W 128) > (1 W 128) > (H W 128)
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            # (H 128) > (H 1 128) > (H W 128)
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            # (H W 256) > (HW 256) > (HW 1 256)
        ], dim=-1).flatten(0, 1).unsqeeze(1)

        # Pass through Transformer
        # (256 H W) > (1 256 HW) > (HW 1 256)
        tf_src = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        # (100 256) > (100 1 256)
        tf_tgt = self.query_pos.unsqueeze(1)
        # (1 100 256)
        tf_output = self.transformer(tf_src, tf_tgt).transpose(0, 1)

        # Prediction Linear layer
        # (1 100 256) > (1 100 num_classes+1)
        pred_logits = self.linear_class(tf_output)
        # (1 100 256) > (1 100 box_feature_dim)
        pred_boxes = self.linear_box(tf_output).sigmoid()

        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
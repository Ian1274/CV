import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
import torch.nn.functional as F
import os


# 定义模型
class SwinTransformer(nn.Module):
    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224")
        self.num_features = num_features
        self.feat = nn.Linear(1024, num_features) if num_features > 0 else None # 1024是swin_base_patch4_window7_224的输出维度

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x


# 数据预处理
class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = Transforms.Compose(
            [
                Transforms.Resize((self.height, self.width)), 
                Transforms.ToTensor(), 
                Transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ), 
            ]
        )

    def __call__(self, img):
        return self.transformer(img).unsqueeze(0)


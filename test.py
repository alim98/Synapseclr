import timm_3d
import torch

m = timm_3d.create_model(
    'swin',
    pretrained=True,
    num_classes=0,
    global_pool=''
)

# Shape of input (B, C, H, W, D). B - batch size, C - channels, H - height, W - width, D - depth
res = m(torch.randn(2, 3, 128, 128, 128))
print(f'Output shape: {res.shape}') 
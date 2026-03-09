_base_ = ['./mysegformer_10_mit-b0_1xb8-160k_nyuv2-480x640.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 18, 3],
        downsample_ratio=0.7),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

_base_ = ['../../../configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py']

custom_imports = dict(imports=['projects.M2D.Mamba2D'],
                      allow_failed_imports=False)

model=dict(
    type='MaskRCNN',
    # Backbone config
    backbone=dict(
        _delete_=True,
        type='Mamba2D',
        n_blocks=(3,12,14,3),
        ds_stages=("mf_stem","mf_2","mf_2","mf_2",),
        embed_dim=[96,192,384,576],
        token_mixer=("2D_local", "2D_local", "RopeAttention", "RopeAttention"),
        drop_path_rate=0.2, # Half of 0.4 from pre-training
        res_scale_init_values=[1.0,1.0,1.0,1.0],
        featmaps_out=True,
        inter_flip=True,
        recomp="partial",
        pt_ckpt = "projects/M2D/pretrained/M2D-S.ckpt",
        pt_layers = ("backbone.ds", "backbone.stages")
    ),

    # FPN config
    neck=dict(
        in_channels=[96,192,384,576],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    constructor='MambaOptimConstructor',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=8   # Matches base 8*2=16 effective batch size on single GPU
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='mmdet-mask-rcnn-3x',
                 entity='2D-Mamba',
                 id='M2D-S-3x-01',
                 resume="allow"
             )
        )
    ],
    name='visualizer'
)

default_hooks = dict(
    visualization=dict(
        type='DetVisualizationHook',
        draw=False
    )
)

_base_ = ['../../../configs/swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py']

custom_imports = dict(imports=['projects.M2D.Mamba2D'],
                      allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True) # Default cfg
#norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) # Safer on single GPU

model=dict(
    pretrained=None,
    # Backbone config
    backbone=dict(
        _delete_=True,
        type='Mamba2D',
        n_blocks=(3,3,9,3),
        ds_stages=("mf_stem","mf_2","mf_2","mf_2",),
        embed_dim=[32,64,160,256],
        token_mixer=("2D_local", "2D_local", "RopeAttention", "RopeAttention"),
        drop_path_rate=0.175,
        res_scale_init_values=[1.0,1.0,1.0,1.0],
        featmaps_out=True,
        inter_flip=True,
        recomp="partial",
        pt_ckpt = "projects/M2D/pretrained/M2D-N.ckpt",
        pt_layers = ("backbone.ds", "backbone.stages")
    ),

    # Head config
    decode_head=dict(
        in_channels=[32,64,160,256],
        norm_cfg=norm_cfg
    ),

    auxiliary_head=dict(
        in_channels=160,
        norm_cfg=norm_cfg
    )
)

# Override default optimizer
optimizer = None
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    constructor='MambaOptimConstructor',
    optimizer=dict(type='AdamW', lr=0.00006, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=2
)

# Match effective batch size of 16 from base cfg
train_dataloader = dict(batch_size=8)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='mmseg-upernet',
                 entity='2D-Mamba',
                 id='M2D-N-01',
             )
        )
    ],
    name='visualizer'
)

default_hooks = dict(
    visualization=dict(
        type='SegVisualizationHook',
        draw=False
    )
)

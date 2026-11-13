"""PEARL R2S100K settings. The standalone CLI creates vocabulary.txt at run time."""
_base_ = './base_config.py'
model = dict(clip_path='ViT-B/16', arch='reduced', attn_strategy='pearl', use_kk=True,
             alpha_kk=0.08, align_solver='polar', polar_iters=5, use_prop='on',
             grid=80, slide_crop=224, slide_stride=112, logit_scale=40, prob_thd=0)
test_pipeline = [dict(type='LoadImageFromFile'),
                 dict(type='Resize', scale=(2048, 336), keep_ratio=True),
                 dict(type='PackSegInputs')]

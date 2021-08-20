echo 'configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fcn_hr18s_512x512_160k_ade20k configs/hrnet/fcn_hr18s_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24727 >/dev/null &
echo 'configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fcn_hr18s_512x1024_160k_cityscapes configs/hrnet/fcn_hr18s_512x1024_160k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24728 >/dev/null &
echo 'configs/hrnet/fcn_hr48_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fcn_hr48_512x512_160k_ade20k configs/hrnet/fcn_hr48_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24729 >/dev/null &
echo 'configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fcn_hr48_512x1024_160k_cityscapes configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24730 >/dev/null &
echo 'configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_r50-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24731 >/dev/null &
echo 'configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_r101-d8_512x1024_80k_cityscapes configs/pspnet/pspnet_r101-d8_512x1024_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24732 >/dev/null &
echo 'configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_r101-d8_512x512_160k_ade20k configs/pspnet/pspnet_r101-d8_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24733 >/dev/null &
echo 'configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_r50-d8_512x512_160k_ade20k configs/pspnet/pspnet_r50-d8_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24734 >/dev/null &
echo 'configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_s101-d8_512x512_160k_ade20k configs/resnest/pspnet_s101-d8_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24735 >/dev/null &
echo 'configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab pspnet_s101-d8_512x1024_80k_cityscapes configs/resnest/pspnet_s101-d8_512x1024_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24736 >/dev/null &
echo 'configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab fast_scnn_lr0.12_8x4_160k_cityscapes configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24737 >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deeplabv3plus_r101-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24738 >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deeplabv3plus_r101-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24739 >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deeplabv3plus_r50-d8_512x1024_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24740 >/dev/null &
echo 'configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deeplabv3plus_r50-d8_769x769_80k_cityscapes configs/deeplabv3plus/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24741 >/dev/null &
echo 'configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab upernet_vit-b16_ln_mln_512x512_160k_ade20k configs/vit/upernet_vit-b16_ln_mln_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24742 >/dev/null &
echo 'configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab upernet_deit-s16_ln_mln_512x512_160k_ade20k configs/vit/upernet_deit-s16_ln_mln_512x512_160k_ade20k.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24743 >/dev/null &
echo 'configs/fp16/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes configs/fp16/deeplabv3plus_r101-d8_512x1024_80k_fp16_cityscapes.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24744 >/dev/null &
echo 'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py' &
GPUS=4  GPUS_PER_NODE=4  CPUS_PER_TASK=2 ./tools/slurm_train.sh openmmlab upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py --options checkpoint_config.max_keep_ckpts=1 dist_params.port=24745 >/dev/null &

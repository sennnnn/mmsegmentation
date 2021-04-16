# How to use [HRNetV2-W48 official weights](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master) to test on [our repo](https://github.com/open-mmlab/mmsegmentation)

```
ps: testing on pascal context dataset
```

1. Download HRNetV2-W48 official weights [59 classes](https://1drv.ms/u/s!Aus8VCZ_C_33f5Bfbt4KmLeX8uw) and [69 classes](https://1drv.ms/u/s!Aus8VCZ_C_33gQEHDQrZCiv4R5mf). (You'd better download them in [master branch of official repo](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master))
2. Then, using the convert script to convert official weights to pytorch style:

```
python tools/scripts/convert_hrnet.py [src_path] [dst_path]
```

3. In order to match the inference process, modify codes according to [this commit](https://github.com/sennnnn/mmsegmentation/commit/803dca1007ae745129317b4af2602a28f888355c).
4. Then, using hrnet config to test model on pascal context dataset:

```
python -u tools/test.py [config path] [converted official weights path] --eval mIoU
```

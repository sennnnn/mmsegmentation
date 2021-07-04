import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DummyCityscapes(CustomDataset):
    """`Dummy Cityscapes Dataset.

    This implementation try to create a dataset to simulate cityscapes dataset
    in training environment.
    """
    image_shape = (512, 1024)

    train_size = 2975

    test_size = 500

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        # The train set and val set length of cityscapes
        super().__init__(img_dir='', **kwargs)
        if self.test_mode:
            self.size = self.test_size
        else:
            self.size = self.train_size

        self.dummy_images = {
            i: np.random.randint(
                0, 256, size=(*self.image_shape, 3), dtype=np.uint8)
            for i in range(100)
        }
        self.dummy_masks = {
            i: np.random.randint(
                0, 19, size=(*self.image_shape, ), dtype=np.uint8)
            for i in range(100)
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx %= 100
        data = dict(img_info={}, ann_info={})
        data['img'] = self.dummy_images[idx]
        data['gt_semantic_seg'] = self.dummy_masks[idx]
        shape = data['img'].shape
        data['img_shape'] = self.image_shape
        data['ori_shape'] = self.image_shape
        # Set initial values for default meta_keys
        data['pad_shape'] = self.image_shape
        data['scale_factor'] = 1
        num_channels = 1 if len(shape) < 3 else shape[2]
        data['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        data['ori_filename'] = f'{idx}_dummy.jpg'
        data['filename'] = f'{idx}_dummy.jpg'
        return self.pipeline(data)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        return {}

    def get_gt_seg_maps(self, efficient_test):
        gt_seg_maps = []
        for i in range(self.test_size):
            gt_seg_maps.append(self.dummy_masks[i % 100])

        return gt_seg_maps

import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DummyCityscapes(CustomDataset):
    """`Dummy Cityscapes Dataset.

    This implementation try to create a dataset to simulate cityscapes dataset
    in training environment.
    """
    image_shape = (112, 224)

    def __init__(self, **kwargs):
        # The train set and val set length of cityscapes
        super().__init__(img_dir='', **kwargs)
        if self.test_mode:
            self.size = 500
        else:
            self.size = 2975

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

        self.datas = self.data_retrieval()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.pipeline(self.datas[idx])

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        return {}

    def data_retrieval(self):
        datas = []
        for i in range(self.size):
            i = i % 100
            data = dict(img_info={}, ann_info={})
            data['img'] = self.dummy_images[i]
            data['gt_semantic_seg'] = self.dummy_masks[i]
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
            data['ori_filename'] = f'{i}_dummy.jpg'
            data['filename'] = f'{i}_dummy.jpg'
            datas.append(data)
        return datas

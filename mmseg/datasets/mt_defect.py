from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class MT_defect_Dataset(CustomDataset):
    CLASSES = ("background", "blowhole","crack","fray","break",'uneven')
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg' or '.bmp', seg_map_suffix='.png',reduce_zero_label=False,
                         # ignore_index=0,
           split=split, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None


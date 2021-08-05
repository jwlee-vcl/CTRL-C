from datasets.gsv_dataset import build_gsv
from datasets.hlw_dataset import build_hlw
from datasets.sun360_dataset import build_sun360
from datasets.yud_dataset import build_yud
from datasets.ecd_dataset import build_ecd
from datasets.kitti_dataset import build_kitti

def build_dataset(image_set, cfg):
    return build_gsv(image_set, cfg)

def build_hlw_dataset(image_set, cfg):
    return build_hlw(image_set, cfg)

def build_sun360_dataset(image_set, cfg):
    return build_sun360(image_set, cfg)

def build_yud_dataset(image_set, cfg):
    return build_yud(image_set, cfg)

def build_ecd_dataset(image_set, cfg):
    return build_ecd(image_set, cfg)

def build_kitti_dataset(image_set, cfg):
    return build_kitti(image_set, cfg)
    

from datasets.gsv_dataset import build_gsv
from datasets.sun360_dataset import build_sun360
from datasets.hlw_dataset import build_hlw
from datasets.image_dataset import build_image

def build_gsv_dataset(image_set, cfg):
    return build_gsv(image_set, cfg)

def build_sun360_dataset(image_set, cfg):
    return build_sun360(image_set, cfg)

def build_hlw_dataset(image_set, cfg):
    return build_hlw(image_set, cfg)

def build_image_dataset(image_set, cfg):
    return build_image(image_set, cfg)

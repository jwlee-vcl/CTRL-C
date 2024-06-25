from datasets.gsv_dataset import build_gsv
from datasets.sun360_dataset import build_sun360
from datasets.hlw_dataset import build_hlw
from datasets.image_dataset import build_image

def build_gsv_dataset(image_set, cfg, basepath="/data/google_street_view_191210/manhattan/"):
    return build_gsv(image_set, cfg, basepath)

def build_sun360_dataset(image_set, cfg, basepath="/data/sun360_20200306/"):
    return build_sun360(image_set, cfg, basepath)

def build_hlw_dataset(image_set, cfg, basepath="/data/hlw/images/"):
    return build_hlw(image_set, cfg, basepath)

def build_image_dataset(image_set, cfg, basepath=""):
    return build_image(image_set, cfg, basepath)

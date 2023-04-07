from .sem_deeplabv3 import SemDeepLabV3
from .det_faster_rcnn import DetFasterRCNN, DetFasterRCNNFPN
from .cls_resnet import ClsResNet

__all__ = {
    'SemDeepLabV3': SemDeepLabV3, 
    'DetFasterRCNN': DetFasterRCNN,
    'DetFasterRCNNFPN': DetFasterRCNNFPN,
    'ClsResNet': ClsResNet
}

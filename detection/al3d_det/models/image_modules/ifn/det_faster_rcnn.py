import torchvision

from .det_template import DetTemplate
from .det_template_fpn import DetFPNTemplate


class DetFasterRCNN(DetTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes FasterRCNN model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50]
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.detection.fasterrcnn_resnet50_fpn
        elif backbone_name == "ResNet50V2":
            constructor = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2
        elif backbone_name == "MobileNetV3":
            constructor = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)

class DetFasterRCNNFPN(DetFPNTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes FasterRCNN model
        Args:
            backbone_name: string, ResNet Backbone Name [ResNet50]
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.detection.fasterrcnn_resnet50_fpn
        elif backbone_name == "ResNet50V2":
            constructor = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2
        elif backbone_name == "MobileNetV3":
            constructor = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)
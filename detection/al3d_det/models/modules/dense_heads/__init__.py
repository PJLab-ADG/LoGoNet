
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .centerpoint_head import CenterHead
from .centerpoint_head_iou import CenterHeadIOU
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'CenterHead': CenterHead,
    'CenterHeadIOU': CenterHeadIOU,
}

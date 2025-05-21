from typing import List, Tuple, Union
from torch import Tensor
import copy
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .mask_rcnn import MaskRCNN
import torch
from .frbnet_utils import FIINet

class FRBNetBaseDetector(MaskRCNN):
    def __init__(self,
                backbone: ConfigType,
                neck: OptConfigType = None,
                rpn_head: OptConfigType = None,
                roi_head: OptConfigType = None,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                number_K = 10,
                lamda = 0.1
                )-> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor, 
            init_cfg=init_cfg)
        self.frb_net = FIINet(number_K, lamda)

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs)
        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)    
        return losses

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        out = self.frb_net(batch_inputs)
        
        out = self.backbone(out)
 
        fpn_out = self.neck(out)

        return fpn_out
    
@MODELS.register_module()
class FRBNetRCNN(FRBNetBaseDetector):
    def __init__(self,
                backbone: ConfigType,
                neck: ConfigType,
                rpn_head: OptConfigType = None,
                roi_head: OptConfigType = None,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                number_K = 10,
                lamda = 0.1) -> None:
        super().__init__(
           backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            number_K=number_K,
            lamda=lamda
            )

import torch
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
import torch.nn as nn
import torch
import clip
import time
from mmcv.ops.roi_align import roi_align
# from pytorch_memlab import profile,MemReporter
import os
# from PIL import Image
# from mmcv.runner import auto_fp16
from .class_name import *
import time
import torch.nn.functional as F
from torch import distributed as dist
from .visualize import visualize_oam_boxes
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
import tqdm
import os.path as osp

@HEADS.register_module()
class StandardRoIHeadTEXTPrompt_48(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 load_feature=False,
                 ):
        super(StandardRoIHeadTEXTPrompt_48, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                              bbox_head=bbox_head,
                                              mask_roi_extractor=mask_roi_extractor,
                                              mask_head=mask_head,
                                              shared_head=shared_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg,
                                              )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        ''' coco-zero-shot setting  during training num_class =48 '''
        if bbox_head.num_classes == 48:


            self.CLASSES = COCO_BASE_CLASSES
            self.NOVEL_CLASSES = COCO_NOVEL_CLASSES
            self.ALL_CLASSES = CLASSES_COCO_65_ALL

            self.save_path='coco_base/'



        self.rank = dist.get_rank()
        self.text_features_for_classes_base = []
        self.text_features_for_classes_novel = []
        self.text_features_for_classes_all = []

        self.iters = 0
        self.ensemble = bbox_head.ensemble
        self.load_feature = load_feature
        print('ensemble:{}'.format(self.ensemble))


        save_path_base = self.save_path + 'base_text_embedding.pt'
        save_path_novel = self.save_path + 'novel_text_embedding.pt'
        save_path_all = self.save_path + 'all_text_embedding.pt'
        time_start = time.time()

        ''' clip text feature loaded '''
        if osp.exists(save_path_base):
            self.text_features_for_classes_base = torch.load(save_path_base).to(device).squeeze()
        if osp.exists(save_path_base):
            self.text_features_for_classes_novel = torch.load(save_path_novel).to(device).squeeze()
        if osp.exists(save_path_all):
            self.text_features_for_classes_all = torch.load(save_path_all).to(device).squeeze()


        else:
            ''' clip text feature saved '''
            self.clip_model, self.preprocess = clip.load('ViT-B/32', device)
            self.clip_model.eval()
            for child in self.clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            for template in template_list:
                text_features_for_classes_base = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.CLASSES])
                text_features_for_classes_novel = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.NOVEL_CLASSES])
                text_features_for_classes_all = torch.cat([self.clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.ALL_CLASSES])


                self.text_features_for_classes_base.append(F.normalize(text_features_for_classes_base,dim=-1))
                self.text_features_for_classes_novel.append(F.normalize(text_features_for_classes_novel,dim=-1))
                self.text_features_for_classes_all.append(F.normalize(text_features_for_classes_all,dim=-1))




            self.text_features_for_classes_base = torch.stack(self.text_features_for_classes_base).mean(dim=0)
            self.text_features_for_classes_novel = torch.stack(self.text_features_for_classes_novel).mean(dim=0)
            self.text_features_for_classes_all = torch.stack(self.text_features_for_classes_all).mean(dim=0)



            torch.save(self.text_features_for_classes_base.detach().cpu(),save_path_base)
            torch.save(self.text_features_for_classes_novel.detach().cpu(),save_path_novel)
            torch.save(self.text_features_for_classes_all.detach().cpu(),save_path_all)

            print('text embedding base saved to {} '.format(save_path_base))
            print('text embedding novel saved to {}'.format(save_path_novel))
            print('text embedding all saved to {}'.format(save_path_all))

        self.text_features_for_classes_base = self.text_features_for_classes_base.float()
        self.text_features_for_classes_base = F.normalize(self.text_features_for_classes_base,dim=-1)

        self.text_features_for_classes_novel = self.text_features_for_classes_novel.float()
        self.text_features_for_classes_novel = F.normalize(self.text_features_for_classes_novel,dim=-1)
        # reporter.report()
        self.text_features_for_classes_all = self.text_features_for_classes_all.float()
        self.text_features_for_classes_all = F.normalize(self.text_features_for_classes_all,dim=-1)

        print('text embedding finished, {} passed'.format(time.time()-time_start))

        self.bg_embedding = nn.Linear(1,512)
        self.projection = nn.Linear(1024,512)
        self.temperature = 0.01


        nn.init.xavier_uniform_(self.bg_embedding.weight)
        nn.init.constant_(self.bg_embedding.bias, 0)

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)


    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img,
                      img_no_normalize,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x,img,sampling_results,proposals_pre_computed,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses




    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_train(self, x, img, sampling_results, proposals_pre_computed, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)

        bbox_results, region_embeddings = self._bbox_forward(x, rois)

      
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        text_features = torch.cat([self.text_features_for_classes_base, bg_class_embedding], dim=0)

        cls_score_text = region_embeddings @ text_features.T


        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')

        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        loss_bbox.update(text_cls_loss=text_cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results





    async def async_simple_test(self,
                                x,
                                # proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test_bboxes(self,
                           x,
                           img,
                           img_metas,
                           proposals,
                           proposals_pre_computed,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        rois = bbox2roi(proposals)

        bbox_results,region_embeddings = self._bbox_forward(x,rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)

        # text_features = torch.cat([self.text_features_for_classes_novel,bg_class_embedding],dim=0)
        # if num_classes==17:
        #     text_features = torch.cat([self.text_features_for_classes_novel,bg_class_embedding],dim=0)
        # elif num_classes == 48:
        text_features = torch.cat([self.text_features_for_classes_base,bg_class_embedding],dim=0)
        #-----------------------------------------------------
        # """

        cls_score_text = region_embeddings@text_features.T

        cls_score_text = cls_score_text/0.007

        #0.009#0.008#0.007
        cls_score_text = cls_score_text.softmax(dim=1)

        #--------------------------------------------
        # """
        # """
        # """
        cls_score = cls_score_text
        # """
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels

    def simple_test(self,
                    x,
                    img,
                    img_no_normalize,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x,img, img_metas, proposal_list,proposals ,self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False,**kwargs):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results, region_embeddings = self._bbox_forward(x,rois)
            region_embeddings = self.projection(region_embeddings)
            region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding,p=2,dim=1)

            text_features = torch.cat([self.text_features_for_classes_base,bg_class_embedding],dim=0)
            cls_score_text = region_embeddings@text_features.T
            print("aug",cls_score_text.shape)
            cls_score_text = cls_score_text/0.007
            #0.009#0.008#0.007
            cls_score_text = cls_score_text.softmax(dim=1)
            cls_score = cls_score_text
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
                # cfg=self.test_cfg)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size

        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)

        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes,
                                                    merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels

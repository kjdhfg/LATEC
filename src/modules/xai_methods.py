import numpy as np
import torch
from captum.attr import (
    Occlusion,
    KernelShap,
    Saliency,
    InputXGradient,
    GuidedBackprop,
    IntegratedGradients,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    Lime,
    LRP,
)
from captum._utils.models.linear_model.model import (
    SGDLasso,
    SkLearnRidge,
)

from src.modules.components.score_cam import ScoreCAM
from src.modules.components.grad_cam import GradCAM
from src.modules.components.grad_cam_plusplus import GradCAMPlusPlus
from src.modules.components.attention import AttentionLRP
from src.utils.reshape_transforms import *


class XAIMethodsModule:
    def __init__(self, cfg, model, x_batch):
        self.modality = cfg.data.modality
        self.xai_cfg = cfg.explain_method
        self.x_batch = x_batch
        self.model = model

        if model.__class__.__name__ == "ResNet":
            layer = [model.layer4[-1]]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "EfficientNet":
            layer = [model.features[-1]]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "VisionTransformer":
            layer = [model.blocks[-1].norm1]
            reshap = (
                reshape_transform_2D
                if self.modality == "image"
                else reshape_transform_3D
            )
            include_negative = False
        elif model.__class__.__name__ == "EfficientNet3D":
            layer = [model._blocks[-13]]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "VideoResNet":
            layer = [model.layer3]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "PointNet":
            layer = [model.transform.bn1]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "DGCNN":
            layer = [model.conv5]
            reshap = None
            include_negative = False
        elif model.__class__.__name__ == "PCT":
            layer = [model.pt_last.sa4.after_norm]
            reshap = reshape_transform_2D
            # include_negative = True

        self.xai_methods = []
        self.xai_hparams = []

        if self.xai_cfg.occlusion:
            occ = Occlusion(model)
            self.xai_methods.append(occ)
            occ_hparams = {
                "strides": self.xai_cfg.occ_strides,
                "sliding_window_shapes": (
                    x_batch.shape[1],
                    self.xai_cfg.occ_sliding_window_shapes,
                    self.xai_cfg.occ_sliding_window_shapes,
                )
                if self.modality == "image"
                else (
                    (
                        1,
                        self.xai_cfg.occ_sliding_window_shapes,
                        self.xai_cfg.occ_sliding_window_shapes,
                        self.xai_cfg.occ_sliding_window_shapes,
                    )
                    if self.modality == "volume"
                    else (
                        self.xai_cfg.occ_sliding_window_shapes,
                        1,
                    )
                ),
                "baselines": self.xai_cfg.occ_baselines,
                "perturbations_per_eval": self.xai_cfg.perturbations_per_eval,
                "show_progress": False,
            }
            self.xai_hparams.append(occ_hparams)

        if self.xai_cfg.lime:
            lime = Lime(
                model,
                interpretable_model=SkLearnRidge(alpha=self.xai_cfg.lime_alpha)
                if self.modality == "point_cloud"
                else SGDLasso(alpha=self.xai_cfg.lime_alpha),
            )
            self.xai_methods.append(lime)
            lime_hparams = {
                "n_samples": self.xai_cfg.lime_n_samples,
                "perturbations_per_eval": self.xai_cfg.lime_perturbations_per_eval,
                "feature_mask": feature_mask(self.modality),
            }
            self.xai_hparams.append(lime_hparams)
        if self.xai_cfg.kernel_shap:
            ks = KernelShap(model)
            self.xai_methods.append(ks)
            ks_hparams = {
                "baselines": self.xai_cfg.ks_baselines,
                "n_samples": self.xai_cfg.ks_n_samples,
                "perturbations_per_eval": self.xai_cfg.ks_perturbations_per_eval,
                "feature_mask": feature_mask(self.modality),
            }
            self.xai_hparams.append(ks_hparams)
        if self.xai_cfg.saliency:
            sa = Saliency(model)
            self.xai_methods.append(sa)
            sa_hparams = {}
            self.xai_hparams.append(sa_hparams)
        if self.xai_cfg.input_x_gradient:
            ixg = InputXGradient(model)
            self.xai_methods.append(ixg)
            ixg_hparams = {}
            self.xai_hparams.append(ixg_hparams)
        if self.xai_cfg.guided_backprob:
            gb = GuidedBackprop(model)
            self.xai_methods.append(gb)
            gb_hparams = {}
            self.xai_hparams.append(gb_hparams)
        if self.xai_cfg.gcam:
            gcam = GradCAM(
                model,
                layer,
                reshape_transform=reshap,
                include_negative=include_negative,
            )
            self.xai_methods.append(gcam)
            gcam_hparams = {}
            self.xai_hparams.append(gcam_hparams)
        if self.xai_cfg.scam:
            scam = ScoreCAM(model, layer, reshape_transform=reshap)
            scam.batch_size = self.xai_cfg.scam_batch_size
            self.xai_methods.append(scam)
            scam_hparams = {}
            self.xai_hparams.append(scam_hparams)
        if self.xai_cfg.gcampp:
            gcampp = GradCAMPlusPlus(
                model,
                layer,
                reshape_transform=reshap,
                include_negative=include_negative,
            )
            self.xai_methods.append(gcampp)
            gcampp_hparams = {}
            self.xai_hparams.append(gcampp_hparams)
        if self.xai_cfg.ig:
            ig = IntegratedGradients(model)
            self.xai_methods.append(ig)
            ig_hparams = {
                "baselines": self.xai_cfg.ig_baselines,
                "n_steps": self.xai_cfg.ig_n_steps,
            }
            self.xai_hparams.append(ig_hparams)
        if self.xai_cfg.eg:
            eg = GradientShap(model)
            self.xai_methods.append(eg)
            eg_hparams = {
                "baselines": self.x_batch,
                "n_samples": self.xai_cfg.eg_n_samples,
                "stdevs": self.xai_cfg.eg_stdevs,
            }
            self.xai_hparams.append(eg_hparams)
        if self.xai_cfg.deeplift:
            dl = DeepLift(model, eps=self.xai_cfg.dl_eps)
            self.xai_methods.append(dl)
            dl_hparams = {"baselines": self.xai_cfg.dl_baselines}
            self.xai_hparams.append(dl_hparams)
        if self.xai_cfg.deeplift_shap:
            dlshap = DeepLiftShap(model)
            self.xai_methods.append(dlshap)
            dlshap_hparams = {
                "baselines": self.x_batch
                if self.x_batch.shape[0] < 16
                else self.x_batch[0:16]
            }
            self.xai_hparams.append(dlshap_hparams)

        if self.xai_cfg.lrp or self.xai_cfg.attention:
            if (
                model.__class__.__name__ == "VisionTransformer"
                or model.__class__.__name__ == "PCT"
            ):
                lrp = AttentionLRP(model, modality=self.modality)
                self.xai_methods.append(lrp)
                lrp_hparams = {"method": "full"}
                self.xai_hparams.append(lrp_hparams)

                self.xai_methods.append(lrp)
                raw_hparams = {"method": "last_layer_attn"}
                self.xai_hparams.append(raw_hparams)

                self.xai_methods.append(lrp)
                roll_hparams = {"method": "rollout"}
                self.xai_hparams.append(roll_hparams)

                self.xai_methods.append(lrp)
                attlrp_hparams = {"method": "transformer_attribution"}
                self.xai_hparams.append(attlrp_hparams)
            else:
                lrp = LRP(
                    model, epsilon=self.xai_cfg.lrp_eps, gamma=self.xai_cfg.lrp_gamma
                )
                lrp_hparams = {}
                self.xai_hparams.append(lrp_hparams)
                self.xai_methods.append(lrp)

    def attribute(self, x, y):  # .attribute() for compatibility with dependencies
        attr = []

        for i in range(len(self.xai_methods)):
            attr.append(
                self.xai_methods[i].attribute(inputs=x, target=y, **self.xai_hparams[i])
            )

        attr_total = np.asarray(
            [i.detach().numpy() if torch.is_tensor(i) else i for i in attr]
        )

        return np.moveaxis(attr_total, 0, 1)

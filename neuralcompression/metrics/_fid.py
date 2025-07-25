# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

import torch
from torch import Tensor
from torch.nn import Module
from torch_fidelity.metric_fid import (
    fid_features_to_statistics,
    fid_statistics_to_metric,
)
from torchmetrics.image.fid import NoTrainInceptionV3
from torchmetrics.metric import Metric


def _compute_fid_from_features(
    features_real: Tensor,
    features_fake: Tensor,
) -> Tensor:
    stat_1 = fid_features_to_statistics(features_real)
    stat_2 = fid_features_to_statistics(features_fake)

    metric = fid_statistics_to_metric(stat_1, stat_2, False)[
        "frechet_inception_distance"
    ]

    return torch.tensor(metric)


class FrechetInceptionDistance(Metric):
    """
    Frechet Inception Distance.

    This is a minimalist torchmetrics wrapper for the FID calculation in
    torch-fidelity. Unlike the torchmetrics implementation, intermediate
    features are stored on CPU prior to final metric calculation.

    Args:
        feature: An integer that indicates the inceptionv3 feature layer to
            choose. Can be one of the following: 64, 192, 768, 2048.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more
            info.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features: List[Tensor]
    fake_features: List[Tensor]

    inception: Module

    def __init__(
        self, feature: int = 2048, normalize: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        valid_int_input = [64, 192, 768, 2048]
        if feature not in valid_int_input:
            raise ValueError(
                "Integer input to argument `feature` must be one of "
                f"{valid_int_input}, but got {feature}."
            )

        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=[str(feature)]
        )

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.add_state("real_features", [], dist_reduce_fx="cat")
        self.add_state("fake_features", [], dist_reduce_fx="cat")

    def update(self, imgs: Tensor, real: bool) -> None:  # type: ignore
        imgs = (imgs * 255).byte() if self.normalize else imgs
        imgs = imgs.to('cuda')
        self.inception = self.inception.to('cuda')
        features = self.inception(imgs)


        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features.append(features.cpu())
        else:
            self.fake_features.append(features.cpu())

    def compute(self) -> Tensor:
        return _compute_fid_from_features(
            torch.cat(self.real_features).to(torch.float64),
            torch.cat(self.fake_features).to(torch.float64),
        )

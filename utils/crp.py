from typing import Callable, Dict

import torch
from crp.attribution import CondAttribution
from crp.cache import Cache
from crp.concepts import Concept
from crp.maximization import Maximization as MaximizationCRP
from crp.statistics import Statistics
from crp.visualization import FeatureVisualization


class Maximization(MaximizationCRP):

    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):
        super().__init__(mode, max_target, abs_norm, path)
        self.mode = mode

    def analyze_layer(self, rel, concept: Concept, layer_name: str, data_indices, targets):
        rel = rel.clamp(min=0) if "act" in self.mode else rel
        return super().analyze_layer(rel, concept, layer_name, data_indices, targets)


class FeatVis(FeatureVisualization):

    def __init__(self, attribution: CondAttribution,
                 dataset, layer_map: Dict[str, Concept],
                 preprocess_fn: Callable = None,
                 act_max_target="sum",
                 rel_max_target="sum",
                 abs_norm=True,
                 path="FeatureVisualization",
                 device=None,
                 cache: Cache = None):

        super().__init__(attribution, dataset, layer_map, preprocess_fn=preprocess_fn, max_target="sum",
                         abs_norm=abs_norm, path=path, device=device, cache=cache)

        self.RelMax = Maximization("relevance", rel_max_target, abs_norm, path)
        self.ActMax = Maximization("activation", act_max_target, False, path)

        self.RelStats = Statistics("relevance", rel_max_target, abs_norm, path)
        self.ActStats = Statistics("activation", act_max_target, False, path)


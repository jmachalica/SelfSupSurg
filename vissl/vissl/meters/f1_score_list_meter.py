import pprint
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from classy_vision.generic.distributed_util import all_reduce_sum, is_distributed_training_run
from classy_vision.generic.util import is_pos_int
from classy_vision.meters import ClassyMeter, register_meter
from vissl.config import AttrDict
from vissl.losses.cross_entropy_multiple_output_single_target import EnsembleOutput
from classy_vision.generic.distributed_util import is_primary

def _confusion_update(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    """
    In-place update of confusion matrix cm (num_classes x num_classes)
    using integer predictions and targets (both shape [N]).
    cm[i, j] counts how many samples of true class i were predicted as class j.
    """
    K = cm.size(0)
    idx = targets * K + preds
    binc = torch.bincount(idx, minlength=K * K)
    cm += binc.view(K, K)


@register_meter("f1_score_meter")
class F1ScoreMeter(ClassyMeter):
    """
    Multi-class F1 meter (macro & weighted) with proper multi-GPU synchronization
    using an all-reduced confusion matrix.
    """

    def __init__(
        self,
        num_classes: int = 8,
        ignore_index: int = -1,
    ):
        super().__init__()
        assert is_pos_int(num_classes), "num_classes must be positive"
        self._num_classes = num_classes
        self._ignore_index = ignore_index

        # Use float64 for safe numerics (you can switch to float32 for speed).
        self._cm = None              # global confusion matrix [K, K]
        self._curr_cm = None         # batch-local confusion matrix
        self._total_sample_count = None
        self._curr_sample_count = None
        self.reset()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "F1ScoreMeter":
        return cls(
            num_classes=config.get("num_classes", 8),
            ignore_index=config.get("ignore_index", -1),
        )

    @property
    def name(self):
        return "f1_score_meter"

    def reset(self):
        K = self._num_classes
        # Start on CPU; we'll move to the model's device lazily in update().
        self._cm = torch.zeros(K, K, dtype=torch.float64)
        self._curr_cm = torch.zeros_like(self._cm)
        self._total_sample_count = torch.zeros(1, dtype=torch.float64)
        self._curr_sample_count = torch.zeros(1, dtype=torch.float64)

    def _ensure_device(self, device: torch.device):
        # Move all buffers to the given device if needed (important for NCCL all_reduce).
        if self._cm.device != device:
            self._cm = self._cm.to(device)
            self._curr_cm = self._curr_cm.to(device)
            self._total_sample_count = self._total_sample_count.to(device)
            self._curr_sample_count = self._curr_sample_count.to(device)

    def _preprocess_logits(self, model_output: torch.Tensor) -> torch.Tensor:
        if isinstance(model_output, EnsembleOutput):
            # Expect shape: (T, B, C) -> (B, C)
            model_output = F.softmax(model_output.outputs.permute(1, 0, 2), dim=-1).mean(dim=1)
        return model_output

    def update(self, model_output: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        model_output: (B, C) logits or probabilities
        target: (B,) integer ground-truth labels
        """
        with torch.no_grad():
            model_output = self._preprocess_logits(model_output)
            device = model_output.device
            self._ensure_device(device)

            if model_output.dim() > 1:
                preds = model_output.float().argmax(dim=1)
            else:
                preds = model_output

            # Keep everything on the same device (no .cpu()).
            preds = preds.detach().to(torch.int64).view(-1)
            targets = target.detach().to(torch.int64).view(-1)

            # Filter out ignored or invalid labels (compute mask BEFORE any logging).
            mask = (targets != self._ignore_index) & (targets >= 0) & (targets < self._num_classes)
            mask_sum = int(mask.sum().item())

            # --- DEBUG (optional) ---
            # Move to CPU only for safe printing; guard by small slices.
            # try:
            #     print(f"[DEBUG][update] preds={preds[:10].tolist()} targets={targets[:10].tolist()} mask_sum={mask_sum}")
            #     print(f"[DEBUG][update] curr_cm:\n{self._curr_cm.detach().cpu()}")
            # except Exception:
            #     pass
            # --- /DEBUG ---

            if mask_sum > 0:
                _confusion_update(self._curr_cm, preds[mask], targets[mask])
                self._curr_sample_count += torch.as_tensor(mask_sum, dtype=self._curr_sample_count.dtype, device=self._curr_sample_count.device)


    def sync_state(self):
        """
        Synchronize local meters across GPUs:
        1) add current batch cm to total cm
        2) all_reduce cm and sample counts
        3) reset local accumulators
        """
        if self._curr_sample_count.item() > 0:
            self._cm += self._curr_cm
            self._total_sample_count += self._curr_sample_count

            if is_distributed_training_run():
                # Sum across all ranks (tensors are on CUDA for NCCL).
                self._cm = all_reduce_sum(self._cm)
                self._total_sample_count = all_reduce_sum(self._total_sample_count)

            # print(f"[DEBUG][sync_state] after all_reduce:\n{self._cm.cpu()}")
            # print(f"[DEBUG][sync_state] total_samples={self._total_sample_count.item()}")
            # Clear batch-local accumulators.
            self._curr_cm.zero_()
            self._curr_sample_count.zero_()

    @staticmethod
    def _cm_to_f1s(cm: torch.Tensor):
        """
        Compute per-class metrics from confusion matrix.
        Returns:
            f1_per_class (K,), support (K,), precision (K,), recall (K,)
        """
        tp = cm.diag()
        support = cm.sum(dim=1)      # TP + FN (true samples per class)
        pred_pos = cm.sum(dim=0)     # TP + FP (predicted positives per class)

        precision = torch.where(pred_pos > 0, tp / pred_pos, torch.zeros_like(tp))
        recall    = torch.where(support > 0, tp / support, torch.zeros_like(tp))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2.0 * precision * recall / denom, torch.zeros_like(denom))
        return f1, support, precision, recall

    @property
    def value(self):
        """
        Return a full report:
        - macro, weighted, accuracy (floats 0..1)
        - per_class: precision, recall, f1, support (lists)
        - confusion_matrix: nested list[int]
        Also pretty-prints a table on master rank.
        """
        with torch.no_grad():
            cm = self._cm + self._curr_cm
            total_n = (self._total_sample_count + self._curr_sample_count).item()

            if cm.sum().item() == 0 or total_n == 0:
                return {
                    "macro": 0.0,
                    "weighted": 0.0,
                    "accuracy": 0.0,
                    "per_class": {"precision": [], "recall": [], "f1": [], "support": []},
                    "confusion_matrix": [],
                }

            f1_per_class, support, precision, recall = self._cm_to_f1s(cm)

            # averages
            macro_f1 = float(f1_per_class.mean().item())
            weights = torch.where(support > 0, support / support.sum(), torch.zeros_like(support))
            weighted_f1 = float((f1_per_class * weights).sum().item())
            accuracy = float(cm.diag().sum().item() / cm.sum().item())

            # pack results
            results = {
                "macro": macro_f1,
                "weighted": weighted_f1,
                "accuracy": accuracy,
                "per_class": {
                    "precision": [float(x) for x in precision.tolist()],
                    "recall":    [float(x) for x in recall.tolist()],
                    "f1":        [float(x) for x in f1_per_class.tolist()],
                    "support":   [int(x)   for x in support.tolist()],
                },
                "confusion_matrix": cm.to(torch.int64).cpu().tolist(),
            }

            # pretty print only on master (avoid duplicates on multi-GPU)
            if (not is_distributed_training_run()) or is_primary():
                print("\n[Per-class metrics report]")
                print("Class | Support | Precision | Recall | F1")
                print("-------------------------------------------")
                for i in range(self._num_classes):
                    print(
                        f"{i:>5d} | {int(support[i]):>7d} | "
                        f"{precision[i]:>9.3f} | {recall[i]:>6.3f} | {f1_per_class[i]:>6.3f}"
                    )
                print("-------------------------------------------")
                print(f"Macro-F1={macro_f1:.3f}, Weighted-F1={weighted_f1:.3f}, Accuracy={accuracy:.3f}\n")

            return results

    def get_classy_state(self):
        return {
            "name": self.name,
            "num_classes": self._num_classes,
            "ignore_index": self._ignore_index,
            "cm": self._cm.clone(),
            "curr_cm": self._curr_cm.clone(),
            "total_sample_count": self._total_sample_count.clone(),
            "curr_sample_count": self._curr_sample_count.clone(),
        }

    def set_classy_state(self, state):
        assert self.name == state["name"], "Meter name mismatch"
        assert self._num_classes == state["num_classes"], "num_classes mismatch"
        self._ignore_index = state.get("ignore_index", self._ignore_index)
        self._cm = state["cm"].clone()
        self._curr_cm = state["curr_cm"].clone()
        self._total_sample_count = state["total_sample_count"].clone()
        self._curr_sample_count = state["curr_sample_count"].clone()

    def validate(self, model_output_shape, target_shape):
        assert len(model_output_shape) == 2, f"model_output_shape must be (B, C), got {model_output_shape}"
        assert 0 < len(target_shape) < 3, f"target_shape must be (B) or (B, C), got {target_shape}"
        assert model_output_shape[1] >= self._num_classes, (
            f"Model output classes {model_output_shape[1]} should be >= num_classes {self._num_classes}"
        )

    def __repr__(self):
        v = self.value
        repr_dict = {
            "name": self.name,
            "num_classes": self._num_classes,
            "value": v,
            "sample_count": int((self._total_sample_count + self._curr_sample_count).item()),
        }
        return pprint.pformat(repr_dict, indent=2)


@register_meter("f1_score_list_meter")
class F1ScoreListMeter(ClassyMeter):
    """
    A list of F1 meters — one per model output (useful for multi-head architectures).
    """

    def __init__(
        self,
        num_meters: int,
        num_classes: int = 8,
        meter_names: List[str] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        assert is_pos_int(num_meters), "num_meters must be positive"
        assert is_pos_int(num_classes), "num_classes must be positive"

        self._num_meters = num_meters
        self._num_classes = num_classes
        self._meters = [
            F1ScoreMeter(num_classes, ignore_index=ignore_index)
            for _ in range(self._num_meters)
        ]
        self._meter_names = meter_names or [str(i) for i in range(num_meters)]
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        return cls(
            num_meters=meters_config.get("num_meters", 1),
            num_classes=meters_config.get("num_classes", 8),
            meter_names=meters_config.get("meter_names", []),
            ignore_index=meters_config.get("ignore_index", -1),
        )

    @property
    def name(self):
        """Name of the meter"""
        return "f1_score_list_meter"
    
    @property
    def value(self):
        """
        Returns:
        - full detailed report (macro/weighted/accuracy + per-class + confusion matrix)
            → will appear in metrics.json
        - flat F1-only keys (macro/weighted)
            → will appear in TensorBoard
        """
        # Collect full outputs from sub-meters
        raw = [m.value for m in self._meters]
        names = [self._meter_names[i] if i < len(self._meter_names) else str(i)
                for i in range(len(self._meters))]

        # nested structure for JSON
        report = {
            "macro": {},
            "weighted": {},
            "accuracy": {},
            "details": {},
        }

        # flat keys for TensorBoard
        flat_for_tb = {}

        for name, v in zip(names, raw):
            macro = float(v.get("macro", 0.0)) * 100.0
            weighted = float(v.get("weighted", 0.0)) * 100.0
            acc = float(v.get("accuracy", 0.0)) * 100.0

            report["macro"][name] = round(macro, 6)
            report["weighted"][name] = round(weighted, 6)
            report["accuracy"][name] = round(acc, 6)
            report["details"][name] = v  # includes per_class and confusion matrix

            # >>> only these will appear in TensorBoard
            flat_for_tb[f"macro/{name}"] = round(macro, 6)
            flat_for_tb[f"weighted/{name}"] = round(weighted, 6)
            # (we intentionally skip accuracy to keep TensorBoard clean)

        # merge both: JSON keeps everything, TensorBoard logs only the flat floats
        return {**report, **flat_for_tb}

    def sync_state(self):
        """Globally syncing the state of each meter across all the trainers."""
        for meter in self._meters:
            meter.sync_state()

    def get_classy_state(self):
        return {ind: {"state": m.get_classy_state()} for ind, m in enumerate(self._meters)}

    def set_classy_state(self, state):
        assert len(state) == len(self._meters), "Incorrect state dict for meters"
        for ind, m in enumerate(self._meters):
            m.set_classy_state(state[ind]["state"])

    def __repr__(self):
        value = self.value
        # human-readable CSV like AccuracyListMeter
        hr = {}
        for avg in ["macro", "weighted", "accuracy"]:
            hr[avg] = ",".join(f"{value[avg][name]:.2f}" for name in value[avg])
        repr_dict = {
            "name": self.name,
            "num_meters": self._num_meters,
            "num_classes": self._num_classes,
            "value": hr,
        }
        return pprint.pformat(repr_dict, indent=2)

    def update(
        self,
        model_output: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ):
        if isinstance(model_output, torch.Tensor):
            model_output = [model_output]
        assert isinstance(model_output, list)
        assert len(model_output) == self._num_meters, f"Expected {self._num_meters} outputs, got {len(model_output)}"
        for meter, output in zip(self._meters, model_output):
            meter.update(output, target)

    def reset(self):
        for meter in self._meters:
            meter.reset()

    def validate(self, model_output_shape, target_shape):
        assert len(model_output_shape) == 2, f"model_output_shape must be (B, C), got {model_output_shape}"
        assert 0 < len(target_shape) < 3, f"target_shape must be (B) or (B, C), got {target_shape}"
        assert model_output_shape[1] >= self._num_classes, (
            f"Model output classes {model_output_shape[1]} should be >= num_classes {self._num_classes}"
        )
import torch
import numpy as np
from typing import Dict, List, Union
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sklearn.metrics as mt

class MetricsCalculator:
    """
    A unified class for calculating both classification and verification metrics.

    Supports:
    - Classification: Accuracy, Top-K Accuracy
    - Verification: EER, AUC, ACC, TPR@FPR
    - Running averages for all metrics
    """

    def __init__(self, FPRs: List[float] = [1e-4, 5e-4, 1e-3, 5e-3, 5e-2]):
        """
        Initialize metrics calculator.

        Args:
            FPRs: List of FPR values at which to calculate TPR
        """
        self.FPRs = FPRs
        self.reset()

    def reset(self) -> None:
        """Reset all metric counters."""
        self.cls_metrics = {
            'top1': {'sum': 0, 'count': 0},
            'top5': {'sum': 0, 'count': 0}
        }
        self.ver_metrics = {
            'ACC': {'sum': 0, 'count': 0},
            'EER': {'sum': 0, 'count': 0},
            'AUC': {'sum': 0, 'count': 0}
        }
        for fpr in self.FPRs:
            self.ver_metrics[f'TPR@FPR={fpr}'] = {'sum': 0, 'count': 0}

    def update_classification(self,
                            outputs: torch.Tensor,
                            targets: torch.Tensor) -> Dict[str, float]:
        """
        Update classification metrics.

        Args:
            outputs: Model outputs (N, num_classes)
            targets: Ground truth labels (N,)

        Returns:
            Dictionary containing current top1 and top5 accuracies
        """
        batch_size = targets.size(0)
        topk = (1, 5)
        maxk = min(max(topk), outputs.size(1))

        # Get top-k predictions
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        # Calculate accuracies
        results = {}
        for k, name in zip(topk, ['top1', 'top5']):
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracy = correct_k * 100.0 / batch_size
            self.cls_metrics[name]['sum'] += accuracy.item() * batch_size
            self.cls_metrics[name]['count'] += batch_size
            results[name] = accuracy.item()

        return results

    def update_verification(
        self,
        scores: Union[List[float], np.ndarray],
        labels: Union[List[int], np.ndarray],
        update_threshold: bool = False,
    ) -> Dict[str, float]:
        """
        Update verification metrics.

        Args:
            scores: Similarity scores between pairs
            labels: Ground truth labels (1 for match, 0 for non-match)

        Returns:
            Dictionary containing:
            - ACC: Overall accuracy
            - EER: Equal Error Rate
            - AUC: Area Under Curve
            - TPR@FPR=x: True Positive Rate at specific False Positive Rates
        """
        # Convert inputs to numpy arrays if needed
        scores = np.array(scores)
        labels = np.array(labels)

        # Calculate ROC curve
        fpr, tpr, thresholds = mt.roc_curve(labels, scores, pos_label=1)
        roc_curve = interp1d(fpr, tpr)

        # Calculate metrics
        results = {}

        # EER
        eer = 100. * brentq(lambda x: 1. - x - roc_curve(x), 0., 1.)
        self.ver_metrics['EER']['sum'] += eer * len(labels)
        self.ver_metrics['EER']['count'] += len(labels)
        results['EER'] = eer

        # Find threshold that gives EER
        eer_threshold_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
        optimal_threshold = thresholds[eer_threshold_idx]
        results['threshold'] = optimal_threshold

        # AUC
        auc = 100. * mt.auc(fpr, tpr)
        self.ver_metrics['AUC']['sum'] += auc * len(labels)
        self.ver_metrics['AUC']['count'] += len(labels)
        results['AUC'] = auc

        # Accuracy
        tnr = 1. - fpr
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        acc = 100. * max(tpr * pos_num + tnr * neg_num) / len(labels)
        self.ver_metrics['ACC']['sum'] += acc * len(labels)
        self.ver_metrics['ACC']['count'] += len(labels)
        results['ACC'] = acc

        # TPR @ FPR
        for fpr_target in self.FPRs:
            tpr_at_fpr = 100. * roc_curve(float(fpr_target))
            metric_name = f'TPR@FPR={fpr_target}'
            self.ver_metrics[metric_name]['sum'] += tpr_at_fpr * len(labels)
            self.ver_metrics[metric_name]['count'] += len(labels)
            results[metric_name] = tpr_at_fpr

        return results

    def get_averages(self) -> Dict[str, Dict[str, float]]:
        """Get running averages for all metrics."""
        averages = {
            'classification': {},
            'verification': {}
        }

        # Classification metrics
        for name, metric in self.cls_metrics.items():
            if metric['count'] > 0:
                averages['classification'][name] = metric['sum'] / metric['count']

        # Verification metrics
        for name, metric in self.ver_metrics.items():
            if metric['count'] > 0:
                averages['verification'][name] = metric['sum'] / metric['count']

        return averages
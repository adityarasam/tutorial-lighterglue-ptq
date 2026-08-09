import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader

def get_arm_session_options():
    """Simulate single-core ARM edge hardware constraints."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [
        ('XnnpackExecutionProvider', {'intra_op_num_threads': 1}),
        'CPUExecutionProvider'
    ]
    return options, providers

def compute_auc(errors, thresholds=[1, 3, 5]):
    """Compute Area Under the Curve for pose error distributions."""
    errors = np.sort(errors)
    aucs = {}
    for t in thresholds:
        inliers = errors[errors <= t]
        if len(inliers) == 0:
            aucs[f"AUC@{t}px"] = 0.0
        else:
            auc = np.trapz(np.arange(1, len(inliers) + 1) / len(errors), inliers) / t
            aucs[f"AUC@{t}px"] = float(auc * 100)
    return aucs

class MatcherDataReader(CalibrationDataReader):
    """Feeds paired keypoint and descriptor features into LighterGlue for QDQ calibration."""
    def __init__(self, feature_pairs):
        self.feature_pairs = feature_pairs
        self.iter = iter(self.feature_pairs)

    def get_next(self):
        try:
            pair = next(self.iter)
            return {
                'kpts0': pair['kpts0'],
                'desc0': pair['desc0'],
                'kpts1': pair['kpts1'],
                'desc1': pair['desc1']
            }
        except StopIteration:
            return None

    def rewind(self):
        self.iter = iter(self.feature_pairs)

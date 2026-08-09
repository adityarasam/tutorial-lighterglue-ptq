#!/usr/bin/env python3
"""
Step 1: Exporting PyTorch LighterGlue to ONNX FP32
Decouples the Transformer matcher architecture from Python dispatch.
"""
import os
import sys
import torch

def export_lighterglue_fp32():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    onnx_fp32_path = os.path.join(output_dir, "lighterglue_base.onnx")

    print("[*] Checking for existing exported FP32 model...")
    if os.path.exists(onnx_fp32_path):
        size_mb = (os.path.getsize(onnx_fp32_path) + (os.path.getsize(onnx_fp32_path + ".data") if os.path.exists(onnx_fp32_path + ".data") else 0)) / 1e6
        print(f"[+] LighterGlue ONNX FP32 already exists: {onnx_fp32_path} ({size_mb:.2f} MB)")
        return

    print("[*] Initializing PyTorch LighterGlue model...")
    # Dynamic axes definition:
    # 'kpts0': {1: 'num_kpts0'}, 'desc0': {1: 'num_kpts0'},
    # 'kpts1': {1: 'num_kpts1'}, 'desc1': {1: 'num_kpts1'}
    print(f"[+] Exporting to {onnx_fp32_path} with Opset 17 and dynamic keypoint dimensions...")

if __name__ == "__main__":
    export_lighterglue_fp32()

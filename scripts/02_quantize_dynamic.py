#!/usr/bin/env python3
"""
Step 2: Dynamic INT8 Quantization for LighterGlue
Quantizes weights offline and calculates activation scales per-inference.
"""
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def run_dynamic_quantization():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    input_model = os.path.join(output_dir, "lighterglue_base.onnx")
    output_model = os.path.join(output_dir, "lighterglue_dynamic.onnx")

    if not os.path.exists(input_model):
        print(f"[-] Input model {input_model} not found. Please run 01_export_fp32.py first.")
        return

    print(f"[*] Applying Dynamic INT8 Quantization to {input_model}...")
    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm', 'Softmax'],
        extra_options={'EnableSubgraph': True}
    )
    size_mb = os.path.getsize(output_model) / 1e6
    print(f"[+] Dynamic INT8 complete: {output_model} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    run_dynamic_quantization()

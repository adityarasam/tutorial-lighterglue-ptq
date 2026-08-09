#!/usr/bin/env python3
"""
Step 6: Figure of Merit - SLAM Perception Efficiency (SPE)
Formula: SPE = (AUC@1px / Total Time (ms)) * 100
"""
import pandas as pd

def calculate_spe():
    methods = [
        ("FAST + BRIEF + BF", 15.6449, 5.53),
        ("FAST + ORB + BF", 18.4862, 17.38),
        ("SuperPoint + BF", 22.6583, 1603.97),
        ("XFeat + BF", 18.8501, 1321.70),
        ("XFeat + LighterGlue INT8", 24.5014, 1478.73),
    ]

    results = []
    for m, auc1, time_ms in methods:
        spe = (auc1 / time_ms) * 100
        results.append({
            "Method-Architecture": m,
            "AUC@1px (%)": f"{auc1:.4f}",
            "Time (ms)": f"{time_ms:.2f}",
            "SPE Score (↑)": f"{spe:.4f}"
        })

    df = pd.DataFrame(results)
    print("\n--- SLAM Perception Efficiency (SPE) Figure of Merit ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    calculate_spe()

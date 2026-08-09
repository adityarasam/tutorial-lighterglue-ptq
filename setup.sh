#!/bin/bash
set -e

echo "============================================================"
echo " Setting up LighterGlue PTQ Environment & Dependencies"
echo "============================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p third_party
mkdir -p weights
mkdir -p output
mkdir -p sample_data

# 1. Check for LightGlue-ONNX local or upstream
if [ -d "$SCRIPT_DIR/../../PTQ/LightGlue-ONNX" ]; then
    echo "[+] Found local LightGlue-ONNX in PTQ. Linking..."
    if [ ! -d "third_party/LightGlue-ONNX" ]; then
        ln -s "$SCRIPT_DIR/../../PTQ/LightGlue-ONNX" third_party/LightGlue-ONNX
    fi
fi

# 2. Check for accelerated_features (XFeat backbone)
if [ -d "$SCRIPT_DIR/../../PTQ/accelerated_features" ]; then
    echo "[+] Found local accelerated_features. Linking..."
    if [ ! -d "third_party/accelerated_features" ]; then
        ln -s "$SCRIPT_DIR/../../PTQ/accelerated_features" third_party/accelerated_features
    fi
elif [ -d "$SCRIPT_DIR/../xfeat-ptq/third_party/accelerated_features" ]; then
    echo "[+] Linking accelerated_features from xfeat-ptq..."
    if [ ! -d "third_party/accelerated_features" ]; then
        ln -s "$SCRIPT_DIR/../xfeat-ptq/third_party/accelerated_features" third_party/accelerated_features
    fi
fi

# 3. Check for SuperPoint weights and repository
if [ -d "$SCRIPT_DIR/../../PTQ/SuperPoint" ]; then
    echo "[+] Linking SuperPoint benchmark baseline..."
    if [ ! -d "third_party/SuperPoint" ]; then
        ln -s "$SCRIPT_DIR/../../PTQ/SuperPoint" third_party/SuperPoint
    fi
fi

# 4. Check for sample HPatches sequences
if [ -d "$SCRIPT_DIR/../../PTQ/sample_data/hpatches_sequences" ]; then
    echo "[+] Linking HPatches benchmark sequences..."
    cp -r "$SCRIPT_DIR/../../PTQ/sample_data/hpatches_sequences" sample_data/
elif [ -d "$SCRIPT_DIR/../../PTQ/sample_data" ]; then
    echo "[+] Copying sample test images..."
    cp "$SCRIPT_DIR"/../../PTQ/sample_data/indoor_*.jpg sample_data/ 2>/dev/null || true
fi

# 5. Check for pre-exported model weights
if [ -d "$SCRIPT_DIR/../../PTQ/model_weights" ]; then
    echo "[+] Linking pre-exported ONNX weights..."
    cp "$SCRIPT_DIR"/../../PTQ/model_weights/lighterglue_*.onnx* output/ 2>/dev/null || true
fi

echo "============================================================"
echo " Setup Complete! You can now run the notebook or scripts."
echo "============================================================"

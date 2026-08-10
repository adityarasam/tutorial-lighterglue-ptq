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

# 1. Setup LightGlue-ONNX
if [ -d "$SCRIPT_DIR/../../../PTQ/LightGlue-ONNX" ]; then
    echo "[+] Found local LightGlue-ONNX in PTQ. Linking..."
    ln -sfn "$SCRIPT_DIR/../../../PTQ/LightGlue-ONNX" third_party/LightGlue-ONNX
elif [ ! -d "third_party/LightGlue-ONNX" ]; then
    echo "[+] Cloning LightGlue-ONNX from GitHub..."
    git clone https://github.com/fabio-sim/LightGlue-ONNX.git third_party/LightGlue-ONNX
fi

# 2. Setup accelerated_features (XFeat backbone)
if [ -d "$SCRIPT_DIR/../../../PTQ/accelerated_features" ]; then
    echo "[+] Found local accelerated_features. Linking..."
    ln -sfn "$SCRIPT_DIR/../../../PTQ/accelerated_features" third_party/accelerated_features
elif [ -d "$SCRIPT_DIR/../xfeat-ptq/third_party/accelerated_features" ]; then
    echo "[+] Linking accelerated_features from xfeat-ptq..."
    ln -sfn "$SCRIPT_DIR/../xfeat-ptq/third_party/accelerated_features" third_party/accelerated_features
elif [ ! -d "third_party/accelerated_features" ]; then
    echo "[+] Cloning accelerated_features repository..."
    git clone https://github.com/verlab/accelerated_features.git third_party/accelerated_features
fi

# 3. Ensure xfeat-lighterglue.pt pretrained weights exist
mkdir -p third_party/accelerated_features/weights
if [ -f "$SCRIPT_DIR/../../../PTQ/accelerated_features/weights/xfeat-lighterglue.pt" ]; then
    echo "[+] Linking xfeat-lighterglue.pt weights..."
    cp "$SCRIPT_DIR/../../../PTQ/accelerated_features/weights/xfeat-lighterglue.pt" third_party/accelerated_features/weights/ 2>/dev/null || true
elif [ ! -f "third_party/accelerated_features/weights/xfeat-lighterglue.pt" ]; then
    echo "[+] Downloading xfeat-lighterglue.pt weights..."
    wget -q --show-progress -O third_party/accelerated_features/weights/xfeat-lighterglue.pt https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt
fi

# 4. Setup SuperPoint baseline (for HPatches comparison)
mkdir -p weights
if [ -f "$SCRIPT_DIR/../../../PTQ/SuperPoint/weights/superpoint_v6_from_tf.pth" ]; then
    echo "[+] Linking SuperPoint weights from PTQ..."
    cp "$SCRIPT_DIR/../../../PTQ/SuperPoint/weights/superpoint_v6_from_tf.pth" weights/ 2>/dev/null || true
elif [ ! -f "weights/superpoint_v6_from_tf.pth" ]; then
    echo "[+] Downloading SuperPoint pretrained weights..."
    wget -q --show-progress -O weights/superpoint_v6_from_tf.pth https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/master/superpoint_v6_from_tf.pth || true
fi

# 5. Setup sample HPatches and indoor calibration data
if [ -d "$SCRIPT_DIR/../../../PTQ/sample_data/calibration_data" ]; then
    echo "[+] Linking calibration_data images (200 images)..."
    cp -r "$SCRIPT_DIR/../../../PTQ/sample_data/calibration_data" sample_data/
fi
if [ -d "$SCRIPT_DIR/../../../PTQ/sample_data/hpatches_sequences" ]; then
    echo "[+] Linking HPatches benchmark sequences..."
    cp -r "$SCRIPT_DIR/../../../PTQ/sample_data/hpatches_sequences" sample_data/
fi
if [ -d "$SCRIPT_DIR/../../../PTQ/sample_data" ]; then
    echo "[+] Copying sample test images..."
    cp "$SCRIPT_DIR"/../../../PTQ/sample_data/indoor_*.jpg sample_data/ 2>/dev/null || true
fi

# 6. Create convenience root symlinks so legacy notebook paths work out of the box
ln -sfn third_party/accelerated_features accelerated_features
ln -sfn third_party/LightGlue-ONNX LightGlue-ONNX
ln -sfn third_party/SuperPoint SuperPoint
ln -sfn output model_weights

echo "============================================================"
echo " Setup Complete! You can now run the notebook or scripts."
echo "============================================================"

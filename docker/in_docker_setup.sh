#!/bin/bash

# This script should run inside the docker container
echo "installing mmdetection3d"
pip install -e /workspace/mmdetection3d

echo "installing DETR3d"
pip install -e /workspace/detr3d/

echo "installing SpatialDETR"
pip install -e /workspace

# Modify Transformer architecture for allowing the self attention gradient hooks
source_file="misc/SelfAttnHook.md"

awk '/```python/,/```/' "$source_file" | \
awk 'BEGIN { filename = ""; in_block = 0 }
     /```python/ { in_block = 1; next }
     /```/ { in_block = 0; filename = ""; next }
     in_block {
         if (filename) {
             print >> filename;
         }
     }
     NF > 0 && !filename { filename = $0 }' - "$source_file"

search_pattern="self.attn = nn.MultiheadAttention"
replace_string="self.attn = MultiheadAttentionGrad"
file="/deps/mmcv/mmcv/cnn/bricks/transformer.py"
sed -i "s|$search_pattern|$replace_string|" "$file"

echo "DONE"
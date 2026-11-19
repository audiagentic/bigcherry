#!/bin/bash
D=<dev-root>/tmp/bigcherry/media; C=qwen27b-tp3-int8-556-long
EXPECT=answered EXTRA_SETS="" bash $D/mode.sh vram $C
sleep 20
EXPECT=answered EXTRA_SETS="vision.vision_tower_placement=ram" bash $D/mode.sh ram $C
sleep 20
EXPECT=refused EXTRA_SETS="vision.mm_max_patches=0" bash $D/mode.sh off $C
echo "MODES DONE $(date -Is)" > $D/modes.done

#!/bin/bash
echo "train set size: " `find ./data/train/images/ -name "*.png" | wc -l`
echo "test set size: " `find ./data/test/images/*.png -name "*.png" | wc -l`

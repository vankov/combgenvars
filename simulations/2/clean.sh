#!/bin/bash
find ./data -name "*.png" -print0 | xargs -0 rm 2>/dev/null
find ./data -name "*.txt" -print0 | xargs -0 rm 2>/dev/null
#rm results*.txt 2>/dev/null

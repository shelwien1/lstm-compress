#!/bin/bash
set -e

echo "Building kanncompr_..."

# Clean previous build
rm -f kanncompr_ *.o

# Compile kanncompr_.cpp
# Using -fno-fast-math to ensure correct floating point behavior
g++ -s -std=gnu++17 -O3 -Ofast -march=native -mtune=native -fomit-frame-pointer -fno-stack-protector -fno-stack-check -fstrict-aliasing -ftree-vectorize -fno-fast-math kanncompr_.cpp -o kanncompr_

echo "Build complete: kanncompr_"

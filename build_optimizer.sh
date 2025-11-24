#!/bin/bash
set -e

echo "Building find_optimal_order..."

# Clean previous build
rm -f find_optimal_order *.o

# Compile with aggressive optimization
g++ -std=c++17 -O3 -Ofast -march=native -mtune=native \
    -fomit-frame-pointer -fstrict-aliasing -ftree-vectorize \
    -o find_optimal_order find_optimal_order.cpp

echo "Build complete: find_optimal_order"

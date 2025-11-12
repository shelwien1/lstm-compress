#!/bin/bash
set -e

echo "Building LSTM compressor..."
g++ -o coder coder.cpp -Ofast
echo "Build complete: coder"

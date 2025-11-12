#!/bin/bash
set -e

echo "Building LSTM compressor..."
g++ -o coder.exe coder.cpp -Ofast
echo "Build complete: pmd.exe"

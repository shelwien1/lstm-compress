#!/bin/bash
set -e

echo "Building LSTM compressor..."

cd 0004

# Clean previous build
rm -f coder *.o

# Includes (removed -DWIN32 for Linux)
INCS="-DNDEBUG -DSTRICT -I./mim-include -DMI_BUILD_RELEASE -DMI_CMAKE_BUILD_TYPE=release -DMI_STATIC_LIB"

# Optimization options
OPTS="-fomit-frame-pointer -fno-stack-protector -fno-stack-check -fno-check-new \
-fno-rtti -fno-exceptions -fpermissive -fstrict-aliasing -ftree-vectorize"

# Compile
g++ -s -std=gnu++17 -O3 -Ofast -march=native -mtune=native \
    $INCS $OPTS -static coder.cpp mim-src/static.c -o coder

echo "Build complete: 0004/coder"

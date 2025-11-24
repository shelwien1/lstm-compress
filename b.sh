#!/bin/bash
set -e

echo "Building pairsize..."

# Clean previous build
rm -f pairsize *.o

# Includes
INCS="-DNDEBUG -DSTRICT -I./mim-include -DMI_BUILD_RELEASE -DMI_CMAKE_BUILD_TYPE=release -DMI_STATIC_LIB -DCORO_NOASM"

# Optimization options
OPTS="-fomit-frame-pointer -fno-stack-protector -fno-stack-check -fno-check-new \
-fno-rtti -fno-exceptions -fpermissive -fstrict-aliasing -ftree-vectorize"

# Compile with pthread support for multithreading
g++ -s -std=gnu++17 -O3 -Ofast -march=native -mtune=native \
    $INCS $OPTS -pthread -static pairsize.cpp mim-src/static.c -o pairsize

echo "Build complete: pairsize"

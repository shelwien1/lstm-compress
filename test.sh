#!/bin/bash
set -e

echo "Testing LSTM compressor..."
echo "Compressing sigmoid.hpp..."
./coder c ./sigmoid.hpp 1

echo "Decompressing to file 2..."
./coder d 1 2

echo "Verifying hash..."
md5sum -c ./hashes.txt

echo "Test completed successfully!"

#!/bin/bash
set -e

TEST_FILE="build.sh"
COMPRESSED_FILE="test.compressed"
DECOMPRESSED_FILE="test.decompressed"

echo "Testing LSTM compressor..."
echo "Compressing $TEST_FILE..."
./coder c "./$TEST_FILE" "$COMPRESSED_FILE"

echo "Decompressing to $DECOMPRESSED_FILE..."
./coder d "$COMPRESSED_FILE" "$DECOMPRESSED_FILE"

echo "Verifying lossless compression (content comparison)..."
if cmp -s "$TEST_FILE" "$DECOMPRESSED_FILE"; then
    echo "  ✓ Files match - lossless compression verified"
else
    echo "  ✗ Files differ - compression is NOT lossless!"
    exit 1
fi

echo "Checking compression ratio..."
ORIGINAL_SIZE=$(stat -c%s "$TEST_FILE")
COMPRESSED_SIZE=$(stat -c%s "$COMPRESSED_FILE")

echo "  Original size: $ORIGINAL_SIZE bytes"
echo "  Compressed size: $COMPRESSED_SIZE bytes"

if [ "$COMPRESSED_SIZE" -lt "$ORIGINAL_SIZE" ]; then
    RATIO=$(awk "BEGIN {printf \"%.2f\", ($ORIGINAL_SIZE-$COMPRESSED_SIZE)*100.0/$ORIGINAL_SIZE}")
    echo "  ✓ Compression successful: $RATIO% reduction"
else
    echo "  ✗ Warning: Compressed size is not smaller than original"
    exit 1
fi

echo "Cleaning up temporary files..."
rm -f "$COMPRESSED_FILE" "$DECOMPRESSED_FILE"

echo "Test completed successfully!"

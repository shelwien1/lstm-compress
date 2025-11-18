#!/bin/bash
set -e

TEST_FILE="build.sh"
COMPRESSED_FILE="test.compressed"
DECOMPRESSED_FILE="test.decompressed"
BACKUP_FILE="test.backup"

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

echo "Comparing compressed file size with backup..."
if [ ! -f "$BACKUP_FILE" ]; then
    echo "  ✗ Backup file not found!"
    exit 1
fi

BACKUP_SIZE=$(stat -c%s "$BACKUP_FILE")
echo "  Backup size: $BACKUP_SIZE bytes"
echo "  Current compressed size: $COMPRESSED_SIZE bytes"

if [ "$COMPRESSED_SIZE" -eq "$BACKUP_SIZE" ]; then
    echo "  ✓ Compressed file size matches backup - behavior unchanged"
else
    echo "  ✗ ERROR: Compressed file size differs from backup!"
    echo "  This indicates the compression behavior has changed."
    exit 1
fi

echo "Cleaning up temporary files..."
rm -f "$DECOMPRESSED_FILE"

echo "Test completed successfully!"

#!/bin/bash
set -e

echo "Testing kanncompr_..."
echo

# Clean up previous test files
rm -f 1 2

# Test 1: Compress g.bat to file "1"
echo "Step 1: Compressing g.bat to file '1'..."
./kanncompr_ c g.bat 1
echo "✓ Compression completed"
echo

# Test 2: Decompress "1" to file "2"
echo "Step 2: Decompressing '1' to file '2'..."
./kanncompr_ d 1 2
echo "✓ Decompression completed"
echo

# Test 3: Verify that "1" is smaller than g.bat
echo "Step 3: Checking file sizes..."
SIZE_ORIGINAL=$(stat -c%s g.bat)
SIZE_COMPRESSED=$(stat -c%s 1)

echo "  Original size (g.bat): $SIZE_ORIGINAL bytes"
echo "  Compressed size (1):   $SIZE_COMPRESSED bytes"

if [ $SIZE_COMPRESSED -lt $SIZE_ORIGINAL ]; then
    RATIO=$(awk "BEGIN {printf \"%.2f\", 100.0 * $SIZE_COMPRESSED / $SIZE_ORIGINAL}")
    echo "✓ Compressed file is smaller ($RATIO% of original)"
else
    echo "✗ FAILED: Compressed file is not smaller than original!"
    exit 1
fi
echo

# Test 4: Verify that g.bat and "2" have matching hashes
echo "Step 4: Verifying decompressed file matches original..."
HASH_ORIGINAL=$(sha256sum g.bat | awk '{print $1}')
HASH_DECOMPRESSED=$(sha256sum 2 | awk '{print $1}')

echo "  Original hash (g.bat): $HASH_ORIGINAL"
echo "  Decompressed hash (2): $HASH_DECOMPRESSED"

if [ "$HASH_ORIGINAL" = "$HASH_DECOMPRESSED" ]; then
    echo "✓ Hashes match - decompression is correct!"
else
    echo "✗ FAILED: Hashes do not match!"
    echo "  This may require rebuilding with -fno-fast-math flag"
    exit 1
fi
echo

echo "================================================"
echo "All tests passed successfully!"
echo "================================================"

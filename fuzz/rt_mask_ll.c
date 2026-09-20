/* Round-trip fuzzer for lossless mask chunks (spec/format.md §12): encoding
 * arbitrary voxels must succeed, stay inside the bound, report a stored grid
 * edge of 128, decode to the source's support exactly as 0/255, and every block
 * decode must equal the corresponding region of the full decode. */
#include "../volcomp.h"

#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t src[VOLCOMP_CHUNK_VOXELS], dec[VOLCOMP_CHUNK_VOXELS], blk[4096];
  static uint8_t *enc, *grid;
  if (!enc) enc = malloc(VOLCOMP_ENCODE_BOUND);
  if (!grid) grid = malloc(VOLCOMP_CHUNK_VOXELS);
  if (size < 2) return 0;
  /* tile the payload; data[0] sets a threshold so both sparse and dense masks occur */
  size_t p = size - 1;
  for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)(data[1 + i % p] > data[0] ? 255 : 0);
  size_t n;
  if (volcomp_mask_encode_lossless(src, enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) abort();
  if (n > VOLCOMP_MASK_LL_BOUND) abort();
  uint32_t dim = 0;
  if (volcomp_mask_info(enc, n, &dim) != VOLCOMP_OK || dim != VOLCOMP_CHUNK_DIM) abort();
  if (volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK) abort();
  if (volcomp_decode(enc, n, dec, sizeof dec) != VOLCOMP_OK) abort();
  if (memcmp(grid, dec, sizeof dec)) abort();
  for (size_t i = 0; i < sizeof src; i++)
    if (dec[i] != (src[i] ? 255u : 0u)) abort();
  uint32_t b = data[1] % 512;
  uint32_t bz = b >> 6, by = (b >> 3) & 7, bx = b & 7;
  if (volcomp_decode_block(enc, n, bz, by, bx, blk, sizeof blk) != VOLCOMP_OK) abort();
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++)
      if (memcmp(blk + z * 256 + y * 16, dec + ((size_t)(bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16, 16))
        abort();
  return 0;
}

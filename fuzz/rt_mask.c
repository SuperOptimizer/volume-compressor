/* Round-trip fuzzer for mask chunks: encoding arbitrary voxels must succeed,
 * the stream must decode, the STORED 64^3 grid must be the exact 2x2x2 majority
 * pool of the source, and every block decode must equal the full decode. */
#include "../volcomp.h"

#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t src[VOLCOMP_CHUNK_VOXELS], dec[VOLCOMP_CHUNK_VOXELS], blk[4096];
  static uint8_t grid[VOLCOMP_MASK_VOXELS];
  static uint8_t *enc;
  if (!enc) enc = malloc(VOLCOMP_ENCODE_BOUND);
  if (size < 2) return 0;
  /* tile the payload; data[0] sets a threshold so both sparse and dense masks occur */
  size_t p = size - 1;
  for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)(data[1 + i % p] > data[0] ? 255 : 0);
  size_t n;
  if (volcomp_mask_encode(src, enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) abort();
  if (n > 9 + VOLCOMP_MASK_VOXELS / 8) abort();
  uint32_t dim = 0;
  if (volcomp_mask_info(enc, n, &dim) != VOLCOMP_OK || dim != VOLCOMP_MASK_DIM) abort();
  if (volcomp_mask_decode_stored(enc, n, grid, sizeof grid) != VOLCOMP_OK) abort();
  for (uint32_t z = 0; z < VOLCOMP_MASK_DIM; z++)
    for (uint32_t y = 0; y < VOLCOMP_MASK_DIM; y++)
      for (uint32_t x = 0; x < VOLCOMP_MASK_DIM; x++) {
        uint32_t c = 0;
        for (uint32_t dz = 0; dz < 2; dz++)
          for (uint32_t dy = 0; dy < 2; dy++)
            for (uint32_t dx = 0; dx < 2; dx++)
              c += src[(((size_t)(2 * z + dz) * 128u) + 2 * y + dy) * 128u + 2 * x + dx] != 0;
        uint8_t want = (uint8_t)(c * 2 >= 8 ? 255 : 0);
        if (grid[((size_t)z * VOLCOMP_MASK_DIM + y) * VOLCOMP_MASK_DIM + x] != want) abort();
      }
  if (volcomp_decode(enc, n, dec, sizeof dec) != VOLCOMP_OK) abort();
  uint32_t b = data[1] % 512;
  uint32_t bz = b >> 6, by = (b >> 3) & 7, bx = b & 7;
  if (volcomp_decode_block(enc, n, bz, by, bx, blk, sizeof blk) != VOLCOMP_OK) abort();
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++)
      if (memcmp(blk + z * 256 + y * 16, dec + ((size_t)(bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16, 16))
        abort();
  return 0;
}

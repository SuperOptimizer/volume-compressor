/* Round-trip fuzzer: encode(arbitrary voxels, arbitrary q) must succeed, decode
 * must succeed, and every block decode must equal the full decode. (The
 * format guarantees no error bound; hostile content can ring arbitrarily.) */
#include "../volcomp.h"

#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t src[VOLCOMP_CHUNK_VOXELS], dec[VOLCOMP_CHUNK_VOXELS], blk[4096];
  static uint8_t *enc;
  if (!enc) enc = malloc(VOLCOMP_ENCODE_BOUND);
  if (size < 2) return 0;
  float q = 1.0f + (float)data[0] * (254.0f / 255.0f);
  /* tile the fuzz payload with a slowly varying offset so both flat and noisy blocks occur */
  size_t p = size - 1;
  for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)(data[1 + i % p] + (i >> 12));
  size_t n;
  if (volcomp_encode(src, q, enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) abort();
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

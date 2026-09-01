/* Hostile-input fuzzer: arbitrary bytes must never crash the decoder. */
#include "../volcomp.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t dst[VOLCOMP_CHUNK_VOXELS], blk[VOLCOMP_BLOCK_VOXELS];
  (void)volcomp_decode(data, size, dst, sizeof dst);
  uint32_t b = size ? data[0] : 0;
  (void)volcomp_decode_block(data, size, b & 7, (b >> 3) & 7, (b >> 6) & 7, blk, sizeof blk);
  return 0;
}

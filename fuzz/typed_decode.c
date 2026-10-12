/* Hostile-input fuzzer for typed chunks (spec/format.md §14): arbitrary bytes must
 * never crash volcomp_decode_typed / volcomp_decode_block_typed, whatever dtype the
 * stream claims, and whenever the whole chunk decodes, every block decode must
 * equal the corresponding region of it. */
#include "../volcomp.h"

#include <stdlib.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t dst[4 * VOLCOMP_CHUNK_VOXELS], blk[4 * VOLCOMP_BLOCK_VOXELS];
  volcomp_dtype dt = VOLCOMP_DTYPE_U16;
  const bool known = volcomp_chunk_dtype(data, size, &dt) == VOLCOMP_OK;
  const size_t es = known ? volcomp_dtype_size(dt) : 2;
  const bool full = volcomp_decode_typed(data, size, dst, sizeof dst, dt) == VOLCOMP_OK;
  uint32_t b = size ? data[size - 1] : 0;
  for (int k = 0; k < 2; k++, b = (b * 167u + 13u) & 511u) {
    const uint32_t bz = b >> 6 & 7u, by = b >> 3 & 7u, bx = b & 7u;
    if (volcomp_decode_block_typed(data, size, bz, by, bx, blk, sizeof blk, dt) != VOLCOMP_OK || !full) continue;
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        if (memcmp(blk + (z * 16 + y) * 16 * es, dst + (((size_t)(bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16) * es,
                   16 * es))
          abort();
  }
  /* the u8 calls refuse a typed chunk without reading its payload */
  if (known && dt != VOLCOMP_DTYPE_U8 && volcomp_decode(data, size, dst, sizeof dst) != VOLCOMP_ERR_ARG) abort();
  return 0;
}

/* Exported (non-static) entry points over volcomp.h for ctypes / FFI users.
 * Build: clang -O3 -mavx2 -mfma -shared -fPIC -o libvolcomp.so volcomp_shim.c -lm
 * (or the CMake target `volcomp_shim`). The ABI is the header's, minus `static`. */
#include "../volcomp.h"

#define VOLCOMP_EXPORT __attribute__((visibility("default")))

VOLCOMP_EXPORT const char *volcomp_shim_version(void) { return VOLCOMP_VERSION_STRING; }
VOLCOMP_EXPORT size_t volcomp_shim_encode_bound(void) { return VOLCOMP_ENCODE_BOUND; }
VOLCOMP_EXPORT const char *volcomp_shim_status_string(int s) { return volcomp_status_string((volcomp_status)s); }
VOLCOMP_EXPORT int volcomp_shim_encode(const uint8_t *src, float q, void *dst, size_t cap, size_t *out_n) {
  return (int)volcomp_encode(src, q, dst, cap, out_n);
}
VOLCOMP_EXPORT int volcomp_shim_decode(const void *enc, size_t n, uint8_t *dst, size_t cap) {
  return (int)volcomp_decode(enc, n, dst, cap);
}
VOLCOMP_EXPORT int volcomp_shim_decode_block(const void *enc, size_t n, unsigned bz, unsigned by, unsigned bx,
                                             uint8_t *dst, size_t cap) {
  return (int)volcomp_decode_block(enc, n, bz, by, bx, dst, cap);
}
VOLCOMP_EXPORT void volcomp_shim_deblock(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q) {
  volcomp_deblock(vol, nz, ny, nx, q);
}

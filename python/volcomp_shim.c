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
VOLCOMP_EXPORT int volcomp_shim_decode_smooth(const void *enc, size_t n, uint8_t *dst, size_t cap, float sigma,
                                               unsigned flags) {
  return (int)volcomp_decode_smooth(enc, n, dst, cap, sigma, flags);
}
VOLCOMP_EXPORT void volcomp_shim_deblock_ex(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q, unsigned flags) {
  volcomp_deblock_ex(vol, nz, ny, nx, q, flags);
}
VOLCOMP_EXPORT int volcomp_shim_decode_block(const void *enc, size_t n, unsigned bz, unsigned by, unsigned bx,
                                             uint8_t *dst, size_t cap) {
  return (int)volcomp_decode_block(enc, n, bz, by, bx, dst, cap);
}
VOLCOMP_EXPORT int volcomp_shim_mask_encode(const uint8_t *src, void *dst, size_t cap, size_t *out_n) {
  return (int)volcomp_mask_encode(src, dst, cap, out_n);
}
VOLCOMP_EXPORT int volcomp_shim_mask_info(const void *enc, size_t n, uint32_t *dim) {
  return (int)volcomp_mask_info(enc, n, dim);
}
VOLCOMP_EXPORT int volcomp_shim_mask_decode_stored(const void *enc, size_t n, uint8_t *dst, size_t cap) {
  return (int)volcomp_mask_decode_stored(enc, n, dst, cap);
}
VOLCOMP_EXPORT size_t volcomp_shim_mask_voxels(void) { return VOLCOMP_MASK_VOXELS; }
VOLCOMP_EXPORT int volcomp_shim_mask_encode_lossless(const uint8_t *src, void *dst, size_t cap, size_t *out_n) {
  return (int)volcomp_mask_encode_lossless(src, dst, cap, out_n);
}
VOLCOMP_EXPORT size_t volcomp_shim_mask_ll_bound(void) { return VOLCOMP_MASK_LL_BOUND; }
VOLCOMP_EXPORT size_t volcomp_shim_surface_bound(void) { return VOLCOMP_SURFACE_ENCODE_BOUND; }
VOLCOMP_EXPORT int volcomp_shim_surface_encode(const uint8_t *src, float q, unsigned thr, void *dst, size_t cap,
                                               size_t *out_n) {
  return (int)volcomp_surface_encode(src, q, thr, dst, cap, out_n);
}
VOLCOMP_EXPORT int volcomp_shim_surface_info(const void *enc, size_t n, uint32_t *thr, uint32_t *margin) {
  return (int)volcomp_surface_info(enc, n, thr, margin);
}
VOLCOMP_EXPORT int volcomp_shim_is_lossless(const void *enc, size_t n, int *out) {
  bool b = false;
  int st = (int)volcomp_is_lossless(enc, n, &b);
  *out = b;
  return st;
}
VOLCOMP_EXPORT int volcomp_shim_stream_q(const void *enc, size_t n, float *q) {
  return (int)volcomp_stream_q(enc, n, q);
}
VOLCOMP_EXPORT void volcomp_shim_deblock(uint8_t *vol, size_t nz, size_t ny, size_t nx, float q) {
  volcomp_deblock(vol, nz, ny, nx, q);
}

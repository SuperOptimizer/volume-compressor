/* WebAssembly entry points over volcomp.h for the browser viewer (web/).
 * Built by web/build.sh with Emscripten as a standalone module (no JS glue):
 * the worker instantiates it, copies an encoded chunk into vc_in(), calls
 * vc_decode / vc_decode_smooth, and reads the 128^3 result from vc_out().
 * One module instance per worker; the buffers are owned by the module. */
#include "../../volcomp.h"

#define EXPORT __attribute__((used, visibility("default")))

static uint8_t *g_in = NULL;
static size_t g_in_cap = 0;
static uint8_t *g_out = NULL;
static uint8_t *g_block = NULL;

/* The input buffer, grown to at least n bytes (0 on allocation failure). The
 * pointer can move when it grows, so call this before every copy. */
EXPORT uint8_t *vc_in(size_t n) {
  if (n > g_in_cap) {
    free(g_in);
    g_in_cap = n + (n >> 2) + 4096;
    g_in = (uint8_t *)malloc(g_in_cap);
    if (!g_in) g_in_cap = 0;
  }
  return g_in;
}
/* The 128^3 z-major output buffer (VOLCOMP_CHUNK_VOXELS bytes). */
EXPORT uint8_t *vc_out(void) {
  if (!g_out) g_out = (uint8_t *)malloc(VOLCOMP_CHUNK_VOXELS);
  return g_out;
}
/* The 16^3 output buffer of vc_decode_block. */
EXPORT uint8_t *vc_block_out(void) {
  if (!g_block) g_block = (uint8_t *)malloc(VOLCOMP_BLOCK_VOXELS);
  return g_block;
}
EXPORT int vc_decode(size_t n) { return (int)volcomp_decode(g_in, n, vc_out(), VOLCOMP_CHUNK_VOXELS); }
/* flags: VOLCOMP_DEBLOCK_ZERO_GUARD (1) | VOLCOMP_SMOOTH_GATED (2) */
EXPORT int vc_decode_smooth(size_t n, float strength, unsigned flags) {
  return (int)volcomp_decode_smooth(g_in, n, vc_out(), VOLCOMP_CHUNK_VOXELS, strength, flags);
}
EXPORT int vc_decode_block(size_t n, unsigned bz, unsigned by, unsigned bx) {
  return (int)volcomp_decode_block(g_in, n, bz, by, bx, vc_block_out(), VOLCOMP_BLOCK_VOXELS);
}
/* ---- chunk info (all read the encoded chunk in vc_in) ---- */
/* header byte 5 (0 lossy, 1..3 lossless, 4/5 mask, 6 surface), or -1 if not a volcomp stream */
EXPORT int vc_chunk_mode(size_t n) {
  if (n < 8 || g_in[0] != 'V' || g_in[1] != 'O' || g_in[2] != 'L' || g_in[3] != 'C') return -1;
  return g_in[5];
}
/* the stream's q (0 for lossless/mask), or -1 on error */
EXPORT float vc_stream_q(size_t n) {
  float q = 0;
  return volcomp_stream_q(g_in, n, &q) == VOLCOMP_OK ? q : -1.0f;
}
EXPORT int vc_is_lossless(size_t n) {
  bool b = false;
  return volcomp_is_lossless(g_in, n, &b) == VOLCOMP_OK ? (int)b : -1;
}
/* surface chunk threshold (margin in *margin_out if non-null), or -1 if not a surface chunk */
EXPORT int vc_surface_thr(size_t n) {
  uint32_t thr = 0, margin = 0;
  return volcomp_surface_info(g_in, n, &thr, &margin) == VOLCOMP_OK ? (int)thr : -1;
}
/* stored-grid edge of a mask chunk (64 / 128), or -1 if not a mask chunk */
EXPORT int vc_mask_dim(size_t n) {
  uint32_t d = 0;
  return volcomp_mask_info(g_in, n, &d) == VOLCOMP_OK ? (int)d : -1;
}
EXPORT const char *vc_version(void) { return VOLCOMP_VERSION_STRING; }
EXPORT const char *vc_kernels(void) { return volcomp_kernels(); }
EXPORT const char *vc_status_string(int s) { return volcomp_status_string((volcomp_status)s); }

/* shard-pack: write a shard from a temp directory of chunks, parse its index,
 * decode every present chunk and compare with the single-file encode. */
#include "../volcomp.h"
#include "check.h"
#include "shard_pack.h"

#include <sys/stat.h>
#include <unistd.h>

static void write_chunk(const char *dir, uint32_t z, uint32_t y, uint32_t x, const uint8_t *v) {
  char p[4096];
  snprintf(p, sizeof p, "%s/%u_%u_%u.u8", dir, z, y, x);
  FILE *f = fopen(p, "wb");
  CHECK(f && fwrite(v, 1, VOLCOMP_CHUNK_VOXELS, f) == VOLCOMP_CHUNK_VOXELS);
  if (f) fclose(f);
}

int main(void) {
  char dir[] = "/tmp/volcomp_shard_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  static uint8_t a[VOLCOMP_CHUNK_VOXELS], b[VOLCOMP_CHUNK_VOXELS], z[VOLCOMP_CHUNK_VOXELS];
  vt_synth_chunk(a, 11);
  vt_synth_chunk(b, 22);
  memset(z, 0, sizeof z);
  write_chunk(dir, 0, 0, 0, a);
  write_chunk(dir, 3, 5, 7, b);
  write_chunk(dir, 1, 1, 1, z); /* all-zero -> missing */
  char out[4200];
  snprintf(out, sizeof out, "%s/out.shard", dir);
  unsigned present = 0;
  uint64_t bytes = 0;
  CHECK_EQ(shard_pack(dir, out, 6.0f, &present, &bytes), 0);
  CHECK_EQ(present, 2);
  size_t n;
  uint8_t *img = shard_read_file(out, &n);
  CHECK(img != NULL);
  CHECK_EQ(n, bytes + SHARD_INDEX_BYTES);
  uint64_t off[512], nb[512];
  CHECK(shard_index_parse(img, n, off, nb));
  unsigned np = 0;
  for (uint32_t i = 0; i < 512; i++)
    if (off[i] != ~0ull) np++;
  CHECK_EQ(np, 2);
  CHECK(off[1 * 64 + 1 * 8 + 1] == ~0ull && nb[1 * 64 + 1 * 8 + 1] == ~0ull);
  /* chunk 0 and chunk (3,5,7) decode identically to a direct encode */
  static uint8_t enc[VOLCOMP_ENCODE_BOUND], d1[VOLCOMP_CHUNK_VOXELS], d2[VOLCOMP_CHUNK_VOXELS];
  const uint8_t *srcs[2] = {a, b};
  uint32_t ids[2] = {0, 3 * 64 + 5 * 8 + 7};
  for (int k = 0; k < 2; k++) {
    size_t en;
    CHECK_EQ(volcomp_encode(srcs[k], 6.0f, enc, sizeof enc, &en), VOLCOMP_OK);
    CHECK_EQ(nb[ids[k]], en);
    CHECK(off[ids[k]] + nb[ids[k]] <= n - SHARD_INDEX_BYTES);
    CHECK(memcmp(img + off[ids[k]], enc, en) == 0);
    CHECK_EQ(volcomp_decode(img + off[ids[k]], (size_t)nb[ids[k]], d1, sizeof d1), VOLCOMP_OK);
    CHECK_EQ(volcomp_decode(enc, en, d2, sizeof d2), VOLCOMP_OK);
    CHECK(memcmp(d1, d2, sizeof d1) == 0);
  }
  /* corrupt the index CRC */
  img[n - 1] ^= 1;
  CHECK(!shard_index_parse(img, n, off, nb));
  free(img);
  /* cleanup */
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  CHECK(system(cmd) == 0);
  TEST_END();
}

#include "../src/vc/vc.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
__AFL_FUZZ_INIT();
int main(void){
#ifdef __AFL_HAVE_MANUAL_CONTROL
  __AFL_INIT();
#endif
  unsigned char *buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(10000)) {
    int len = __AFL_FUZZ_TESTCASE_LEN;
    vc_archive *a = vc_open(buf, (size_t)len);
    if (!a) continue;
    vc_dims d;
    for (int lod=0; lod<VC_NLOD; ++lod) {
      if (vc_lod_dims(a, lod, &d) != VC_OK) continue;
      size_t n=(size_t)d.nx*d.ny*d.nz; if(!n||n>(64u<<20)) continue;
      unsigned char *o=malloc(n); vc_dims od;
      if(o){ vc_decode_lod(a,lod,o,&od); free(o); }
      unsigned char atom[VC_ATOM3]; vc_decode_atom(a,lod,0,0,0,atom);
    }
    vc_close(a);
  }
  return 0;
}

// Parity test: vc_open_streaming (over a memcpy-from-buffer cb) must decode
// bit-identically to vc_open on the same archive bytes. Also exercises a
// GET-counting cb to confirm the streaming path actually calls read.
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
typedef unsigned char u8;
#define CK(c,...) do{if(!(c)){printf("FAIL: " __VA_ARGS__);printf("\n");return 1;}}while(0)

// streaming byte source = the whole archive in RAM (stand-in for S3+cache)
typedef struct { const u8* buf; size_t len; long reads; long bytes; } src;
static vc_status memsrc(void* ud, uint64_t off, uint32_t len, u8* dst){
    src* s=(src*)ud; if(off+len > s->len) return VC_ERR_FORMAT;
    memcpy(dst, s->buf+off, len); s->reads++; s->bytes+=len; return VC_OK;
}

static void make_atom(u8* v, unsigned az,unsigned ay,unsigned ax){
    for(unsigned z=0;z<VC_ATOM;z++)for(unsigned y=0;y<VC_ATOM;y++)for(unsigned x=0;x<VC_ATOM;x++)
        v[(z*VC_ATOM+y)*VC_ATOM+x]=(u8)(40 + ((az*7+ay*13+ax*17+x+y+z)%150));
}
int main(void){
    const char* path="/tmp/vc_stream_test.vca";
    vc_dims d0={256,192,160};
    vc_writer* w=vc_create(path,d0,30.0f); CK(w,"create");
    for(int l=0;l<VC_NLOD;l++) vc_set_base_q(w,l,1.0f);
    unsigned acx=(d0.nx+31)/32, acy=(d0.ny+31)/32, acz=(d0.nz+31)/32;
    u8 vox[VC_ATOM3];
    for(unsigned az=0;az<acz;az++)for(unsigned ay=0;ay<acy;ay++)for(unsigned ax=0;ax<acx;ax++){
        if(az==0&&ay==0&&ax==0) continue;                 // leave ABSENT
        if(az==acz-1&&ay==0&&ax==0){vc_mark_zero_atom(w,0,az,ay,ax);continue;} // ZERO
        make_atom(vox,az,ay,ax); vc_append_atom(w,0,az,ay,ax,vox);
    }
    vc_writer_close(w);

    int fd=open(path,O_RDONLY); struct stat st; fstat(fd,&st);
    void* m=mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    const u8* buf=(const u8*)m; size_t len=st.st_size;

    vc_archive* flat=vc_open(buf,len); CK(flat,"vc_open");
    src s={buf,len,0,0};
    vc_archive* strm=vc_open_streaming(memsrc,&s,len); CK(strm,"vc_open_streaming");

    // every atom: coverage + decode must match bit-for-bit
    u8 a[VC_ATOM3], b[VC_ATOM3]; long checked=0, present=0;
    for(unsigned az=0;az<acz;az++)for(unsigned ay=0;ay<acy;ay++)for(unsigned ax=0;ax<acx;ax++){
        vc_cover cf=vc_atom_coverage(flat,0,az,ay,ax), cs=vc_atom_coverage(strm,0,az,ay,ax);
        CK(cf==cs,"coverage mismatch at %u,%u,%u: %d vs %d",az,ay,ax,cf,cs);
        vc_status rf=vc_decode_atom(flat,0,(int)ax,(int)ay,(int)az,a);
        vc_status rs=vc_decode_atom(strm,0,(int)ax,(int)ay,(int)az,b);
        CK(rf==rs,"status mismatch at %u,%u,%u",az,ay,ax);
        CK(memcmp(a,b,VC_ATOM3)==0,"DECODE MISMATCH at %u,%u,%u",az,ay,ax);
        if(cf==VC_PRESENT)present++; checked++;
    }
    // region decode parity
    vc_box box={0,0,0,64,64,64};
    u8 rgf[64*64*64], rgs[64*64*64];
    CK(vc_decode_region(flat,0,box,rgf)==VC_OK,"region flat");
    CK(vc_decode_region(strm,0,box,rgs)==VC_OK,"region strm");
    CK(memcmp(rgf,rgs,sizeof rgf)==0,"REGION MISMATCH");

    printf("streaming parity OK: %ld atoms (%ld present) bit-identical; region OK; cb reads=%ld bytes=%ld\n",
           checked,present,s.reads,s.bytes);
    CK(s.reads>0,"streaming cb was never called?!");
    vc_close(flat); vc_close(strm); munmap(m,len); unlink(path);
    return 0;
}

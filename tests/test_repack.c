// test_repack — build a v1 archive, run vc_repack -> v2, and assert the v2 decodes
// BIT-IDENTICALLY to v1 for every atom + a region, plus v2 streaming parity. The
// repack changes only the container layout, so decode must be unchanged.
// argv[1] = path to the vc_repack binary (passed by ctest via $<TARGET_FILE:vc_repack>).
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

static u8* map_file(const char* p, size_t* len){
    int fd=open(p,O_RDONLY); if(fd<0)return NULL; struct stat st; fstat(fd,&st);
    void* m=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(m==MAP_FAILED)return NULL; *len=st.st_size; return (u8*)m;
}
static void make_atom(u8* v,unsigned az,unsigned ay,unsigned ax){
    for(unsigned z=0;z<VC_ATOM;z++)for(unsigned y=0;y<VC_ATOM;y++)for(unsigned x=0;x<VC_ATOM;x++)
        v[(z*VC_ATOM+y)*VC_ATOM+x]=(u8)(50+((az*7+ay*13+ax*17+x+y+z)%170));
}
// streaming byte source over an in-RAM copy
typedef struct{const u8*buf;size_t len;}src;
static vc_status memsrc(void*ud,uint64_t off,uint32_t len,u8*dst){
    src*s=(src*)ud; if(off+len>s->len)return VC_ERR_FORMAT; memcpy(dst,s->buf+off,len); return VC_OK;
}

int main(int argc,char**argv){
    const char* repack = argc>1?argv[1]:"./vc_repack";
    const char* v1path="/tmp/vc_repack_v1.vca";
    const char* v2path="/tmp/vc_repack_v2.vca";

    // ---- build a multi-LOD v1 with present/zero/absent atoms ----
    vc_dims d0={256,192,160};
    vc_writer* w=vc_create(v1path,d0,30.0f); CK(w,"create");
    for(int l=0;l<VC_NLOD;l++) vc_set_base_q(w,l,1.0f);
    // populate LOD0 + LOD1 + LOD2 so multiple LODs have regions (coarse-first matters)
    for(int lod=0;lod<3;lod++){
        vc_dims d=d0; for(int i=0;i<lod;i++){d.nx=(d.nx+1)/2;d.ny=(d.ny+1)/2;d.nz=(d.nz+1)/2;}
        unsigned acx=(d.nx+31)/32,acy=(d.ny+31)/32,acz=(d.nz+31)/32;
        u8 vox[VC_ATOM3];
        for(unsigned az=0;az<acz;az++)for(unsigned ay=0;ay<acy;ay++)for(unsigned ax=0;ax<acx;ax++){
            if(lod==0&&az==0&&ay==0&&ax==0) continue;                  // leave ABSENT
            if(lod==0&&az==acz-1&&ay==0&&ax==0){vc_mark_zero_atom(w,0,az,ay,ax);continue;} // ZERO
            make_atom(vox,az,ay,ax); CK(vc_append_atom(w,lod,az,ay,ax,vox)==VC_OK,"append l%d",lod);
        }
    }
    vc_writer_close(w);

    // ---- run vc_repack v1 -> v2 ----
    char cmd[1024]; snprintf(cmd,sizeof cmd,"\"%s\" \"%s\" \"%s\" >/dev/null 2>&1",repack,v1path,v2path);
    int rc=system(cmd); CK(rc==0,"vc_repack exit %d",rc);

    // ---- open both, assert v2 is version 2 + every atom decodes identically ----
    size_t l1,l2; u8* b1=map_file(v1path,&l1); u8* b2=map_file(v2path,&l2);
    CK(b1&&b2,"map archives");
    CK(b2[4]==2,"v2 header version (got %d)",b2[4]);   // FH_VER offset 4
    vc_archive* a1=vc_open(b1,l1); CK(a1,"open v1");
    vc_archive* a2=vc_open(b2,l2); CK(a2,"open v2");

    long checked=0,present=0;
    u8 o1[VC_ATOM3],o2[VC_ATOM3];
    for(int lod=0;lod<VC_NLOD;lod++){
        vc_dims d; if(vc_lod_dims(a2,lod,&d)!=VC_OK)break;
        unsigned acx=(d.nx+31)/32,acy=(d.ny+31)/32,acz=(d.nz+31)/32;
        for(unsigned az=0;az<acz;az++)for(unsigned ay=0;ay<acy;ay++)for(unsigned ax=0;ax<acx;ax++){
            vc_cover c1=vc_atom_coverage(a1,lod,az,ay,ax), c2=vc_atom_coverage(a2,lod,az,ay,ax);
            CK(c1==c2,"coverage mismatch l%d (%u,%u,%u): %d vs %d",lod,az,ay,ax,c1,c2);
            vc_status r1=vc_decode_atom(a1,lod,(int)ax,(int)ay,(int)az,o1);
            vc_status r2=vc_decode_atom(a2,lod,(int)ax,(int)ay,(int)az,o2);
            CK(r1==r2,"status mismatch l%d",lod);
            CK(memcmp(o1,o2,VC_ATOM3)==0,"DECODE MISMATCH l%d (%u,%u,%u)",lod,az,ay,ax);
            if(c1==VC_PRESENT)present++; checked++;
        }
    }
    // region decode parity
    vc_box box={0,0,0,64,64,64}; u8 rg1[64*64*64],rg2[64*64*64];
    CK(vc_decode_region(a1,0,box,rg1)==VC_OK,"region v1");
    CK(vc_decode_region(a2,0,box,rg2)==VC_OK,"region v2");
    CK(memcmp(rg1,rg2,sizeof rg1)==0,"REGION MISMATCH");

    // v2 streaming parity (open v2 via callback, compare to flat v2)
    src s={b2,l2}; vc_archive* a2s=vc_open_streaming(memsrc,&s,l2); CK(a2s,"open v2 streaming");
    CK(b2[4]==2,"streamed is v2");
    for(int lod=0;lod<3;lod++){
        vc_dims d; vc_lod_dims(a2,lod,&d);
        unsigned acx=(d.nx+31)/32,acy=(d.ny+31)/32,acz=(d.nz+31)/32;
        for(unsigned az=0;az<acz;az+=2)for(unsigned ay=0;ay<acy;ay+=2)for(unsigned ax=0;ax<acx;ax+=2){
            vc_decode_atom(a2,lod,(int)ax,(int)ay,(int)az,o1);
            vc_decode_atom(a2s,lod,(int)ax,(int)ay,(int)az,o2);
            CK(memcmp(o1,o2,VC_ATOM3)==0,"v2 streaming mismatch l%d (%u,%u,%u)",lod,az,ay,ax);
        }
    }

    printf("repack OK: v1->v2 %ld atoms (%ld present) bit-identical; region OK; v2 streaming parity OK\n",checked,present);
    vc_close(a1); vc_close(a2); vc_close(a2s);
    munmap(b1,l1); munmap(b2,l2); unlink(v1path); unlink(v2path);
    return 0;
}

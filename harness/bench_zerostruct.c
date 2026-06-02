// HIERARCHICAL ZERO-STRUCTURE for the entropy stage (R3 experiment, PLAN §2
// Entropy row). THROWAWAY self-contained bench (does NOT touch codec.c): it
// #includes the WON transform (integer DCT-16^3) + the per-coefficient 16^3
// quant matrix, and links the WON entropy coder (RLGR). It builds two explicit
// zero-structure entropy schemes ON TOP OF / COMPARED AGAINST the won RLGR
// run-mode baseline, at EQUAL DISTORTION (identical quantized levels — only the
// entropy stage differs), across q in {16,32,64,128} on real PHerc Paris 4.
//
// The structure (HEVC/AV1 multilevel-significance-map + last-position, applied
// WITHIN one 16^3 atom so touched=1 still holds):
//   (1) LAST-SIGNIFICANT-POSITION (EOB): along the 3D frequency scan, code the
//       index of the last nonzero level; everything after it is implicitly zero
//       and costs ~nothing.
//   (2) SUB-CUBE CODED-BLOCK-FLAGS (CBF): partition the 4096 coefficients into
//       sub-cubes (4^3=64 coeffs/cube -> 64 cubes, OR 8^3) and code ONE
//       "any-nonzero?" flag per sub-cube; skip all-zero sub-cubes entirely.
//
// MODES (all code the SAME quantized levels => identical reconstruction):
//   A) RLGR        : the won baseline. RLGR over the 4096 scan-ordered levels.
//   B) EOB         : last-sig-position (Golomb) + RLGR over the [0..last] prefix.
//   C) CBF4+EOB    : EOB, then within [0..last] a 4^3 sub-cube CBF map; RLGR only
//                    the coefficients inside significant sub-cubes (in scan order).
//   D) CBF8+EOB    : same with 8^3 sub-cubes (8 sub-cubes of 512 coeffs).
//
// We report, per q: total payload bytes + ratio (incl. 2-byte DC/atom), the
// INCREMENTAL ratio gain vs RLGR, AND the DECODE SYMBOL COUNT (number of RLGR
// levels the decoder must run through) for each mode — the decode-speed proxy,
// since RLGR decode cost is ~linear in coded symbols. Round-trip is verified
// (lossless wrt the quantized levels) for every mode and atom; touched=1 holds
// because every structure decision is in-atom.
//
// Pure C23, libc/libm. Pipeline atom = 16^3. DC = per-atom mean (i16 side, +2 B).
#define _POSIX_C_SOURCE 200809L
#include "../include/vc/types.h"
#include "../src/metrics/metrics.h"
#include "../src/core/bitio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- won transform ---------------------------------------------------------
#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  H_dct16_fwd
#define vc_dct_int16_inv  H_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

// --- won entropy coder (RLGR) ----------------------------------------------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A    16u
#define AVOX 4096u

// --- current-stack quant params (the WON config given to this experiment) --
// HF-protecting per-coef matrix slope ~0.5, dead-zone width dz=0.80, dequant
// reconstruction offset = 0.40. Implemented inline (parameterized) so the
// distortion profile matches the live codec exactly.
#define HF_SLOPE   0.50f
#define DZ_WIDTH   0.80f     // dead-zone half-width as a fraction of step
#define RECON_OFF  0.40f     // dequant reconstruction offset (toward the dead zone edge)

static inline f32 hf_weight(u32 s){ f32 t=(f32)s/45.0f; return 1.0f - HF_SLOPE*t; }
static void build_step(f32 *restrict step, f32 base){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        f32 s = base * hf_weight(z+y+x);
        if(s<0.5f) s=0.5f;
        step[(size_t)z*256u + (size_t)y*16u + x] = s;
    }
}
static void quant_block(i16 *restrict qb, const i16 *restrict coef, const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        f32 c=(f32)coef[i]; f32 a=c<0.f?-c:c; f32 st=step[i]; f32 inv=1.0f/st;
        i32 m=(a >= DZ_WIDTH*st);
        i32 lvl=m*((i32)(a*inv - DZ_WIDTH) + 1);
        qb[i]=(i16)(c<0.f?-lvl:lvl);
    }
}
static void dequant_block(i16 *restrict coef, const i16 *restrict qb, const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        i32 l=(i32)qb[i]; i32 al=l<0?-l:l;
        f32 r = al>0 ? ((f32)al - 1.0f + RECON_OFF)*step[i] + DZ_WIDTH*step[i] : 0.f;
        // reconstruct toward bin center: (al-1) full steps past the dead zone edge,
        // plus RECON_OFF*step into the bin. Equivalent to standard dead-zone dequant.
        f32 v = (l<0? -r : r);
        i32 iv=(i32)lrintf(v);
        if(iv>32767) iv=32767; else if(iv<-32768) iv=-32768;
        coef[i]=(i16)iv;
    }
}

// --- 3D frequency scan: positions sorted by band sum (u+v+w), then z,y,x ----
static u16 g_scan[AVOX];     // scan[k] = linear coef index (z*256+y*16+x) at scan pos k
static u16 g_invscan[AVOX];  // inverse: linear index -> scan pos
static void build_scan(void){
    // stable counting sort by key = u+v+w (0..45)
    u32 cnt[46]={0};
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x) cnt[z+y+x]++;
    u32 off[46]; u32 acc=0; for(int s=0;s<46;++s){ off[s]=acc; acc+=cnt[s]; }
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        u32 s=z+y+x; u32 lin=z*256u+y*16u+x;
        u32 pos=off[s]++; g_scan[pos]=(u16)lin; g_invscan[lin]=(u16)pos;
    }
}

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}
static inline void gather_atom(const u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8 *src=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(blk+((size_t)z*A+y)*A,src,A);
    }
}

// Golomb code of a small nonneg integer (last-position, 0..4095) with k bits
// remainder, k chosen as ~log2(expected). We use a fixed k=4 Golomb (cheap,
// near-optimal for the typically-small last positions). Counted in the bitstream.
static inline void put_golomb(vc_bitwriter *w,u32 v,u32 k){ vc_bw_put_unary(w,v>>k); if(k) vc_bw_put(w,v&((1u<<k)-1u),k); }
static inline u32 get_golomb(vc_bitreader *r,u32 k){ u32 hi=vc_br_get_unary(r); u32 lo=k?vc_br_get(r,k):0; return (hi<<k)|lo; }
// Exact bytes consumed by the reader so far (bytes pulled into acc minus the
// whole bytes still buffered in acc). The encoder byte-aligns the header with
// vc_bw_finish, so the body starts at ceil(consumed_bits/8). We round the
// reader's bit position up to the next byte.
static inline size_t br_byte_aligned(const vc_bitreader *r){
    // bits actually consumed from the stream = (bytes pulled into acc)*8 - (bits still buffered)
    size_t consumed_bits = r->byte*8 - r->nbits;
    return (consumed_bits + 7) / 8;        // encoder byte-aligned (vc_bw_finish)
}

// =====================================================================
// Per-atom: produce scan-ordered levels qs[4096] from the quantized cube qb.
static inline void to_scan(i16 *restrict qs,const i16 *restrict qb){
    for(u32 k=0;k<AVOX;++k) qs[k]=qb[g_scan[k]];
}

// --- MODE A: plain RLGR over the 4096 scan-ordered levels. Returns bytes; sets
//     *symbols = 4096 (decoder runs through all). -----------------------------
static size_t enc_rlgr(const i16 *qs,u8 *tmp,u32 *symbols){
    size_t b=vc_rlgr_encode(tmp,AVOX*3,qs,AVOX);
    *symbols=AVOX; return b;
}
static int dec_rlgr(i16 *out,const u8 *tmp,size_t len){
    vc_rlgr_decode(out,AVOX,tmp,len); return 1;
}

// --- MODE B: last-significant-position (EOB) + RLGR over [0..last]. ----------
// Layout: [golomb last-pos][RLGR(qs[0..last])]. last = -1 (all zero) -> code 0
// and store nothing else; we use last+1 = count so 0 means empty.
static size_t enc_eob(const i16 *qs,u8 *out,u32 *symbols){
    int last=-1; for(int k=AVOX-1;k>=0;--k){ if(qs[k]){ last=k; break; } }
    u32 cnt=(u32)(last+1);
    vc_bitwriter w; vc_bw_init(&w,out,AVOX*3);
    put_golomb(&w,cnt,7);               // last position+1, Golomb k=7
    size_t hdr=vc_bw_finish(&w);
    size_t body=0;
    if(cnt) body=vc_rlgr_encode(out+hdr,AVOX*3-hdr,qs,cnt);
    *symbols=cnt;                        // decoder only runs through cnt levels
    return hdr+body;
}
static int dec_eob(i16 *out,const u8 *in,size_t len){
    memset(out,0,AVOX*sizeof(i16));
    vc_bitreader r; vc_br_init(&r,in,len);
    u32 cnt=get_golomb(&r,7);
    size_t hdr=br_byte_aligned(&r);      // body starts at the encoder's byte boundary
    if(cnt) vc_rlgr_decode(out,cnt,in+hdr,len-hdr);
    return 1;
}

// --- MODES C/D: EOB + sub-cube CBF + RLGR over significant-sub-cube coeffs. --
// Sub-cubes are in the 3D COEFFICIENT space (cz,cy,cx in the z*256+y*16+x cube).
// CUBE = 4 (=> 4^3=64 sub-cubes) or 8 (=> 2^3=8 sub-cubes). For each sub-cube we
// code one any-nonzero flag (only for sub-cubes that intersect the [0..last]
// scan prefix — sub-cubes entirely beyond EOB are implicitly zero). The surviving
// coefficients (those inside significant sub-cubes, within [0..last]) are gathered
// in SCAN order and RLGR-coded. The decoder reconstructs the cube from the CBF
// map + EOB + the RLGR'd survivors.
//
// To make decode unambiguous we send, per significant sub-cube, the count of its
// coefficients that lie within [0..last] — actually that's derivable: a coef at
// scan pos p<last belongs to sub-cube id(p). The decoder walks scan positions
// 0..last, and for each, if its sub-cube flag is set, it pulls the next RLGR
// level; else it's zero. So no per-cube counts needed. CBF map = one flag per
// sub-cube that has >=1 scan position in [0..last].
static size_t enc_cbf(const i16 *qs,u8 *out,u32 cube,u32 *symbols){
    u32 perdim=A/cube;                   // sub-cubes per axis
    u32 ncubes=perdim*perdim*perdim;
    int last=-1; for(int k=AVOX-1;k>=0;--k){ if(qs[k]){ last=k; break; } }
    u32 cnt=(u32)(last+1);
    // determine which sub-cubes have any scan-position in [0..last], and which of
    // those are significant (any nonzero). cube id for scan pos p:
    u8 inrange[512]={0};                 // sub-cube intersects [0..last]
    u8 sig[512]={0};                     // sub-cube has a nonzero in [0..last]
    // map a linear coef index -> sub-cube id
    // (z,y,x) from lin = z*256+y*16+x ; sub id = (z/cube)*perdim^2+(y/cube)*perdim+(x/cube)
    i16 survivors[AVOX]; u32 ns=0;
    for(u32 p=0;p<cnt;++p){
        u32 lin=g_scan[p]; u32 z=lin>>8,y=(lin>>4)&15,x=lin&15;
        u32 cid=(z/cube)*perdim*perdim+(y/cube)*perdim+(x/cube);
        inrange[cid]=1; if(qs[p]) sig[cid]=1;
    }
    vc_bitwriter w; vc_bw_init(&w,out,AVOX*3);
    put_golomb(&w,cnt,7);
    // CBF flags: one bit per IN-RANGE sub-cube, in sub-cube id order (decoder
    // computes in-range identically from cnt+scan).
    for(u32 c=0;c<ncubes;++c) if(inrange[c]) vc_bw_put(&w,sig[c],1);
    size_t hdr=vc_bw_finish(&w);
    // survivors in scan order: scan pos p in [0..last] whose sub-cube is sig.
    for(u32 p=0;p<cnt;++p){
        u32 lin=g_scan[p]; u32 z=lin>>8,y=(lin>>4)&15,x=lin&15;
        u32 cid=(z/cube)*perdim*perdim+(y/cube)*perdim+(x/cube);
        if(sig[cid]) survivors[ns++]=qs[p];
    }
    size_t body=0;
    if(ns) body=vc_rlgr_encode(out+hdr,AVOX*3-hdr,survivors,ns);
    *symbols=ns;                         // decoder runs through only `ns` RLGR levels
    return hdr+body;
}
static int dec_cbf(i16 *out,const u8 *in,size_t len,u32 cube){
    u32 perdim=A/cube; u32 ncubes=perdim*perdim*perdim;
    memset(out,0,AVOX*sizeof(i16));
    vc_bitreader r; vc_br_init(&r,in,len);
    u32 cnt=get_golomb(&r,7);
    // recompute in-range set from cnt (identical to encoder)
    u8 inrange[512]={0}; u8 sig[512]={0};
    for(u32 p=0;p<cnt;++p){
        u32 lin=g_scan[p]; u32 z=lin>>8,y=(lin>>4)&15,x=lin&15;
        u32 cid=(z/cube)*perdim*perdim+(y/cube)*perdim+(x/cube);
        inrange[cid]=1;
    }
    for(u32 c=0;c<ncubes;++c) if(inrange[c]) sig[c]=(u8)vc_br_get(&r,1);
    size_t hdr=br_byte_aligned(&r);      // body starts at the encoder's byte boundary
    // count survivors
    u32 ns=0;
    for(u32 p=0;p<cnt;++p){
        u32 lin=g_scan[p]; u32 z=lin>>8,y=(lin>>4)&15,x=lin&15;
        u32 cid=(z/cube)*perdim*perdim+(y/cube)*perdim+(x/cube);
        if(sig[cid]) ns++;
    }
    i16 sv[AVOX];
    if(ns) vc_rlgr_decode(sv,ns,in+hdr,len-hdr);
    // Reconstruct in SCAN order (out[p] = scan-position p level), matching how the
    // other modes return their levels and how the caller compares against qs.
    u32 si=0;
    for(u32 p=0;p<cnt;++p){
        u32 lin=g_scan[p]; u32 z=lin>>8,y=(lin>>4)&15,x=lin&15;
        u32 cid=(z/cube)*perdim*perdim+(y/cube)*perdim+(x/cube);
        if(sig[cid]) out[p]=sv[si++];
    }
    return 1;
}

// =====================================================================
typedef struct { size_t bytes; double ratio; unsigned long long symbols; double dec_mbs; int rt_ok; } modestat;

int main(int argc,char**argv){
    build_scan();
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    const float qs_list[]={16.f,32.f,64.f,128.f};
    int nfiles=2; if(argc>1){ files[0]=argv[1]; labels[0]=argv[1]; nfiles=1; }

    for(int fi=0;fi<nfiles;++fi){
        size_t len; u8 *vol=read_file(files[fi],&len);
        if(!vol||len<256*256*256){ fprintf(stderr,"missing %s\n",files[fi]); continue; }
        u32 d=256,h=256,w=256, naz=d/A,nay=h/A,nax=w/A; u32 natoms=naz*nay*nax;
        size_t raw=(size_t)d*h*w;
        printf("\n## %s  (%ux%ux%u, atom=16^3, %u atoms)\n",labels[fi],d,h,w,natoms);
        printf("entropy mode    |   q  |  bytes  | ratio  | vs-RLGR | dec syms/atom | dec MB/s | RT\n");
        printf("----------------+------+---------+--------+---------+---------------+----------+----\n");

        i16 *coef=malloc(AVOX*sizeof(i16));
        i16 *qb=malloc(AVOX*sizeof(i16));
        i16 *qs=malloc(AVOX*sizeof(i16));
        i16 *dec=malloc(AVOX*sizeof(i16));
        u8  *tmp=malloc(AVOX*4);
        u8  srcblk[AVOX];
        f32 step[AVOX];

        for(int qi=0;qi<4;++qi){
            float q=qs_list[qi];
            build_step(step,q);
            // accumulate per-mode stats over all atoms at this q (identical levels).
            const int NM=4;
            modestat st[4]={0};
            int rt[4]={1,1,1,1};
            double dect[4]={0};
            for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
                gather_atom(vol,h,w,az,ay,ax,srcblk);
                i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=srcblk[i];
                i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
                H_dct16_fwd(coef,srcblk,dc);
                quant_block(qb,coef,step);
                to_scan(qs,qb);
                // ---- MODE A: RLGR ----
                { u32 sym; size_t b=enc_rlgr(qs,tmp,&sym);
                  double t0=now_sec(); dec_rlgr(dec,tmp,b); dect[0]+=now_sec()-t0;
                  for(u32 k=0;k<AVOX;++k) if(dec[k]!=qs[k]){rt[0]=0;break;}
                  st[0].bytes+=b+2; st[0].symbols+=sym; }
                // ---- MODE B: EOB ----
                { u32 sym; size_t b=enc_eob(qs,tmp,&sym);
                  double t0=now_sec(); dec_eob(dec,tmp,b); dect[1]+=now_sec()-t0;
                  for(u32 k=0;k<AVOX;++k) if(dec[k]!=qs[k]){rt[1]=0;break;}
                  st[1].bytes+=b+2; st[1].symbols+=sym; }
                // ---- MODE C: CBF4+EOB ----
                { u32 sym; size_t b=enc_cbf(qs,tmp,4,&sym);
                  double t0=now_sec(); dec_cbf(dec,tmp,b,4); dect[2]+=now_sec()-t0;
                  for(u32 k=0;k<AVOX;++k) if(dec[k]!=qs[k]){rt[2]=0;break;}
                  st[2].bytes+=b+2; st[2].symbols+=sym; }
                // ---- MODE D: CBF8+EOB ----
                { u32 sym; size_t b=enc_cbf(qs,tmp,8,&sym);
                  double t0=now_sec(); dec_cbf(dec,tmp,b,8); dect[3]+=now_sec()-t0;
                  for(u32 k=0;k<AVOX;++k) if(dec[k]!=qs[k]){rt[3]=0;break;}
                  st[3].bytes+=b+2; st[3].symbols+=sym; }
            }
            const char *names[4]={"A RLGR (base)  ","B EOB          ","C CBF4(4^3)+EOB","D CBF8(8^3)+EOB"};
            double base_ratio = (double)raw/(double)st[0].bytes;
            for(int m=0;m<NM;++m){
                st[m].ratio=(double)raw/(double)st[m].bytes;
                double sympa=(double)st[m].symbols/(double)natoms;
                double mbs = dect[m]>0 ? raw/1e6/dect[m] : 0;
                double vs = (st[m].ratio/base_ratio - 1.0)*100.0;
                // rt: every atom must match; track via a fresh full check flag
                printf("%-15s | %4.0f | %7zu | %5.2fx | %+6.2f%% | %13.1f | %8.0f | %s\n",
                       names[m],q,st[m].bytes,st[m].ratio, (m==0?0.0:vs), sympa, mbs,
                       rt[m]?"ok":"FAIL");
            }
            printf("----------------+------+---------+--------+---------+---------------+----------+----\n");
        }
        free(coef);free(qb);free(qs);free(dec);free(tmp);free(vol);
    }
    return 0;
}

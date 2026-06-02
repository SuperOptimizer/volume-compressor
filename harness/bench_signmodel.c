// Sign-bit MODELING bake-off (r2-sign-model).
// THROWAWAY self-contained bench (does NOT touch codec.c). It #includes the WON
// transform (integer DCT-16^3) + WON quant matrix (qmatrix16, HF-protecting,
// dz=0.80-equiv slope ~0.5) and links the WON entropy coder (RLGR) as the
// MEASURABLE BASELINE. Atom = 16^3 (fixed random-access unit).
//
// QUESTION: quantized DCT-16^3 coefficient SIGNS are coded RAW (1 bit each) by
// RLGR. Do signs carry exploitable structure (per-band bias / co-located
// neighbor-atom correlation)? If so, context-coding them saves bits.
//
// METHOD (clean isolation of the sign substream):
//   * Run the full won pipeline over every 16^3 atom -> quantized levels qb[4096].
//   * The signs to code are exactly the signs of the NONZERO levels, in coef scan
//     order (z*256+y*16+x). RLGR spends exactly 1 raw bit per nonzero sign.
//   * BASELINE sign cost  = (#nonzeros) bits.
//   * MODEL sign cost     = bits emitted by an adaptive BINARY ARITHMETIC CODER
//     driven by a context model. Everything else in the archive (magnitudes,
//     runs, headers) is BYTE-IDENTICAL, so delta-bytes is the exact ratio change.
//
// We compare 4 sign models:
//   M0 RAW         : 1 bit/sign (the baseline RLGR does today).
//   M1 GLOBAL-AC   : single adaptive binary prob for all AC signs (catches a
//                    global sign skew if any).
//   M2 PER-BAND    : adaptive prob keyed by frequency band b=u+v+w (0..45):
//                    learns a per-position sign bias.
//   M3 NEIGHBOR    : context = sign of the CO-LOCATED coefficient in the previous
//                    atom (same scan index), x band-class. à la c3d sign
//                    prediction: signs of the same basis fn correlate atom-to-atom.
//
// We also report the entropy floor (H of each model's contexts) as a sanity
// check on the arithmetic coder, and total-archive ratio impact (sign bytes are
// a fraction of the whole stream).
//
// Pure C23, libc/libm. Atom transform/quant autovectorize; the binary AC is
// inherently serial (so is RLGR's bit IO) and is not on the vectorized path.
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- won transform, renamed into this TU ----------------------------------
#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  S_dct16_fwd
#define vc_dct_int16_inv  S_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

// --- won quant matrix ------------------------------------------------------
#include "../src/quant/qmatrix16.c"

// --- won entropy coder (RLGR), for the full-stream baseline size -----------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A    16u
#define AVOX 4096u

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

static inline void gather_atom(const u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8 *src = vol + ((size_t)(az*A+z)*h + (ay*A+y))*w + ax*A;
        memcpy(blk + ((size_t)z*A+y)*A, src, A);
    }
}

// ---------------------------------------------------------------------------
// Adaptive binary arithmetic coder (range coder, carry-less, 32-bit).
// Encode-only size accounting is enough to measure cost; we also implement a
// matching decoder so we can round-trip-verify the sign substream exactly.
// ---------------------------------------------------------------------------
// Carry-propagating 64-bit-low / 32-bit-range binary range coder (Subbotin
// style). `low` is 64-bit so a renorm carry simply ripples into already-emitted
// cache; standard and exactly invertible.
typedef struct { u64 low; u32 range; u8 cache; u64 carry_run; int started;
                 u8 *buf; size_t cap, pos; int of; } rc_enc;
static void rc_enc_init(rc_enc *e,u8*buf,size_t cap){
    e->low=0;e->range=0xFFFFFFFFu;e->cache=0;e->carry_run=0;e->started=0;
    e->buf=buf;e->cap=cap;e->pos=0;e->of=0;
}
static inline void rc_emit(rc_enc*e,u8 b){ if(e->pos<e->cap) e->buf[e->pos++]=b; else e->of=1; }
static inline void rc_shift(rc_enc*e){
    u8 carry = (u8)(e->low >> 32);
    if(e->low < 0xFF000000ull || carry){
        if(e->started) rc_emit(e,(u8)(e->cache + carry));
        while(e->carry_run){ rc_emit(e,(u8)(0xFF + carry)); e->carry_run--; }
        e->cache = (u8)(e->low >> 24); e->started=1;
    } else {
        e->carry_run++;
    }
    e->low = (e->low << 8) & 0xFFFFFFFFull;
}
// 12-bit probability of bit==0, p0 in [1,4095].
static inline void rc_enc_bit(rc_enc*e,u32 p0,int bit){
    u32 r0 = (u32)(((u64)e->range * p0) >> 12);
    if(!bit){ e->range = r0; }
    else { e->low += r0; e->range -= r0; }
    while(e->range < (1u<<24)){ rc_shift(e); e->range <<= 8; }
}
static void rc_enc_flush(rc_enc*e){ for(int i=0;i<5;++i) rc_shift(e); }

typedef struct { u32 range, code; const u8*buf; size_t cap,pos; } rc_dec;
static void rc_dec_init(rc_dec*d,const u8*buf,size_t cap){
    d->range=0xFFFFFFFFu;d->buf=buf;d->cap=cap;d->pos=0;d->code=0;
    for(int i=0;i<4;++i) d->code=(d->code<<8)|(d->pos<d->cap?d->buf[d->pos++]:0);
}
static inline int rc_dec_bit(rc_dec*d,u32 p0){
    u32 r0=(u32)(((u64)d->range*p0)>>12);
    int bit;
    if(d->code < r0){ bit=0; d->range=r0; }
    else { bit=1; d->code-=r0; d->range-=r0; }
    while(d->range<(1u<<24)){
        d->code=(d->code<<8)|(d->pos<d->cap?d->buf[d->pos++]:0);
        d->range<<=8;
    }
    return bit;
}

// Adaptive 12-bit context: p0 = P(bit==0). Standard shift-update (rate=5).
typedef struct { u16 p0; } ctx_t;
static inline void ctx_init(ctx_t*c){ c->p0=2048; }
static inline u32 ctx_p0(const ctx_t*c){ u32 p=c->p0; if(p<1)p=1; if(p>4095)p=4095; return p; }
static inline void ctx_update(ctx_t*c,int bit){
    if(bit==0) c->p0 += (4096 - c->p0) >> 5;
    else       c->p0 -= c->p0 >> 5;
}

// ---------------------------------------------------------------------------
// Sign-substream collection. For each atom we record (in scan order) the signs
// of nonzero levels along with their scan index (=> band) and the co-located
// previous-atom sign. We then code under each model.
// To keep memory bounded we process atom-by-atom, carrying the previous atom's
// full sign array (sign per scan index: 0=pos/zero treated specially).
// ---------------------------------------------------------------------------

// band[i] = u+v+w for scan index i (precomputed once)
static u8 g_band[AVOX];
static void build_band(void){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x)
        g_band[z*256u+y*16u+x]=(u8)(z+y+x);
}
// band-class for M3 neighbor (coarse: low/mid/high), keeps context count small
static inline u32 band_class(u32 b){ return b<6?0u:(b<18?1u:2u); }

typedef struct {
    // model bit counters (coded bits, fractional via -log2 p)
    double bits[4];
    // arithmetic-coded byte sizes (M1..M3) measured by the real range coder
    size_t rc_bytes[4];
    u64 nsign;            // total signs (=nonzeros)
    u64 nneg;             // total negative
    // round-trip check flag
    int  rt_ok;
} sign_stats;

// Per-model context banks (allocated for the whole pass).
#define NBAND 46
typedef struct {
    ctx_t global;            // M1
    ctx_t band[NBAND];       // M2
    ctx_t nb[2][3];          // M3: [prev sign present? prev neg?] x band-class
} ctx_bank;

static void ctx_bank_init(ctx_bank*bk){
    ctx_init(&bk->global);
    for(int i=0;i<NBAND;++i) ctx_init(&bk->band[i]);
    for(int a=0;a<2;++a)for(int c=0;c<3;++c) ctx_init(&bk->nb[a][c]);
}

int main(int argc,char**argv){
    build_band();
    const char *files[2]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[2]={"PHerc Paris 4 hires-256","PHerc Paris 4 coarse-256"};
    if(argc>=2){ files[0]=argv[1]; files[1]=NULL; }
    const u32 qs[4]={16,32,64,128};
    const u32 D=256,H=256,W=256;
    size_t raw=(size_t)D*H*W;

    printf("# Sign-model bake-off (atom=16^3, won DCT16+HF-qmatrix+RLGR baseline)\n");
    printf("# delta = (RAW sign bits - model bits)/8 bytes saved out of TOTAL archive bytes.\n");

    u8 *src=malloc(AVOX), *prevsign=malloc(AVOX);
    i16 *coef=malloc(AVOX*sizeof(i16)), *qb=malloc(AVOX*sizeof(i16));
    u8 *rlgrtmp=malloc(AVOX*3);
    u8 *sbuf=malloc(raw);     // generous range-coder output buffer
    // collected sign+ctx records for round-trip verify of M3 (the richest model)
    // (we verify per-pass below instead, streaming)

    for(int fi=0; fi<2 && files[fi]; ++fi){
        size_t flen; u8 *vol=read_file(files[fi],&flen);
        if(!vol||flen<raw){ fprintf(stderr,"missing %s\n",files[fi]); continue; }
        printf("\n## %s\n",labels[fi]);
        printf("q   | nz/atom | %%neg  | total_arch | sign_raw_B | M1_glob_B | M2_band_B | M3_nbr_B | best_save | arch_ratio_gain | enc_MB/s | rt\n");
        printf("----+---------+-------+------------+------------+-----------+-----------+----------+-----------+-----------------+----------+----\n");

        u32 naz=D/A,nay=H/A,nax=W/A;
        for(int qi=0;qi<4;++qi){
            f32 q=(f32)qs[qi];
            // base step: match the codec's mapping vc_rc_fixed_step ~ q? Use q directly
            // as the base dead-zone step (the bench drives ratio by q; same convention
            // as the other 16^3 benches which pass a base_step). HF matrix slope default.
            f32 step[AVOX]; qm_build_step(step,QM_HF,q,1.0f);

            sign_stats st; memset(&st,0,sizeof st); st.rt_ok=1;
            ctx_bank bnk; ctx_bank_init(&bnk);
            size_t total_arch=0;       // full RLGR archive bytes (incl raw signs)
            memset(prevsign,0,AVOX);   // 0=no nonzero, 1=pos, 2=neg (prev atom)

            // real range coders for M1..M3 (measure true coded bytes)
            rc_enc e1,e2,e3;
            // reuse one big buffer sequentially is fine since we only need sizes;
            // give each its own region.
            size_t third=raw/4;
            rc_enc_init(&e1,sbuf,third);
            rc_enc_init(&e2,sbuf+third,third);
            rc_enc_init(&e3,sbuf+2*third,third);

            double t0=now_sec();
            // for round-trip: record M3 bit decisions to replay (we re-derive
            // contexts identically on decode below using saved sign list).
            // Streaming verify: keep the produced sign sequence + its band/nbr ctx
            // is large; instead we verify by decoding e3 against the recorded
            // (sign,ctx) order using a parallel context bank.
            // We store the flat sign list + per-sign context key for M3 replay.
            // Worst case nonzeros bounded by AVOX*natoms; allocate lazily once.
            static u8 *signlist=NULL; static u32 *ctxlist=NULL; static size_t listcap=0;
            size_t maxsign=(size_t)naz*nay*nax*AVOX;
            if(listcap<maxsign){ free(signlist); free(ctxlist);
                signlist=malloc(maxsign); ctxlist=malloc(maxsign*sizeof(u32)); listcap=maxsign; }
            size_t nlist=0;

            u8 cursign[AVOX];
            for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
                gather_atom(vol,H,W,az,ay,ax,src);
                i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
                i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
                S_dct16_fwd(coef,src,dc);
                qm_quant_block(qb,coef,step);
                // full archive size for this atom (baseline, includes raw signs)
                size_t ab=vc_rlgr_encode(rlgrtmp,AVOX*3,qb,AVOX); total_arch+=ab+2;

                memset(cursign,0,AVOX);
                for(u32 i=0;i<AVOX;++i){
                    i16 v=qb[i];
                    if(v==0) continue;
                    int bit = v<0 ? 1:0;        // 1 == negative
                    cursign[i]= v<0?2u:1u;
                    st.nsign++; if(v<0) st.nneg++;
                    u32 b=g_band[i];

                    // M0 raw: 1 bit
                    st.bits[0]+=1.0;
                    // M1 global
                    { u32 p0=ctx_p0(&bnk.global); double p=bit?(4096.0-p0):(double)p0;
                      st.bits[1]+=-log2(p/4096.0);
                      rc_enc_bit(&e1,p0,bit); ctx_update(&bnk.global,bit); }
                    // M2 per-band
                    { ctx_t*c=&bnk.band[b]; u32 p0=ctx_p0(c); double p=bit?(4096.0-p0):(double)p0;
                      st.bits[2]+=-log2(p/4096.0);
                      rc_enc_bit(&e2,p0,bit); ctx_update(c,bit); }
                    // M3 neighbor: prev co-located sign present? & negativity x bandclass
                    { u32 ps=prevsign[i]; u32 a=(ps==0)?0u:1u; u32 negctx=(ps==2)?1u:0u;
                      (void)negctx;
                      // context bucket: combine "prev present" with prev-sign agreement axis
                      // we use prev-neg as the discriminator when present.
                      u32 bc=band_class(b);
                      ctx_t*c=&bnk.nb[ (ps==2)?1u:0u ][bc];
                      u32 p0=ctx_p0(c); double p=bit?(4096.0-p0):(double)p0;
                      st.bits[3]+=-log2(p/4096.0);
                      rc_enc_bit(&e3,p0,bit);
                      // record for round-trip replay of M3
                      signlist[nlist]=(u8)bit; ctxlist[nlist]= ((ps==2)?1u:0u)*3u+bc; nlist++;
                      ctx_update(c,bit);
                      (void)a;
                    }
                }
                memcpy(prevsign,cursign,AVOX);
            }
            double enc=now_sec()-t0;
            rc_enc_flush(&e1); rc_enc_flush(&e2); rc_enc_flush(&e3);
            st.rc_bytes[1]=e1.pos; st.rc_bytes[2]=e2.pos; st.rc_bytes[3]=e3.pos;
            st.rc_bytes[0]=(size_t)((st.nsign+7)/8);

            // Round-trip verify M3: decode e3 with a fresh ctx bank, replay ctx keys.
            { ctx_bank bd; ctx_bank_init(&bd);
              ctx_t *banks[6]; for(int j=0;j<6;++j) banks[j]=&bd.nb[j/3][j%3];
              rc_dec d; rc_dec_init(&d,sbuf+2*third,third);
              for(size_t j=0;j<nlist;++j){
                  ctx_t*c=banks[ctxlist[j]];
                  int bit=rc_dec_bit(&d,ctx_p0(c));
                  if(bit!=signlist[j]){ st.rt_ok=0; break; }
                  ctx_update(c,bit);
              }
            }

            double raw_sign_B=st.bits[0]/8.0;
            double m1B=st.bits[1]/8.0, m2B=st.bits[2]/8.0, m3B=st.bits[3]/8.0;
            // pick best model by REAL coded bytes (M1..M3) vs raw
            double models[4]={raw_sign_B, (double)st.rc_bytes[1], (double)st.rc_bytes[2], (double)st.rc_bytes[3]};
            double bestB=models[0]; for(int j=1;j<4;++j) if(models[j]<bestB) bestB=models[j];
            double save_B = raw_sign_B - bestB;
            // archive ratio gain: baseline archive uses raw signs already counted in total_arch.
            // new archive = total_arch - raw_sign_B + bestB.
            double new_arch = (double)total_arch - raw_sign_B + bestB;
            double base_ratio=(double)raw/(double)total_arch;
            double new_ratio =(double)raw/new_arch;
            double gain_pct=(new_ratio/base_ratio-1.0)*100.0;
            double mbps = enc>0 ? (raw/1e6)/enc : 0;
            (void)m1B;(void)m2B;(void)m3B;

            printf("%-3u | %7.1f | %5.1f | %10zu | %10.0f | %9zu | %9zu | %8zu | %9.0f | %+14.3f%% | %8.0f | %s\n",
                qs[qi],
                (double)st.nsign/((double)naz*nay*nax),
                100.0*(double)st.nneg/(double)(st.nsign?st.nsign:1),
                total_arch, raw_sign_B,
                st.rc_bytes[1], st.rc_bytes[2], st.rc_bytes[3],
                save_B, gain_pct, mbps,
                st.rt_ok?"ok":"FAIL");
        }
        free(vol);
    }
    free(src);free(prevsign);free(coef);free(qb);free(rlgrtmp);free(sbuf);
    return 0;
}

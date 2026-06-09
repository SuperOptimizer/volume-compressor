// Separable 3D DCT-II / inverse for variable block sizes (4/8/16/32), float.
// No reversibility/determinism requirement (v2 is lossy-only) — float for
// speed/simplicity; precision optimization (int/fp16) is a later concern.
// Matrices generated once per size. Orthonormal so inverse = transpose.
#ifndef V2_DCT_H
#define V2_DCT_H
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DCT_MAXN 32
// dct_mat[S][k][n] for S in {4,8,16,32}; indexed by log2(S).
static float g_dctm[6][DCT_MAXN][DCT_MAXN];   // [log2(S)][k][n]
static int   g_dct_ready = 0;

static void dct_build(void) {
    if (g_dct_ready) return;
    for (int S = 4; S <= DCT_MAXN; S <<= 1) {
        int l = 0, t = S; while (t > 1) { t >>= 1; l++; }   // log2
        for (int k = 0; k < S; ++k) {
            double ck = (k == 0) ? sqrt(1.0/S) : sqrt(2.0/S);
            for (int n = 0; n < S; ++n)
                g_dctm[l][k][n] = (float)(ck * cos(M_PI*(2.0*n+1.0)*k/(2.0*S)));
        }
    }
    g_dct_ready = 1;
}

static inline int dct_log2(int S){ int l=0; while(S>1){S>>=1;l++;} return l; }

// 1D DCT-II forward along contiguous lines: out[k]=sum_n M[k][n] in[n].
static void dct1d_fwd(const float *m, const float *in, float *out, int S) {
    for (int k = 0; k < S; ++k) { float a=0; const float *mk=m+(size_t)k*DCT_MAXN; for (int n=0;n<S;++n) a += mk[n]*in[n]; out[k]=a; }
}
static void dct1d_inv(const float *m, const float *in, float *out, int S) {
    for (int n = 0; n < S; ++n) { float a=0; for (int k=0;k<S;++k) a += m[(size_t)k*DCT_MAXN+n]*in[k]; out[n]=a; }
}

// 3D forward DCT of an S^3 block (row-major z,y,x). coef same layout.
static void dct3_fwd(const float *blk, float *coef, int S) {
    const float *m = &g_dctm[dct_log2(S)][0][0];
    static _Thread_local float a[DCT_MAXN*DCT_MAXN*DCT_MAXN], b[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    float line[DCT_MAXN], oline[DCT_MAXN];
    size_t S2=(size_t)S*S;
    // along x (contiguous)
    for (size_t off=0; off<(size_t)S*S; ++off) { dct1d_fwd(m, blk+off*S, a+off*S, S); }
    // along y
    for (int z=0;z<S;++z) for (int x=0;x<S;++x){
        for(int y=0;y<S;++y) line[y]=a[((size_t)z*S+y)*S+x];
        dct1d_fwd(m,line,oline,S);
        for(int y=0;y<S;++y) b[((size_t)z*S+y)*S+x]=oline[y];
    }
    // along z
    for (int y=0;y<S;++y) for (int x=0;x<S;++x){
        for(int z=0;z<S;++z) line[z]=b[((size_t)z*S+y)*S+x];
        dct1d_fwd(m,line,oline,S);
        for(int z=0;z<S;++z) coef[((size_t)z*S+y)*S+x]=oline[z];
    }
    (void)S2;
}
static void dct3_inv(const float *coef, float *blk, int S) {
    const float *m = &g_dctm[dct_log2(S)][0][0];
    static _Thread_local float a[DCT_MAXN*DCT_MAXN*DCT_MAXN], b[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    float line[DCT_MAXN], oline[DCT_MAXN];
    for (int y=0;y<S;++y) for (int x=0;x<S;++x){
        for(int z=0;z<S;++z) line[z]=coef[((size_t)z*S+y)*S+x];
        dct1d_inv(m,line,oline,S);
        for(int z=0;z<S;++z) a[((size_t)z*S+y)*S+x]=oline[z];
    }
    for (int z=0;z<S;++z) for (int x=0;x<S;++x){
        for(int y=0;y<S;++y) line[y]=a[((size_t)z*S+y)*S+x];
        dct1d_inv(m,line,oline,S);
        for(int y=0;y<S;++y) b[((size_t)z*S+y)*S+x]=oline[y];
    }
    for (size_t off=0; off<(size_t)S*S; ++off) dct1d_inv(m, b+off*S, blk+off*S, S);
}
#endif

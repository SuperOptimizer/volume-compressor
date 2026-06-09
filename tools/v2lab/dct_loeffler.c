// Recursive fast DCT-II (O(N log N)) for N=16, float reference, verified against the
// direct matmul before any integer/codec port. Decomposition (orthonormal DCT-II):
//   s[n]=x[n]+x[N-1-n], d[n]=x[n]-x[N-1-n], n<H=N/2
//   X[2m]   = sqrt(H/N) * DCTII_H(s)[m]            (clean recursion)
//   X[2m+1] = DCTIV_H(d)[m] * (orthonormal scaling)   (the fiddly odd half)
// We implement DCT-IV recursively too is overkill for N=16; do DCT-IV by its own
// fast even/odd OR (pragmatic) a direct H-point DCT-IV (H=8 -> 64 MACs) — still far
// fewer than the full 16-pt direct (256) or partial-butterfly (128). Verify exact.
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double Cd[33][33];     // direct DCT-II matrix cache for verify (per N)
static void buildC(int N,double M[][33]){
    for(int k=0;k<N;++k){ double ck=(k==0)?sqrt(1.0/N):sqrt(2.0/N);
        for(int n=0;n<N;++n) M[k][n]=ck*cos(M_PI*(2*n+1)*k/(2.0*N)); }
}
static void directII(int N,const double*x,double*X){
    static double M[33][33]; buildC(N,M);
    for(int k=0;k<N;++k){ double a=0; for(int n=0;n<N;++n) a+=M[k][n]*x[n]; X[k]=a; }
}
// direct DCT-IV (H point): Y[k] = sqrt(2/H) sum_n x[n] cos(pi/H (n+1/2)(k+1/2))
static void directIV(int H,const double*x,double*Y){
    for(int k=0;k<H;++k){ double a=0; for(int n=0;n<H;++n) a+=cos(M_PI/H*(n+0.5)*(k+0.5))*x[n]; Y[k]=sqrt(2.0/H)*a; }
}
// fast DCT-II via 1-level even/odd split (recursion on even half, direct DCT-IV on odd)
static void fastII(int N,const double*x,double*X){
    if(N==1){ X[0]=x[0]; return; }
    int H=N/2; double s[33],d[33],Xe[33],Xo[33];
    for(int n=0;n<H;++n){ s[n]=x[n]+x[N-1-n]; d[n]=x[n]-x[N-1-n]; }
    fastII(H,s,Xe);                      // even outputs (recurse)
    directIV(H,d,Xo);                    // odd outputs (DCT-IV of diffs)
    double scale=sqrt((double)H/N);   // BOTH halves scale by sqrt(H/N) (verified empirically)
    for(int m=0;m<H;++m){ X[2*m]=scale*Xe[m]; X[2*m+1]=scale*Xo[m]; }
}
int main(){
    for(int N=4;N<=16;N<<=1){
        double x[16],Xd[16],Xf[16]; srand(N);
        for(int i=0;i<N;++i) x[i]=(rand()%200-100)/10.0;
        directII(N,x,Xd); fastII(N,x,Xf);
        double maxe=0; for(int i=0;i<N;++i){ double e=fabs(Xd[i]-Xf[i]); if(e>maxe)maxe=e; }
        printf("N=%2d: max|direct-fast| = %.2e  %s\n",N,maxe, maxe<1e-9?"OK":"*** MISMATCH ***");
    }
    return 0;
}

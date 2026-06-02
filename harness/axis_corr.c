// Axis-correlation diagnostic: which array axis is the MOST correlated direction?
// Volume is ZYX (z = first/outermost index). Computes lag-1 Pearson correlation
// of adjacent voxels along each of the 3 axes, plus mean |delta|, to verify the
// user's domain claim that Z (page top->bottom) is most correlated.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

static uint8_t *rd(const char*p, size_t n){
    FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    uint8_t*b=malloc(n); if(fread(b,1,n,f)!=n){fprintf(stderr,"short\n");exit(1);}
    fclose(f); return b;
}

// lag-1 correlation along an axis given stride (in voxels) and count of valid pairs.
static void corr_axis(const uint8_t*v,size_t D,size_t H,size_t W,int axis,
                      double*rho,double*mad){
    double sx=0,sy=0,sxx=0,syy=0,sxy=0,sad=0; size_t n=0;
    for(size_t z=0;z<D;++z)for(size_t y=0;y<H;++y)for(size_t x=0;x<W;++x){
        size_t i=(z*H+y)*W+x; size_t z2=z,y2=y,x2=x;
        if(axis==0){ if(z+1>=D)continue; z2=z+1; }
        if(axis==1){ if(y+1>=H)continue; y2=y+1; }
        if(axis==2){ if(x+1>=W)continue; x2=x+1; }
        size_t j=(z2*H+y2)*W+x2;
        double a=v[i],b=v[j];
        sx+=a; sy+=b; sxx+=a*a; syy+=b*b; sxy+=a*b; sad+=fabs(a-b); n++;
    }
    double mx=sx/n,my=sy/n;
    double cov=sxy/n-mx*my, vx=sxx/n-mx*mx, vy=syy/n-my*my;
    *rho=cov/sqrt(vx*vy); *mad=sad/n;
}

int main(int argc,char**argv){
    const char*files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char*lab[]={"hires-256","coarse-256"};
    size_t D=256,H=256,W=256,N=D*H*W;
    for(int f=0;f<2;++f){
        uint8_t*v=rd(files[f],N);
        printf("\n== %s (ZYX, Z=axis0=outermost=page top->bottom) ==\n",lab[f]);
        printf("axis     | lag-1 rho | mean|delta|\n");
        const char*an[]={"Z (axis0)","Y (axis1)","X (axis2)"};
        for(int a=0;a<3;++a){ double r,m; corr_axis(v,D,H,W,a,&r,&m);
            printf("%-9s| %8.5f | %8.4f\n",an[a],r,m); }
        free(v);
    }
    return 0;
}

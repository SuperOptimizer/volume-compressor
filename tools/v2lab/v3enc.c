// v3enc — frozen-library CLI: build a .v3 archive from a zarr volume.
// usage: v3enc <zarr_root> <out.v3> [quality=8] [dim=1024]
#include "v3archive_api.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <zarr_root> <out.v3> [quality=8] [dim=1024]\n",argv[0]); return 1; }
    float q = argc>3 ? (float)atof(argv[3]) : 8.0f;
    int dim = argc>4 ? atoi(argv[4]) : 1024;
    int rc = v3_build_from_zarr(argv[1],argv[2],dim,q);
    if(rc==0) fprintf(stderr,"wrote %s (q=%.1f, dim=%d)\n",argv[2],q,dim);
    return rc;
}

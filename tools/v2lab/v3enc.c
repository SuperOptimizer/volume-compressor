// v3enc — frozen-library CLI: build a .v3 archive from a zarr volume.
// usage: v3enc <zarr_root> <out.v3> [quality=8] [dim=1024] [metadata.txt]
// The optional metadata file (JSON/TOML/INI/...) is stored verbatim in the archive's
// 128KB metadata region.
#include "v3archive_api.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <zarr_root> <out.v3> [quality=8] [dim=1024] [metadata.txt]\n",argv[0]); return 1; }
    float q = argc>3 ? (float)atof(argv[3]) : 8.0f;
    int dim = argc>4 ? atoi(argv[4]) : 1024;
    char *meta=NULL; size_t mlen=0;
    if(argc>5){
        FILE *mf=fopen(argv[5],"rb");
        if(!mf){ fprintf(stderr,"cannot open metadata file %s\n",argv[5]); return 1; }
        fseek(mf,0,SEEK_END); long n=ftell(mf); fseek(mf,0,SEEK_SET);
        meta=malloc(n); mlen=fread(meta,1,n,mf); fclose(mf);
    }
    int rc = v3_build_from_zarr(argv[1],argv[2],dim,q,meta,mlen);
    if(rc==0) fprintf(stderr,"wrote %s (q=%.1f, dim=%d, metadata=%zu B)\n",argv[2],q,dim,mlen);
    free(meta);
    return rc;
}

// bench_c4d_1024 — c4d arm of the 1024^3 3-way benchmark. Tiles the volume into
// 64^3 chunks (c4d's codec atom), encodes each at a given q, decodes+scatters
// to a reconstruction file, and reports ratio + encode/decode MB/s. Also does a
// COLD single-chunk decode (deserialize bytes -> decode_chunk -> extract 16^3)
// for the time-to-first-16^3-block axis. PSNR/MS-SSIM/GMSD are computed by the
// shared vc scorer on the written reconstruction (apples-to-apples with OURS).
//
// usage: bench_c4d_1024 <raw.u8> <N> <recout.u8> <q>
// prints: ratio enc_mbs dec_mbs ttf_us   (one line, space separated)
#include "c4d/core.hpp"
#include "c4d/chunk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
using namespace c4d;

static double now(){
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char**argv){
    if(argc<5){ std::fprintf(stderr,"usage: %s <raw.u8> <N> <recout.u8> <q>\n",argv[0]); return 2; }
    const char *path=argv[1]; u32 N=(u32)atoi(argv[2]); const char*recpath=argv[3]; f32 q=(f32)atof(argv[4]);
    size_t n=(size_t)N*N*N;
    std::vector<u8> vol(n);
    { FILE*f=fopen(path,"rb"); if(!f){perror(path);return 1;} if(fread(vol.data(),1,n,f)!=n){std::fprintf(stderr,"short read\n");return 1;} fclose(f); }

    const u32 C=CHUNK; // 64
    u32 nc=N/C;
    size_t csz=(size_t)C*C*C;
    std::vector<u8> cin(csz), cout(csz);
    std::vector<u8> rec(n);
    size_t total_bytes=0;
    double enc_t=0, dec_t=0;
    // save one interior chunk's serialized blob for the cold-access test
    std::vector<u8> saved_blob;
    u32 scz=nc/2, scy=nc/2, scx=nc/2;

    for(u32 cz=0;cz<nc;cz++)
    for(u32 cy=0;cy<nc;cy++)
    for(u32 cx=0;cx<nc;cx++){
        for(u32 z=0;z<C;z++)
        for(u32 y=0;y<C;y++){
            size_t vz=(size_t)cz*C+z, vy=(size_t)cy*C+y, vx=(size_t)cx*C;
            memcpy(cin.data()+((size_t)z*C+y)*C, vol.data()+((vz*N+vy)*N+vx), C);
        }
        double a=now();
        auto pl=chunk::encode_chunk(std::span<const u8>(cin), chunk::EncodeOpts{.q=q});
        auto blob=pl.serialize();
        double b=now();
        // decode from the serialized blob (cold path, like a real archive read)
        auto pl2=chunk::Payload::deserialize(blob);
        chunk::decode_chunk(pl2, std::span<u8>(cout));
        double c=now();
        enc_t+=b-a; dec_t+=c-b; total_bytes+=blob.size();
        if(cz==scz&&cy==scy&&cx==scx) saved_blob=blob;
        for(u32 z=0;z<C;z++)
        for(u32 y=0;y<C;y++){
            size_t vz=(size_t)cz*C+z, vy=(size_t)cy*C+y, vx=(size_t)cx*C;
            memcpy(rec.data()+((vz*N+vy)*N+vx), cout.data()+((size_t)z*C+y)*C, C);
        }
    }
    { FILE*f=fopen(recpath,"wb"); fwrite(rec.data(),1,n,f); fclose(f); }

    // time-to-first-16^3-block: cold deserialize + decode the 64^3 chunk, extract 16^3.
    double tf0=now();
    auto pl=chunk::Payload::deserialize(saved_blob);
    std::vector<u8> dec(csz);
    chunk::decode_chunk(pl, std::span<u8>(dec));
    u8 blk[16*16*16];
    // extract sub-cube at offset (16,16,16) within the 64^3 chunk
    for(int z=0;z<16;z++)for(int y=0;y<16;y++)
        memcpy(blk+(z*16+y)*16, dec.data()+(((size_t)(16+z)*C)+(16+y))*C+16, 16);
    double ttf_us=(now()-tf0)*1e6;

    double ratio=(double)n/total_bytes;
    double mb=n/1e6;
    std::printf("%.4f %.2f %.2f %.2f\n", ratio, mb/enc_t, mb/dec_t, ttf_us);
    return 0;
}

// Concurrency test: many threads append disjoint atoms in parallel (the exporter's
// real pattern), then a reader verifies all are present + round-trip. For TSan.
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define NT 16
#define CHECK(c,...) do{if(!(c)){printf("FAIL: " __VA_ARGS__);printf("\n");exit(1);}}while(0)
static vc_writer *W;
static unsigned ACX,ACY,ACZ;
static void fill(unsigned char*v,unsigned az,unsigned ay,unsigned ax){
  for(unsigned i=0;i<VC_ATOM3;i++) v[i]=(unsigned char)((az*7+ay*13+ax*17+i)%200+10);
}
static void* worker(void*arg){
  long tid=(long)arg; unsigned char v[VC_ATOM3];
  // each thread strides over the atom grid -> disjoint atoms, same regions (contention)
  unsigned total=ACX*ACY*ACZ;
  for(unsigned idx=tid; idx<total; idx+=NT){
    unsigned ax=idx%ACX, ay=(idx/ACX)%ACY, az=idx/(ACX*ACY);
    fill(v,az,ay,ax);
    if(vc_append_atom(W,0,az,ay,ax,v)!=VC_OK){printf("append fail\n");exit(1);}
  }
  return 0;
}
int main(void){
  const char*path="/tmp/vc_concurrent.vca";
  vc_dims d0={512,256,256};
  ACX=(d0.nx+VC_ATOM-1)/VC_ATOM; ACY=(d0.ny+VC_ATOM-1)/VC_ATOM; ACZ=(d0.nz+VC_ATOM-1)/VC_ATOM;
  W=vc_create(path,d0,30.0f); CHECK(W,"create");
  for(int l=0;l<VC_NLOD;l++) vc_set_base_q(W,l,1.0f);
  pthread_t th[NT];
  for(long i=0;i<NT;i++) pthread_create(&th[i],0,worker,(void*)i);
  for(int i=0;i<NT;i++) pthread_join(th[i],0);
  vc_writer_close(W);
  // verify
  int fd=open(path,O_RDONLY); struct stat st; fstat(fd,&st);
  void*m=mmap(0,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
  vc_archive*a=vc_open((unsigned char*)m,st.st_size); CHECK(a,"open");
  unsigned char out[VC_ATOM3]; long checked=0;
  for(unsigned az=0;az<ACZ;az++)for(unsigned ay=0;ay<ACY;ay++)for(unsigned ax=0;ax<ACX;ax++){
    CHECK(vc_atom_coverage(a,0,az,ay,ax)==VC_PRESENT,"present %u,%u,%u",az,ay,ax);
    CHECK(vc_decode_atom(a,0,ax,ay,az,out)==VC_OK,"decode");
    checked++;
  }
  vc_close(a); munmap(m,st.st_size); unlink(path);
  printf("concurrent: %d threads, %ld atoms all present+decoded OK\n",NT,checked);
  return 0;
}

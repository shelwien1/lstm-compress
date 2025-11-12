
typedef unsigned char  byte;
typedef unsigned int uint;
typedef unsigned long long qword;

#include "lstm-model.inc"

#define INC_FLEN
#include "common.inc"

#include "sh_v2f.inc"

Rangecoder rc;

static const uint CNUM = 256;

char cmap[CNUM];

int main( int argc, char** argv ) {

  if( argc<3 ) return 1;

  uint f_DEC = (argv[1][0]=='d');
  FILE* f = fopen(argv[2],"rb"); if( f==0 ) return 2;
  FILE* g = fopen(argv[3],"wb"); if( g==0 ) return 3;

  uint i,c,code,low,total=0,freq[CNUM],f_len,f_pos;
  for( i=0; i<CNUM; i++ ) total+=(freq[i]=1);

  for( i=0; i<CNUM; i++ ) cmap[i]=0;

  if( f_DEC==0 ) {
    f_len = flen(f);
    fwrite( &f_len, 1,sizeof(f_len), g );

    for( f_pos=0; f_pos<f_len; f_pos++ ) cmap[getc(f)]=1;

    fseek( f, 0, SEEK_SET );

    rc.StartEncode(g); 

  } else {
    f_len = 0;
    fread( &f_len, 1,sizeof(f_len), f );
    rc.StartDecode(f);
  }

  for( i=0,total=0; i<CNUM; i++ ) total+=( cmap[i]=rc.rc_BProcess(SCALE/2,cmap[i]) );

  srand(0xDEADBEEF);
  ByteModel* PM = new ByteModel( cmap, new Lstm(0, total, 90, 3, 10, 0.05, 2) );

  for( f_pos=0; f_pos<f_len; f_pos++ ) {

    for( i=0,total=0; i<CNUM; i++ ) {
      freq[i] = PM->probs_[i]*SCALE;
      freq[i] += ((freq[i]==0) & cmap[i]);
      total += freq[i];
    }

    if( f_DEC==0 ) {
      c = getc(f); 
      for( i=0,low=0; i<c; i++ ) low+=freq[i];
      rc.rc_Process(low,freq[c],total);
    } else {
      code = rc.rc_GetFreq(total);
      for( c=0,low=0; low+freq[c]<=code; c++ ) low+=freq[c];
      rc.rc_Process(low,freq[c],total);
    }

    if( f_DEC==1 ) putc(c,g);

    PM->Update( c );
  }

  if( f_DEC==0 ) rc.FinishEncode();

  fclose(g);
  fclose(f);

  return 0;
}

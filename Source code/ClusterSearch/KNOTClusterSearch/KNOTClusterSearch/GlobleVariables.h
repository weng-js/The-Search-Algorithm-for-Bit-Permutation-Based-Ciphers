#ifndef _GLOBLEVARIABLES_H_
#define _GLOBLEVARIABLES_H_
#include<vector>
#include<emmintrin.h>
#include<map>
using namespace std;
typedef unsigned char  u8; //8-bit
typedef unsigned short u16;

#define ALIGNED_(x) __declspec(align(x))
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x)

#define Block_SIZE   256 // 256 384 512
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define State_NUM    (Block_SIZE/128)
#define ARR_LEN      (SBox_NUM/2) // SBox_NUM/2
#define STATE_LEN    (sizeof(__m128i)*State_NUM)

#define TYPE 0 //Diff
//#define TYPE 1 //Linear

#if(Block_SIZE==256)
#define RNUM 52
#elif(Block_SIZE==384)
#define RNUM 76
#else
#define RNUM 100
#endif

extern int BestB[RNUM + 1];
extern double weight[4];

extern double FWWeightMinandMax[SBox_SIZE][2];   
extern u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) FWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

extern int Sbox_loc[SBox_NUM][3]; 
extern ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE][State_NUM];
extern ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE][State_NUM];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];

extern int Rnum;
extern int Bn;

typedef struct STATE {
	int rnum;    //record round
	int j;       //record sbox
	int W;    //record weight of 1~rnum-1
	int w;    //record weight of rnum

	u8 sbx_a[ARR_LEN]; // [0] record active sbox  
	u8 sbx_in[ARR_LEN];// [1] record input
	int sbx_num;        //record: num of active sbx - 1

	STATE() : rnum(1), j(0), w(0), W(0), sbx_num(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
	};
	STATE(int data, double data_w) : rnum(data), j(0), w(0), W(data_w), sbx_num(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
	};

	~STATE() {};
}STATE;


#endif // !_GLOBLEVARIABLES_H_




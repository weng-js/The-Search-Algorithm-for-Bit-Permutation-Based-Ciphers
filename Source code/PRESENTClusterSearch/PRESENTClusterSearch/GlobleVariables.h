#ifndef _GLOBLEVARIABLES_H_
#define _GLOBLEVARIABLES_H_
#include<vector>
#include<emmintrin.h>
#include<map>
using namespace std;

typedef unsigned char  u8;		//8-bit
typedef unsigned short u16;
typedef unsigned long long u64; //64-bit

#define ALIGNED_(x) __declspec(align(x))
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) 

#define Block_SIZE   64  
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define ARR_LEN      16 // SBox_NUM/2
#define STATE_LEN    (sizeof(__m128i))
#define Group_NUM    4

//#define TYPE 0 //Diff
#define TYPE 1 //Linear
#define RNUM 31

extern int BestB[RNUM + 1];
extern double weight[4];
extern int WeightLen;


extern int FWWeightMinandMax[SBox_SIZE][2];     
extern u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; // sbox*input*output
             
extern int Sbox_loc[SBox_NUM];                        
extern int FWGroup_SBox[Group_NUM][4];
extern ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];

extern int Rnum;
extern int Bn;

typedef struct STATE {
	int rnum;    //record round
	u8 j;       //record sbox
	int W;    //record weight of 1~rnum-1
	int w;    //record weight of rnum

	u8 sbx_a[ARR_LEN]; // [0] record active sbox  
	u8 sbx_in[ARR_LEN];// [1] record input
	u8 sbx_g[ARR_LEN]; 
	int sbx_num;        //record: num of active sbx - 1
	int g_num;
	int nr_minw;      

	STATE() : rnum(1), j(0), w(0), W(0), sbx_num(0), g_num(0), nr_minw(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_g, 0, ARR_LEN);
	};
	STATE(int data, int data_w) : rnum(data), j(0), w(0), W(data_w), sbx_num(0), g_num(0), nr_minw(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_g, 0, ARR_LEN);
	};

	~STATE() {};
}STATE;

#endif // !_GLOBLEVARIABLES_H_




#ifndef _GLOBLEVARIABLES_H_
#define _GLOBLEVARIABLES_H_
#include<vector>
#include<emmintrin.h>
#include<nmmintrin.h>
#include<fstream>
#include<map>
using namespace std;

typedef unsigned char       u8; //8-bit
typedef unsigned short	   u16;
typedef unsigned long long u64; //64-bit

#define ALIGNED_(x) __declspec(align(x))
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) 

#define Block_SIZE   64  
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define ARR_LEN      16
#define STATE_LEN    (sizeof(__m128i))

#define TYPE 0 //Diff
//#define TYPE 1 //Linear 
#define RNUM 41 

extern int BestB[RNUM + 1];
extern int BestASN[RNUM + 1]; 
extern double weight[4];
extern int WeightLen;

extern int SBoxPermutation[SBox_NUM];    
extern int INVSBoxPermutation[SBox_NUM]; 
extern int State1Permutation[SBox_NUM];    
extern int INVState1Permutation[SBox_NUM]; 
extern int State2Permutation[SBox_NUM];    //P1
extern int INVState2Permutation[SBox_NUM]; //P1

extern __m128i State1PermutationSSE;
extern __m128i INVState1PermutationSSE;
extern __m128i SBoxPermutationSSE;
extern __m128i INVSBoxPermutationSSE;

extern int FWWeightMinandMax[SBox_SIZE][2];        
extern u8 FWWeightOrderValue[SBox_SIZE][SBox_SIZE]; 
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern double FWWeightOrderWMinusMin[SBox_SIZE][SBox_SIZE];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];
extern double DDTorLATMinusMin[SBox_SIZE][SBox_SIZE];

extern int Rnum;
extern int Bn;

typedef struct STATE {
	int rnum;      //record round
	u8 j;          //record sbox
	int W;         //record weight of 1~rnum-1
	int w;		   //record weight of rnum
	int nr_minw;   
	u8 sbx_a[ARR_LEN];	 // [0]record active sbox  
	u8 sbx_in[ARR_LEN];
	bool sbx_tag[ARR_LEN];
	int sbx_num;        //record: num of active sbx - 1
	int nr_sbx_num;

	STATE() : rnum(1), j(0), w(0), W(0), sbx_num(0), nr_minw(0), nr_sbx_num(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_tag, 0, ARR_LEN);
	};
	STATE(int data_r, int data_W) : rnum(data_r), j(0), w(0), W(data_W), sbx_num(0), nr_minw(0), nr_sbx_num(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_tag, 0, ARR_LEN);
	};

	~STATE() {};
}STATE;

#endif // !_GLOBLEVARIABLES_H_




#ifndef _GLOBLEVARIABLES_H_
#define _GLOBLEVARIABLES_H_

#include<vector>
#include<emmintrin.h>
#include<fstream>
#include<map>
using namespace std;

typedef unsigned char       u8; //8-bit
typedef unsigned short     u16; //16-bit
typedef unsigned long long u64; //64-bit

#define ALIGNED_(x) __declspec(align(x))
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) 

#define Block_SIZE   32  
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define ARR_LEN      8
#define STATE_LEN    (sizeof(__m128i))
#define ASP_NUM      255 

//#define TYPE 0 //Diff
#define TYPE 1 //Linear 
#define RNUM 32 //32

#define PreRange (RNUM-1)

extern int BestB[RNUM + 1];
extern double weight[4];
extern int WeightLen;

extern int FWSBoxPermutation[SBox_NUM];    
extern int FWINVSBoxPermutation[SBox_NUM]; 

extern int BWSBoxPermutation[SBox_NUM];   

extern int FWSBoxROT[SBox_NUM]; 
extern int BWSBoxROT[SBox_NUM]; 

extern u8  IRFWMinV[SBox_NUM][SBox_SIZE]; 
extern int IRFWMinW[SBox_NUM][SBox_SIZE];

//forward:
extern int FWWeightMinandMax[SBox_NUM][SBox_SIZE][2];        
extern u8  FWWeightOrderV[SBox_NUM][SBox_SIZE][SBox_SIZE];
extern double FWWeightOrderW[SBox_NUM][SBox_SIZE][SBox_SIZE];
extern double FWWeightOrderWMinusMin[SBox_NUM][SBox_SIZE][SBox_SIZE];


//DC value
extern int LB[RNUM + 1][2]; //NA = 0, NA = 1
//ASP
extern u8 ASP_FW_Info[ASP_NUM][ARR_LEN]; 
extern u8 ASP_BW_Info[ASP_NUM][ARR_LEN];
extern int ASPInfo[ASP_NUM]; 

//LB[D_r_0_i_ASP] = LBNA0[r,i,ASP] = max(BW[i-1,ASP]+FW[r-i,ASP],LBNA0[r,i,ASP])
extern int LBNA0[RNUM + 1][RNUM + 1][ASP_NUM];
extern int BWLB[RNUM][ASP_NUM];     
extern int FWLB[RNUM][ASP_NUM];		
extern int ASNLB[RNUM + 1][2];
extern int ASNBWLB[RNUM][ASP_NUM]; 
extern int ASNFWLB[RNUM][ASP_NUM];
extern bool ASNBWLBOver[RNUM][ASP_NUM]; 
extern bool ASNFWLBOver[RNUM][ASP_NUM];

//Total
extern double DDTorLAT[SBox_NUM][SBox_SIZE][SBox_SIZE];
extern double DDTorLATMinusMinW[SBox_NUM][SBox_SIZE][SBox_SIZE];
extern int Rnum;
extern int Bn;
extern FILE* fp;


class RecordLBForValue { 
public:
	int Round_NUM;          
	u8  RecordRnum[RNUM];   
	u8  RecordRdLB[RNUM];   

	RecordLBForValue() : Round_NUM(0) {
		memset(RecordRnum, 0, sizeof(RecordRnum));
		memset(RecordRdLB, 0, sizeof(RecordRdLB));
	};

	RecordLBForValue(int r, int lb) :Round_NUM(1) {
		memset(RecordRnum, 0, sizeof(RecordRnum));
		memset(RecordRdLB, 0, sizeof(RecordRdLB));
		RecordRnum[0] = r;
		RecordRdLB[r - 1] = lb;
	}

	~RecordLBForValue() {};

	int  ReturnLB(int r, int asp);          
	void UpdateOrInsertLB(int r, int lb);  
};

class RecordLBForValue_ASN {	
public:
	int Round_NUM;            
	u8  RecordRnum[RNUM];     
	u8  RecordRdLB[RNUM];     
	bool RdLBOver[RNUM];

	RecordLBForValue_ASN() : Round_NUM(0) {
		memset(RecordRnum, 0, sizeof(RecordRnum));
		memset(RecordRdLB, 0, sizeof(RecordRdLB));
		memset(RdLBOver, 0, sizeof(RdLBOver));
	};

	RecordLBForValue_ASN(int r, int lb) :Round_NUM(1) {
		memset(RecordRnum, 0, sizeof(RecordRnum));
		memset(RecordRdLB, 0, sizeof(RecordRdLB));
		memset(RdLBOver, 0, sizeof(RdLBOver));
		RecordRnum[0] = r;
		RecordRdLB[r - 1] = lb;
	}

	~RecordLBForValue_ASN() {};

	int  ReturnLB(int r, int asp);       
	void UpdateOrInsertLB(int r, int lb); 
};


typedef struct STATE {
	int rnum;     //record round
	u8 j;         //record sbox
	int W;        //record weight of 1~rnum-1 / or ASN
	int w;        //record weight of rnum     / or ASN
	int nr_minw;  

	u8 sbx_a[ARR_LEN]; // [0] record active sbox  
	u8 sbx_in[ARR_LEN];
	bool sbx_tag[ARR_LEN];
	int sbx_num;        //record: num of active sbx - 1
	int nr_sbx_num;

	STATE() : rnum(1), j(0), w(0), W(0), sbx_num(0), nr_minw(0), nr_sbx_num(0) { 
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_tag, 0, ARR_LEN);
	};

	STATE(int data, int data_w) : rnum(data), j(0), w(0), W(data_w), sbx_num(0), nr_minw(0), nr_sbx_num(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_tag, 0, ARR_LEN);
	};

	~STATE() {};
}STATE;



#endif // !_GLOBLEVARIABLES_H_




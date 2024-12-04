#ifndef _GLOBLEVARIABLES_H_
#define _GLOBLEVARIABLES_H_
#include<vector>
#include<emmintrin.h>
#include<fstream>
#include<map>
using namespace std;

typedef unsigned char  u8; //8-bit
typedef unsigned short u16;

#define ALIGNED_(x) __declspec(align(x))
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) 

#define Block_SIZE   64 
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define ARR_LEN      16 // SBox_NUM/2
#define STATE_LEN    (sizeof(__m128i))
#define NA1_NUM      15  
#define NA2_NUM		 225  //15*15
#define NA2_SBoxNUM  120 //(16*15)/2
#define Group_NUM    4
#define Na12UBRnum 10
#define Na3UBRnum  5

//#define TYPE 0 //Diff 0 0
#define TYPE 1 //Linear 10 5
#define RNUM 31 // 


extern int BestB[RNUM + 1];
extern double weight[4];
extern int WeightLen;


extern u8 Round1MinIndex[SBox_SIZE];
extern int Round1MinW[SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) Round1MinSPTable[SBox_NUM][SBox_SIZE]; //sbox*output*state_NUM

//forward search variables
extern int FWWeightMinandMax[SBox_SIZE][2];     
extern u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; // sbox*input*output

//backward search variables
extern int BWWeightMinandMax[SBox_SIZE][2]; 
extern u8 BWWeightOrderIndex[SBox_SIZE][SBox_SIZE];
extern double BWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) BWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; // sbox*input*output*state_NUM

//SBox
extern int INVSbox_loc[SBox_NUM][2];         
extern int Sbox_loc[SBox_NUM];                        
extern int FWGroup_SBox[Group_NUM][4];
extern int BWGroup_SBox[Group_NUM][4];
extern ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];


// NA==1
extern int Na1RoundNPInput[NA1_NUM];           
extern int Na1RoundNPInfo[NA1_NUM][2];						      //[0]fwMinw, [1]bwMinw
extern int Na1InputIndex[SBox_SIZE];           
extern int Na1InOutLink[NA1_NUM][NA1_NUM];						  //Index of input-compatible outputs
extern double Na1OutWeightOrder[NA1_NUM][NA1_NUM];				  //Weight: ascending order
extern int Na1RoundNPFWMinW[NA1_NUM][SBox_NUM];					  //minw 
extern int Na1RoundNPFWASNandG[NA1_NUM][SBox_NUM][2];			  //Record ASN as well as group
extern u8 Na1RoundNPFWARRInfo[NA1_NUM][SBox_NUM][3][ARR_LEN/2];   //Record asn_index, asn_input and the corresponding group.
extern int Na1FWMinW[RNUM][NA1_NUM][SBox_NUM];					  //Na1 as positive input, minimum weight searched in forward rounds (estimated/determined)
extern bool Na1FWMinWOver[RNUM][NA1_NUM][SBox_NUM];				  //Tags if the round has been searched: 0 false 1 true
extern int Na1FWLB[RNUM];										  //The smallest weight corresponding to the number of rounds is used to determine whether the overall subset needs to be searched or not
extern int Na1FWOutLB[RNUM][NA1_NUM][SBox_NUM];					  //Minimum weight for the corresponding number of rounds, [0] minimum weight [1] if or not to the end [2] index of the end -> use directly for indexing purposes
extern int Na1FWOutLBInfo[RNUM][NA1_NUM][SBox_NUM][2];			  //[0]:Whether to search to the end [1]:Corresponding index
extern int Na1RoundNPBWMinW[NA1_NUM][SBox_NUM];					  //minw
extern int Na1RoundNPBWASNandG[NA1_NUM][SBox_NUM][2];			  //Record ASN as well as group
extern u8 Na1RoundNPBWARRInfo[NA1_NUM][SBox_NUM][3][ARR_LEN/2];   //Record asn_index, asn_input and the corresponding group.
extern int Na1BWMinW[RNUM][NA1_NUM][SBox_NUM];		
extern bool Na1BWMinWOver[RNUM][NA1_NUM][SBox_NUM];  
extern int Na1BWLB[RNUM];               

//NA==2 
extern int Na2RoundNPInput[NA2_NUM][3];  //[0]input/output [1]input/output of the first active sbox  [2]input/output of the second active sbox 
extern int Na2RoundNPInfo[NA2_NUM][2];   //[0]fwMinw, [1]bwMinw 
extern int Na2InputIndex[SBox_SIZE][SBox_SIZE]; 
extern int Na2InOutLink[NA2_NUM][NA2_NUM];         
extern double Na2OutWeightOrder[NA2_NUM][NA2_NUM];    
extern int Na2SBoxIndex[NA2_SBoxNUM][2];                   //[0]index of the first active sbox  [1]index of the second active sbox  
extern int Na2SBoxInputIndex[SBox_NUM][SBox_NUM];
extern int Na2RoundNPFWMinW[NA2_NUM][NA2_SBoxNUM];		   //minw
extern int Na2RoundNPFWASNandG[NA2_NUM][NA2_SBoxNUM][2];	
extern u8 Na2RoundNPFWARRInfo[NA2_NUM][NA2_SBoxNUM][3][ARR_LEN/2]; 
extern int*** Na2FWMinW;
extern bool   Na2FWMinWOver[RNUM][NA2_NUM][NA2_SBoxNUM]; 
extern int Na2FWLB[RNUM];							   
extern int*** Na2FWOutLB;
extern int**** Na2FWOutLBInfo;  
extern int Na2RoundNPBWMinW[NA2_NUM][NA2_SBoxNUM];    
extern int Na2RoundNPBWASNandG[NA2_NUM][NA2_SBoxNUM][2];
extern u8 Na2RoundNPBWARRInfo[NA2_NUM][NA2_SBoxNUM][3][ARR_LEN/2]; 
extern int*** Na2BWMinW; 
extern bool   Na2BWMinWOver[RNUM][NA2_NUM][NA2_SBoxNUM];  
extern int Na2BWLB[RNUM + 1];                              

extern map<bool*, pair<__m128i*, int*>> NaBestTrailMap;

extern int NaLB[RNUM + 1][3]; //Minimum weight estimates for different subsets corresponding to the number of rounds [0]Na1 [1]Na2 [2]Na3 -> this subset is estimated with the least active SBox estimate

extern int Rnum;
extern int Bn;
extern int BWBn,FWBn;
extern FILE* fp;

typedef struct STATE {
	int rnum;			//record round
	u8 j;				//record sbox
	int W;				//record weight of 1~rnum-1
	int w;				//record weight of rnum

	u8 sbx_a[ARR_LEN];  // [0] record active sbox  
	u8 sbx_in[ARR_LEN]; // [1] record input
	u8 sbx_g[ARR_LEN]; 
	int sbx_num;        //record: num of active sbx - 1
	int g_num;

	int nr_minw;

	STATE() : rnum(1), j(0), w(0), W(0), sbx_num(0), g_num(0), nr_minw(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_g, 0, ARR_LEN);
	};
	STATE(int data_r, int data_W) : rnum(data_r), j(0), w(0), W(data_W), sbx_num(0), g_num(0), nr_minw(0) {
		memset(sbx_a, 0, ARR_LEN);
		memset(sbx_in, 0, ARR_LEN);
		memset(sbx_g, 0, ARR_LEN);
	};

	~STATE() {};
}STATE;

#endif // !_GLOBLEVARIABLES_H_




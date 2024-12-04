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
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) //对齐

#define Block_SIZE   64
#define SBox_BITSIZE 4
#define SBox_SIZE    0x10
#define SBox_NUM     (Block_SIZE/SBox_BITSIZE)
#define State_NUM    1
#define ARR_LEN      16 // SBox_NUM/2
#define STATE_LEN    (sizeof(__m128i)*State_NUM)
#define NA1_NUM      15  
#define NA2_NUM		 225 //15*15
#define Na12UBRnum 10
#define Na3UBRnum  10

//#define TYPE 0 //Diff 10 5 很快
#define TYPE 1   //Linear 10 10
#define RNUM 25

#define Round1_ASN 5

extern int BestB[RNUM + 1];
extern double weight[4];

//如果Na3可以往下搜，可以使用这个变量，从第一轮开始搜索，每轮ASN>=3
extern u8 Round1MinIndex[SBox_SIZE];
extern double Round1MinW[SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) Round1MinSPTable[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) Round1MinSPTableXor[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM

//正向
extern double FWWeightMinandMax[SBox_SIZE][2];     //输入对应的最小重量和最大
extern u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; //输入对应的输出index
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) FWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

//逆向
extern double BWWeightMinandMax[SBox_SIZE][2]; //输入对应的最小重量和最大
extern u8 BWWeightOrderIndex[SBox_SIZE][SBox_SIZE];
extern double BWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) BWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) BWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

extern ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE][State_NUM];
extern ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE][State_NUM];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];

// NA1和NA2的Input（output）按最小重量和最大重量和hm从小排序到大
// 搜索是先向后搜素，再向前搜索
// 
//NA==1 需要的变量
extern int Na1RoundNPInput[NA1_NUM][5]; //[0]input/output[准确的值，无变换], [1]fwMinw, [2]fwMaxw, [3]bwMinw, [4]bwMaxw 以fw优先，后续变量的index均以这个为准
extern int Na1InputIndex[SBox_SIZE];      //真实值索引
extern int Na1InOutLink[NA1_NUM][NA1_NUM];         //与输入兼容的输出index
extern double Na1OutWeightOrder[NA1_NUM][NA1_NUM]; //重量：升序排列
//正向搜索的变量 不包括当前轮数
extern int Na1RoundNPFWInfo[NA1_NUM][2];               //记录两个：ASN，minw 
extern u8 Na1RoundNPFWARRInfo[NA1_NUM][2][ARR_LEN];   //把asn_index以及asn_input记录
extern int Na1FWMinW[RNUM][NA1_NUM][2];	 //Na1作为正向输入，正向第几轮搜索到的最小重量，[0]最小重量（预估/确定） [1]标记当前轮是否搜索到底:0 false 1 true
extern int Na1FWLB[RNUM];                //对应轮数最小的重量，用来判断整体的子集是否需要搜索，只记录重量即可
extern int Na1FWOutLB[RNUM][NA1_NUM][3];    //对应轮数对应输出最小重量,[0]最小重量 [1]是否到底 [2] 到底的index -> 为了索引直接用
//逆向搜索 包括当前轮数
extern int Na1RoundNPBWInfo[NA1_NUM][2];               //记录两个：ASN，minw
extern u8 Na1RoundNPBWARRInfo[NA1_NUM][2][ARR_LEN];   //把asn_index以及asn_input记录
extern int Na1BWMinW[RNUM][NA1_NUM][2]; //Na1作为逆向输入，逆向第几轮ASN>=2搜索到的最小重量，[0]最小重量（预估/确定）[1]标记当前轮是否搜索到底:0 false 1 true   
extern int Na1BWLB[RNUM];               //对应轮数最小的重量，用来判断整体的子集是否需要搜索，只记录重量即可

//NA==2 需要的变量
//[0]input/output [1]fwminw [2]fwmaxw [3]bwminw [4]bwmaxw [5]第一个SBox的input/output [6]第二个SBox的input/output 以fw优先，后续变量的index以这个为准
extern int Na2RoundNPInput[NA2_NUM][7];
extern int Na2InputIndex[SBox_SIZE][SBox_SIZE]; //真实值索引
extern int Na2InOutLink[NA2_NUM][NA2_NUM];         //与输入兼容的输出index
extern double Na2OutWeightOrder[NA2_NUM][NA2_NUM]; //重量：升序排列
extern ALIGNED_TYPE_(__m128i, 16) Na2FWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //Output经过线性变换后的状态，由于循环移位不变，因此有SBox_NUM/2种可能
extern ALIGNED_TYPE_(__m128i, 16) Na2BWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //逆向，输入的逆线性变换
//正向搜索 不包括当前轮
extern int Na2RoundNPFWInfo[NA2_NUM][SBox_NUM / 2][2];			   //记录两个：ASN，minw
extern u8 Na2RoundNPFWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //把asn_index以及asn_input记录
extern int Na2FWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2作为正向输入，正向第几轮搜索到的最小重量，[0]最小重量（预估/确定） [1]标记当前轮是否搜索到底:0 false 1 true
extern int Na2FWLB[RNUM];							   //对应轮数最小的重量，用来判断整体的子集是否需要搜索，Na2输入
extern int Na2FWOutLB[RNUM][NA2_NUM][SBox_NUM / 2][3];    //对应轮数对应输出最小重量,[0]最小重量 [1]是否到底 [2] 到底的index -> 为了索引直接用
//逆向搜索 包括当前轮
extern int Na2RoundNPBWInfo[NA2_NUM][SBox_NUM / 2][2];             //记录两个：ASN，minw //ASN>=
extern u8 Na2RoundNPBWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //把asn_index以及asn_input记录
extern int Na2BWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2作为逆向输入，逆向第几轮ASN>=2搜索到的最小重量，[0]最小重量（预估/确定）[1]标记当前轮是否搜索到底:0 false 1 true
extern int Na2BWLB[RNUM];                              //对应轮数最小的重量，用来判断整体的子集是否需要搜索

extern map<int*, pair<__m128i**, int*>> NaBestTrailMap;

extern int NaLB[RNUM + 1][3]; //不同子集对应轮数的最小重量估计 [0]Na1 [1]Na2 [2]Na3->该子集估计就是用最少活跃SBox估计

extern int Rnum;
extern int Bn;
extern int BWBn, FWBn;
extern FILE* fp;

typedef struct STATE {
	int rnum;    //record round
	u8 j;       //record sbox
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


//结点用来存储活跃SBox
typedef struct node { //index：active sbox 
	struct node* lc, * rc;
	int index;
	node() : lc(NULL), rc(NULL), index(0) {};
	node(int data) :lc(NULL), rc(NULL), index(data) {};
	~node() {};
}Node, * Tree;

extern Node* T;

#endif // !_GLOBLEVARIABLES_H_




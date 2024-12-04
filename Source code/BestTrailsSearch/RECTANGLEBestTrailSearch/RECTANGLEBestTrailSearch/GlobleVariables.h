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
#define ALIGNED_TYPE_(t,x) t ALIGNED_(x) //����

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

//#define TYPE 0 //Diff 10 5 �ܿ�
#define TYPE 1   //Linear 10 10
#define RNUM 25

#define Round1_ASN 5

extern int BestB[RNUM + 1];
extern double weight[4];

//���Na3���������ѣ�����ʹ������������ӵ�һ�ֿ�ʼ������ÿ��ASN>=3
extern u8 Round1MinIndex[SBox_SIZE];
extern double Round1MinW[SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) Round1MinSPTable[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) Round1MinSPTableXor[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM

//����
extern double FWWeightMinandMax[SBox_SIZE][2];     //�����Ӧ����С���������
extern u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; //�����Ӧ�����index
extern double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) FWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

//����
extern double BWWeightMinandMax[SBox_SIZE][2]; //�����Ӧ����С���������
extern u8 BWWeightOrderIndex[SBox_SIZE][SBox_SIZE];
extern double BWWeightOrderW[SBox_SIZE][SBox_SIZE];
extern ALIGNED_TYPE_(__m128i, 16) BWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
extern ALIGNED_TYPE_(__m128i, 16) BWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

extern ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE][State_NUM];
extern ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE][State_NUM];

extern double DDTorLAT[SBox_SIZE][SBox_SIZE];

// NA1��NA2��Input��output������С���������������hm��С���򵽴�
// ��������������أ�����ǰ����
// 
//NA==1 ��Ҫ�ı���
extern int Na1RoundNPInput[NA1_NUM][5]; //[0]input/output[׼ȷ��ֵ���ޱ任], [1]fwMinw, [2]fwMaxw, [3]bwMinw, [4]bwMaxw ��fw���ȣ�����������index�������Ϊ׼
extern int Na1InputIndex[SBox_SIZE];      //��ʵֵ����
extern int Na1InOutLink[NA1_NUM][NA1_NUM];         //��������ݵ����index
extern double Na1OutWeightOrder[NA1_NUM][NA1_NUM]; //��������������
//���������ı��� ��������ǰ����
extern int Na1RoundNPFWInfo[NA1_NUM][2];               //��¼������ASN��minw 
extern u8 Na1RoundNPFWARRInfo[NA1_NUM][2][ARR_LEN];   //��asn_index�Լ�asn_input��¼
extern int Na1FWMinW[RNUM][NA1_NUM][2];	 //Na1��Ϊ�������룬����ڼ�������������С������[0]��С������Ԥ��/ȷ���� [1]��ǵ�ǰ���Ƿ���������:0 false 1 true
extern int Na1FWLB[RNUM];                //��Ӧ������С�������������ж�������Ӽ��Ƿ���Ҫ������ֻ��¼��������
extern int Na1FWOutLB[RNUM][NA1_NUM][3];    //��Ӧ������Ӧ�����С����,[0]��С���� [1]�Ƿ񵽵� [2] ���׵�index -> Ϊ������ֱ����
//�������� ������ǰ����
extern int Na1RoundNPBWInfo[NA1_NUM][2];               //��¼������ASN��minw
extern u8 Na1RoundNPBWARRInfo[NA1_NUM][2][ARR_LEN];   //��asn_index�Լ�asn_input��¼
extern int Na1BWMinW[RNUM][NA1_NUM][2]; //Na1��Ϊ�������룬����ڼ���ASN>=2����������С������[0]��С������Ԥ��/ȷ����[1]��ǵ�ǰ���Ƿ���������:0 false 1 true   
extern int Na1BWLB[RNUM];               //��Ӧ������С�������������ж�������Ӽ��Ƿ���Ҫ������ֻ��¼��������

//NA==2 ��Ҫ�ı���
//[0]input/output [1]fwminw [2]fwmaxw [3]bwminw [4]bwmaxw [5]��һ��SBox��input/output [6]�ڶ���SBox��input/output ��fw���ȣ�����������index�����Ϊ׼
extern int Na2RoundNPInput[NA2_NUM][7];
extern int Na2InputIndex[SBox_SIZE][SBox_SIZE]; //��ʵֵ����
extern int Na2InOutLink[NA2_NUM][NA2_NUM];         //��������ݵ����index
extern double Na2OutWeightOrder[NA2_NUM][NA2_NUM]; //��������������
extern ALIGNED_TYPE_(__m128i, 16) Na2FWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //Output�������Ա任���״̬������ѭ����λ���䣬�����SBox_NUM/2�ֿ���
extern ALIGNED_TYPE_(__m128i, 16) Na2BWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //��������������Ա任
//�������� ��������ǰ��
extern int Na2RoundNPFWInfo[NA2_NUM][SBox_NUM / 2][2];			   //��¼������ASN��minw
extern u8 Na2RoundNPFWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //��asn_index�Լ�asn_input��¼
extern int Na2FWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2��Ϊ�������룬����ڼ�������������С������[0]��С������Ԥ��/ȷ���� [1]��ǵ�ǰ���Ƿ���������:0 false 1 true
extern int Na2FWLB[RNUM];							   //��Ӧ������С�������������ж�������Ӽ��Ƿ���Ҫ������Na2����
extern int Na2FWOutLB[RNUM][NA2_NUM][SBox_NUM / 2][3];    //��Ӧ������Ӧ�����С����,[0]��С���� [1]�Ƿ񵽵� [2] ���׵�index -> Ϊ������ֱ����
//�������� ������ǰ��
extern int Na2RoundNPBWInfo[NA2_NUM][SBox_NUM / 2][2];             //��¼������ASN��minw //ASN>=
extern u8 Na2RoundNPBWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //��asn_index�Լ�asn_input��¼
extern int Na2BWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2��Ϊ�������룬����ڼ���ASN>=2����������С������[0]��С������Ԥ��/ȷ����[1]��ǵ�ǰ���Ƿ���������:0 false 1 true
extern int Na2BWLB[RNUM];                              //��Ӧ������С�������������ж�������Ӽ��Ƿ���Ҫ����

extern map<int*, pair<__m128i**, int*>> NaBestTrailMap;

extern int NaLB[RNUM + 1][3]; //��ͬ�Ӽ���Ӧ��������С�������� [0]Na1 [1]Na2 [2]Na3->���Ӽ����ƾ��������ٻ�ԾSBox����

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


//��������洢��ԾSBox
typedef struct node { //index��active sbox 
	struct node* lc, * rc;
	int index;
	node() : lc(NULL), rc(NULL), index(0) {};
	node(int data) :lc(NULL), rc(NULL), index(data) {};
	~node() {};
}Node, * Tree;

extern Node* T;

#endif // !_GLOBLEVARIABLES_H_




#include<iostream>
#include<vector>
#include<bitset>
#include<algorithm>
#include<smmintrin.h>
#include<iomanip>
#include "GlobleVariables.h"
#include "GenTable.h"

#if(TYPE == 0)
double weight[4] = { 0,2,3,INFINITY };
#endif

#if(TYPE == 1)
double weight[4] = { 0,1,2,INFINITY };
#endif

int SR_offsets[4] = { 0,1,12,13 };

u8 Round1MinIndex[SBox_SIZE]; //输出对应最小重量的输入index
double Round1MinW[SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) Round1MinSPTable[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM
ALIGNED_TYPE_(__m128i, 16) Round1MinSPTableXor[SBox_NUM][SBox_SIZE][State_NUM]; //sbox*output*state_NUM

//正向
double FWWeightMinandMax[SBox_SIZE][2];     //输入对应的最小重量和最大
u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; //输入对应的输出index
double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
ALIGNED_TYPE_(__m128i, 16) FWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

//逆向
double BWWeightMinandMax[SBox_SIZE][2]; //输入对应的最小重量和最大
u8 BWWeightOrderIndex[SBox_SIZE][SBox_SIZE];
double BWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) BWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM
ALIGNED_TYPE_(__m128i, 16) BWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

// NA1和NA2的Input（output）按最小重量和最大重量和hm从小排序到大
// 搜索是先向后搜素，再向前搜索
// 
//NA==1 需要的变量
int Na1RoundNPInput[NA1_NUM][5]; //[0]input/output[准确的值，无变换], [1]fwMinw, [2]fwMaxw, [3]bwMinw, [4]bwMaxw 以fw优先，后续变量的index均以这个为准
int Na1InputIndex[SBox_SIZE]; //真实值索引
int Na1InOutLink[NA1_NUM][NA1_NUM];         //与输入兼容的输出index
double Na1OutWeightOrder[NA1_NUM][NA1_NUM]; //重量：升序排列
//正向搜索的变量 不包括当前轮数
int Na1RoundNPFWInfo[NA1_NUM][2];               //记录两个：ASN，minw 
u8 Na1RoundNPFWARRInfo[NA1_NUM][2][ARR_LEN];   //把asn_index以及asn_input记录
int Na1FWMinW[RNUM][NA1_NUM][2];	 //Na1作为正向输入，正向第几轮搜索到的最小重量，[0]最小重量（预估/确定） [1]标记当前轮是否搜索到底:0 false 1 true
int Na1FWLB[RNUM];                //对应轮数最小的重量，用来判断整体的子集是否需要搜索，只记录重量即可
int Na1FWOutLB[RNUM][NA1_NUM][3];    //对应轮数对应输出最小重量,[0]最小重量 [1]是否到底 [2] 到底的index -> 为了索引直接用
//逆向搜索 包括当前轮数
int Na1RoundNPBWInfo[NA1_NUM][2];               //记录两个：ASN，minw
u8 Na1RoundNPBWARRInfo[NA1_NUM][2][ARR_LEN];   //把asn_index以及asn_input记录
int Na1BWMinW[RNUM][NA1_NUM][2]; //Na1作为逆向输入，逆向第几轮ASN>=2搜索到的最小重量，[0]最小重量（预估/确定）[1]标记当前轮是否搜索到底:0 false 1 true
int Na1BWLB[RNUM];               //对应轮数最小的重量，用来判断整体的子集是否需要搜索，只记录重量即可

//NA==2 需要的变量
//[0]input/output [1]fwminw [2]fwmaxw [3]bwminw [4]bwmaxw [5]第一个SBox的input/output [6]第二个SBox的input/output 以fw优先，后续变量的index以这个为准
int Na2RoundNPInput[NA2_NUM][7];
int Na2InputIndex[SBox_SIZE][SBox_SIZE]; //真实值索引
int Na2InOutLink[NA2_NUM][NA2_NUM];         //与输入兼容的输出index
double Na2OutWeightOrder[NA2_NUM][NA2_NUM]; //重量：升序排列
ALIGNED_TYPE_(__m128i, 16) Na2FWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //Output经过线性变换后的状态，由于循环移位不变，因此有SBox_NUM/2种可能
ALIGNED_TYPE_(__m128i, 16) Na2BWOutput[NA2_NUM][SBox_NUM / 2][State_NUM]; //逆向，输入的逆线性变换
//正向搜索 不包括当前轮
int Na2RoundNPFWInfo[NA2_NUM][SBox_NUM / 2][2];			   //记录两个：ASN，minw
u8 Na2RoundNPFWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //把asn_index以及asn_input记录
int Na2FWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2作为正向输入，正向第几轮搜索到的最小重量，[0]最小重量（预估/确定） [1]标记当前轮是否搜索到底:0 false 1 true
int Na2FWLB[RNUM];							   //对应轮数最小的重量，用来判断整体的子集是否需要搜索，Na2输入
int Na2FWOutLB[RNUM][NA2_NUM][SBox_NUM / 2][3];    //对应轮数对应输出最小重量,[0]最小重量 [1]是否到底 [2] 到底的index -> 为了索引直接用
//逆向搜索 包括当前轮
int Na2RoundNPBWInfo[NA2_NUM][SBox_NUM / 2][2];             //记录两个：ASN，minw //ASN>=
u8 Na2RoundNPBWARRInfo[NA2_NUM][SBox_NUM / 2][2][ARR_LEN]; //把asn_index以及asn_input记录
int Na2BWMinW[RNUM][NA2_NUM][SBox_NUM / 2][2];  //Na2作为逆向输入，逆向第几轮ASN>=2搜索到的最小重量，[0]最小重量（预估/确定）[1]标记当前轮是否搜索到底:0 false 1 true
int Na2BWLB[RNUM];                              //对应轮数最小的重量，用来判断整体的子集是否需要搜索

int NaLB[RNUM + 1][3]; //不同子集对应轮数的最小重量估计 [0]Na1 [1]Na2 [2]Na3

ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE][State_NUM];
ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE][State_NUM];
ALIGNED_TYPE_(__m128i, 16) tmp_r1SP[SBox_NUM][SBox_SIZE][State_NUM];

int hamming[16] = { 0x0,0x1,0x2,0x4,0x8,0x3,0x5,0x6,0x9,0xa,0xc,0x7,0xb,0xd,0xe,0xf }; //根据hamming排序的值
double DDTorLAT[SBox_SIZE][SBox_SIZE];

void WeightOrderTables() { // weightorder_Table
	int Sbox[SBox_SIZE] = { 0x6,0x5,0xc,0xa,0x1,0xe,0x7,0x9,0xb,0x0,0x3,0xd,0x8,0xf,0x4,0x2 };

	memset(DDTorLAT, 0, sizeof(DDTorLAT));

#if(TYPE == 0)
	//DDT
	for (int del_in = 0; del_in < SBox_SIZE; del_in++) {
		for (int i = 0; i < SBox_SIZE; i++) {
			DDTorLAT[del_in][Sbox[i] ^ Sbox[i ^ del_in]] ++;
		}
	}

	for (int i = 0; i < SBox_SIZE; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			DDTorLAT[i][j] = -log(DDTorLAT[i][j] / SBox_SIZE) / log(2);
		}
	}

#else
	//LAT
	bitset<SBox_BITSIZE> mask_in, mask_out;
	for (int i = 0; i < SBox_SIZE; i++) {
		mask_in = i;
		for (int j = 0; j < SBox_SIZE; j++) {
			mask_out = j;
			for (int k = 0; k < SBox_SIZE; k++) {
				if (((bitset<SBox_BITSIZE>(k) & mask_in).count() & 0x1) == ((bitset<SBox_BITSIZE>(Sbox[k]) & mask_out).count() & 0x1))
					DDTorLAT[i][j]++;
			}
		}
	}
	for (int i = 0; i < SBox_SIZE; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			DDTorLAT[i][j] = -log(abs(DDTorLAT[i][j] / (SBox_SIZE / 2) - 1)) / log(2);
		}
	}

#endif

	int index;
	u8 HM[SBox_SIZE][SBox_BITSIZE + 1]; //记录对应hm表的hm信息
	memset(HM, 0, sizeof(HM));

	for (int i = 0; i < SBox_SIZE; i++) {
		bitset<4> tmp_hm = hamming[i];
		HM[i][SBox_BITSIZE] = tmp_hm.count();
		index = 0;
		for (int j = 0; j < SBox_BITSIZE; j++) {
			if (tmp_hm[j] == 1) {
				HM[i][index] = j;
				index++;
			}
		}
	}

	//for round-1  out->in , 后续再reorder
	memset(Round1MinIndex, 0, sizeof(Round1MinIndex));
	memset(Round1MinW, 0, sizeof(Round1MinW));

	for (int i = 0; i < SBox_SIZE; i++) { //output
		Round1MinW[i] = 0xff;
		for (int j = 0; j < SBox_SIZE; j++) { //input
			if (DDTorLAT[hamming[j]][hamming[i]] < Round1MinW[i]) {
				Round1MinW[i] = DDTorLAT[hamming[j]][hamming[i]];
				Round1MinIndex[i] = hamming[j];
			}
		}
	}

	//for round-i  in->out 
	memset(FWWeightOrderIndex, 0, sizeof(FWWeightOrderIndex));
	memset(FWWeightOrderW, 0, sizeof(FWWeightOrderW));
	memset(FWWeightMinandMax, 0, sizeof(FWWeightMinandMax));
	for (int i = 0; i < SBox_SIZE; i++) {
		index = 0;
		for (int w = 0; w < 4; w++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				if (DDTorLAT[i][hamming[j]] == weight[w]) {
					FWWeightOrderIndex[i][index] = hamming[j];
					FWWeightOrderW[i][index] = weight[w];
					index++;
				}
			}
		}
	}
	bool tag;
	for (int i = 0; i < SBox_SIZE; i++) {
		FWWeightMinandMax[i][0] = FWWeightOrderW[i][0]; //记录输入对应的最小重量
		tag = true;
		for (int j = SBox_SIZE - 1; j >= 0; j--) {
			if (tag && FWWeightOrderW[i][j] != INFINITY) {
				FWWeightMinandMax[i][1] = FWWeightOrderW[i][j];
				tag = false;
			}
			FWWeightOrderW[i][j] -= FWWeightOrderW[i][0];
		}
	}

	//for round_i out->in
	memset(BWWeightOrderIndex, 0, sizeof(BWWeightOrderIndex));
	memset(BWWeightOrderW, 0, sizeof(BWWeightOrderW));
	memset(BWWeightMinandMax, 0, sizeof(BWWeightMinandMax));
	for (int i = 0; i < SBox_SIZE; i++) {
		index = 0;
		for (int w = 0; w < 4; w++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				if (DDTorLAT[hamming[j]][i] == weight[w]) {
					BWWeightOrderIndex[i][index] = hamming[j];
					BWWeightOrderW[i][index] = weight[w];
					index++;
				}
			}
		}
	}

	for (int i = 0; i < SBox_SIZE; i++) {
		BWWeightMinandMax[i][0] = BWWeightOrderW[i][0]; //记录输入对应的最小重量
		tag = true;
		for (int j = SBox_SIZE - 1; j >= 0; j--) {
			if (tag && BWWeightOrderW[i][j] != INFINITY) {
				BWWeightMinandMax[i][1] = BWWeightOrderW[i][j];
				tag = false;
			}
			BWWeightOrderW[i][j] -= BWWeightOrderW[i][0];
		}
	}
}

void GenPandSearchOrder(int Per[], int INVPer[]) { // Per[block_size]; INVPer[block_size]; //逆向
	// 生成P  2d->1d
	int tmp_P1[SBox_BITSIZE][SBox_NUM];
	int index = 0;
	for (int j = 0; j < SBox_NUM; j++) {
		for (int i = 0; i < SBox_BITSIZE; i++) {
			tmp_P1[i][j] = index;
			index++;
		}
	}
	int tmp_P2[SBox_BITSIZE][SBox_NUM]; //置换后
	for (int i = 0; i < SBox_BITSIZE; i++) {
		for (int j = 0; j < SBox_NUM - SR_offsets[i]; j++) tmp_P2[i][j] = tmp_P1[i][j + SR_offsets[i]];
		for (int j = SBox_NUM - SR_offsets[i]; j < SBox_NUM; j++) tmp_P2[i][j] = tmp_P1[i][j - SBox_NUM + SR_offsets[i]];
	}
	index = 0;
	for (int j = 0; j < SBox_NUM; j++) {
		for (int i = 0; i < SBox_BITSIZE; i++) {
			Per[index] = tmp_P2[i][j];
			index++;
		}
	}

	// 生成INVP  2d->1d
	for (int i = 0; i < SBox_BITSIZE; i++) {
		for (int j = 0; j < SBox_NUM - SR_offsets[i]; j++) tmp_P2[i][j + SR_offsets[i]] = tmp_P1[i][j];
		for (int j = SBox_NUM - SR_offsets[i]; j < SBox_NUM; j++) tmp_P2[i][j - SBox_NUM + SR_offsets[i]] = tmp_P1[i][j];
	}
	index = 0;
	for (int j = 0; j < SBox_NUM; j++) {
		for (int i = 0; i < SBox_BITSIZE; i++) {
			INVPer[index] = tmp_P2[i][j];
			index++;
		}
	}
}

void GenSPTable() {
	WeightOrderTables();
	int P[Block_SIZE], INVP[Block_SIZE];
	GenPandSearchOrder(P, INVP);
	bitset<SBox_BITSIZE>  temp1;
	bitset<Block_SIZE>    temp2, mask = 0xf;

	//PTable
	memset(PTable, 0, sizeof(PTable));
	for (int i = 0; i < SBox_NUM; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			temp1 = j; //input  0 ~ f
			temp2 = 0;
			for (int k = 0; k < SBox_BITSIZE; k++)
				temp2[P[i * SBox_BITSIZE + k]] = temp1[k];
			for (int k = 0; k < 16; k++)
				PTable[i][j][k >> 4].m128i_u8[k & 0xf] = ((temp2 >> (k * 4)) & mask).to_ulong();
		}
	}
	//INVPTable
	memset(INVPTable, 0, sizeof(INVPTable));
	for (int i = 0; i < SBox_NUM; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			temp1 = j; //input  0 ~ f
			temp2 = 0;
			for (int k = 0; k < SBox_BITSIZE; k++)
				temp2[INVP[i * SBox_BITSIZE + k]] = temp1[k];
			for (int k = 0; k < 16; k++)
				INVPTable[i][j][k >> 4].m128i_u8[k & 0xf] = ((temp2 >> (k * 4)) & mask).to_ulong();
		}
	}

	//creat_SPTable:
	memset(FWSPTable, 0, sizeof(FWSPTable));
	memset(BWSPTable, 0, sizeof(BWSPTable));
	memset(Round1MinSPTable, 0, sizeof(Round1MinSPTable));
	memset(FWSPTableXor, 0, sizeof(FWSPTableXor));
	memset(BWSPTableXor, 0, sizeof(BWSPTableXor));
	memset(Round1MinSPTableXor, 0, sizeof(Round1MinSPTableXor));
	memset(tmp_r1SP, 0, sizeof(tmp_r1SP));
	
	int index;

	//最后统一根据in_index xor得到输出，故直接给出即可	
	for (int i = 0; i < SBox_NUM; i++) { //sbox
		for (int j = 0; j < SBox_SIZE; j++) { //Input
			//FWSPTable:
			for (int k = 0; k < SBox_SIZE; k++) { //Output
				memcpy(FWSPTable[i][j][k], PTable[i][(FWWeightOrderIndex[j][k])], sizeof(__m128i) * State_NUM);
				if (k) {
					for (int s = 0; s < State_NUM; s++) {
						FWSPTableXor[i][j][k][s] = _mm_xor_si128(FWSPTable[i][j][k][s], FWSPTable[i][j][k - 1][s]);
					}
				}
				else {
					memcpy(FWSPTableXor[i][j][k], FWSPTable[i][j][k], sizeof(__m128i) * State_NUM);
				}
				if (FWWeightOrderW[j][k + 1] == INFINITY) break;
			}
			//BWSPTable:
			for (int k = 0; k < SBox_SIZE; k++) { //Output
				memcpy(BWSPTable[i][j][k], INVPTable[i][(BWWeightOrderIndex[j][k])], sizeof(__m128i) * State_NUM);
				if (k) {
					for (int s = 0; s < State_NUM; s++) {
						BWSPTableXor[i][j][k][s] = _mm_xor_si128(BWSPTable[i][j][k][s], BWSPTable[i][j][k - 1][s]);
					}
				}
				else {
					memcpy(BWSPTableXor[i][j][k], BWSPTable[i][j][k], sizeof(__m128i) * State_NUM);
				}
				if (BWWeightOrderW[j][k + 1] == INFINITY) break;
			}
			memcpy(tmp_r1SP[i][j], PTable[i][hamming[j]], sizeof(__m128i) * State_NUM);
		}
	}
	
	//reorder min1_weightorder 
	u8 temp_index[SBox_SIZE];
	u8 temp_index_hamming[SBox_SIZE][SBox_BITSIZE + 1];
	double temp_weight[SBox_SIZE];
	memcpy(temp_index, Round1MinIndex, sizeof(temp_index));
	memcpy(temp_weight, Round1MinW, sizeof(temp_weight));

	index = 0;
	for (int k = 0; k < 4; k++) { // weight_len
		for (int i = 0; i < SBox_SIZE; i++) { //del_out
			if (temp_weight[i] == weight[k]) {
				Round1MinIndex[index] = temp_index[i];
				Round1MinW[index] = temp_weight[i] - weight[1];
				for (int j = 0; j < SBox_NUM; j++) {
					memcpy(Round1MinSPTable[j][index], tmp_r1SP[j][i], sizeof(__m128i) * State_NUM);
				}
				index++;
			}
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		memcpy(Round1MinSPTableXor[i][0], Round1MinSPTable[i][0], sizeof(__m128i) * State_NUM);
		for (int j = SBox_SIZE - 1; j > 0; j--) {
			for (int k = 0; k < State_NUM; k++) {
				Round1MinSPTableXor[i][j][k] = _mm_xor_si128(Round1MinSPTable[i][j - 1][k], Round1MinSPTable[i][j][k]);
			}
		}		
	}
}

void RecordInfoFW(__m128i output[], int FWInfo[], u8 FWARRInfo[][ARR_LEN], int asnlb) {
	FWInfo[0] = 0; //asn
	FWInfo[1] = 0; //min_w
	for (int i = 0; i < SBox_NUM; i++) {
		if (output[0].m128i_u8[i]) {
			FWARRInfo[0][FWInfo[0]] = i;
			FWARRInfo[1][FWInfo[0]] = output[0].m128i_u8[i];
			FWInfo[1] += FWWeightMinandMax[FWARRInfo[1][FWInfo[0]]][0];
			FWInfo[0]++;
		}
	}
	if (FWInfo[0] < asnlb) FWInfo[1] = 0xffff;
}

void RecordInfoBW(__m128i output[], int BWInfo[], u8 BWARRInfo[][ARR_LEN], int asnlb) {
	BWInfo[0] = 0; //asn
	BWInfo[1] = 0; //min_w
	for (int i = 0; i < SBox_NUM; i++) {
		if ((output[0].m128i_u8[i]) & 0xf) {
			BWARRInfo[0][BWInfo[0]] = i;
			BWARRInfo[1][BWInfo[0]] = (output[0].m128i_u8[i]);
			BWInfo[1] += BWWeightMinandMax[BWARRInfo[1][BWInfo[0]]][0];
			BWInfo[0]++;
		}
	}
	if (BWInfo[0] < asnlb) BWInfo[1] = 0xffff;
}


void GenRoundNpInfo() {	
	//大子集的重量下界
	memset(NaLB, 0, sizeof(NaLB));
	NaLB[1][0] = weight[1]; 	NaLB[1][1] = 2 * weight[1];	NaLB[1][2] = 3 * weight[1];
	NaLB[2][0] = 2 * weight[1];	NaLB[2][1] = 4 * weight[1];	NaLB[2][2] = 6 * weight[1];
	int index;

	memset(Na1RoundNPInput, 0, sizeof(Na1RoundNPInput));
	index = 0;
	for (int tmp_min = 1; tmp_min < 3; tmp_min++) {
		for (int tmp_max = tmp_min; tmp_max < 3; tmp_max++) {
			for (int i = 1; i < SBox_SIZE; i++) {
				if (FWWeightMinandMax[hamming[i]][0] == weight[tmp_min] && FWWeightMinandMax[hamming[i]][1] == weight[tmp_max]) {
					Na1RoundNPInput[index][0] = hamming[i];
					Na1RoundNPInput[index][1] = weight[tmp_min];
					Na1RoundNPInput[index][2] = weight[tmp_max];
					Na1RoundNPInput[index][3] = BWWeightMinandMax[hamming[i]][0];
					Na1RoundNPInput[index][4] = BWWeightMinandMax[hamming[i]][1];
					index++;
				}
			}
		}
	}
	memset(Na1InputIndex, 0, sizeof(Na1InputIndex));
	for (int i = 0; i < NA1_NUM; i++) {
		Na1InputIndex[Na1RoundNPInput[i][0]] = i;
	}


	memset(Na1InOutLink, 0, sizeof(Na1InOutLink));
	memset(Na1OutWeightOrder, 0, sizeof(Na1OutWeightOrder));

	for (int i = 0; i < NA1_NUM; i++) {
		index = 0;
		for (int w = 1; w < 4; w++) {
			for (int j = 0; j < NA1_NUM; j++) {
				if (DDTorLAT[Na1RoundNPInput[i][0]][Na1RoundNPInput[j][0]] == weight[w]) {
					Na1InOutLink[i][index] = j;
					Na1OutWeightOrder[i][index] = weight[w];
					index++;
				}
			}
		}
	}

	memset(Na1RoundNPFWInfo, 0, sizeof(Na1RoundNPFWInfo));	
	memset(Na1RoundNPBWInfo, 0, sizeof(Na1RoundNPBWInfo));
	memset(Na1RoundNPFWARRInfo, 0, sizeof(Na1RoundNPFWARRInfo));
	memset(Na1RoundNPBWARRInfo, 0, sizeof(Na1RoundNPBWARRInfo));	
	
	memset(Na1FWMinW, 0, sizeof(Na1FWMinW)); //给定第一轮输出
	memset(Na1BWMinW, 0, sizeof(Na1BWMinW)); //给定第一轮输入

	memset(Na1FWLB, 0, sizeof(Na1FWLB));
	memset(Na1BWLB, 0, sizeof(Na1BWLB));

	bool tag = false;
	for (int i = 0; i < NA1_NUM; i++) {
		RecordInfoFW(PTable[0][Na1RoundNPInput[i][0]], Na1RoundNPFWInfo[i], Na1RoundNPFWARRInfo[i], 1);     //作为NP轮的输出经过置换后统计的数据
		RecordInfoBW(INVPTable[0][Na1RoundNPInput[i][0]], Na1RoundNPBWInfo[i], Na1RoundNPBWARRInfo[i], 2);  //作为NP轮的输入经过逆置换后统计的数据

		Na1FWMinW[1][i][0] = Na1RoundNPFWInfo[i][1];
		Na1FWMinW[1][i][1] = 1;

		Na1BWMinW[1][i][0] = Na1RoundNPBWInfo[i][1];
		Na1BWMinW[1][i][1] = 1;


		if (i == 0) {
			Na1FWLB[1] = Na1FWMinW[1][i][0];
			Na1BWLB[1] = Na1BWMinW[1][i][0];
		}
		else {
			if (Na1FWMinW[1][i][0] < Na1FWLB[1]) Na1FWLB[1] = Na1FWMinW[1][i][0];
			if (Na1BWMinW[1][i][0] < Na1BWLB[1]) Na1BWLB[1] = Na1BWMinW[1][i][0];
		}
	}

	memset(Na1FWOutLB, 0, sizeof(Na1FWOutLB)); //记录输出兼容输入中的最小逆向重量，为了更准确，应该也包括起始轮多余的重量

	for (int i = 0; i < NA1_NUM; i++) {
		Na1FWOutLB[1][i][0] = Na1FWMinW[1][Na1InOutLink[i][0]][0] + Na1OutWeightOrder[i][0];
		Na1FWOutLB[1][i][1] = 1;
		Na1FWOutLB[1][i][2] = 0;
		for (int j = 1; j < NA1_NUM; j++) {
			if (Na1OutWeightOrder[i][j] == INFINITY) break;
			if (Na1FWMinW[1][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j] < Na1FWOutLB[1][i][0]) {
				Na1FWOutLB[1][i][0] = Na1FWMinW[1][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j];
				Na1FWOutLB[1][i][2] = j;
			}						
		}
	}


	//NA2
	//Na2 Input:
	int Na2MinW = 2 * weight[1], Na2MaxW = 2 * weight[2];
	int Na2hamming[NA2_NUM] = { 0x11, 0x12, 0x14, 0x18, 0x21, 0x22, 0x24, 0x28, 0x41, 0x42,
		0x44, 0x48, 0x81, 0x82, 0x84, 0x88, 0x13, 0x15, 0x16, 0x19, 0x1a, 0x1c, 0x23, 0x25,
		0x26, 0x29, 0x2a, 0x2c, 0x31, 0x32, 0x34, 0x38, 0x43, 0x45, 0x46, 0x49, 0x4a, 0x4c,
		0x51, 0x52, 0x54, 0x58, 0x61, 0x62, 0x64, 0x68, 0x83, 0x85, 0x86, 0x89, 0x8a, 0x8c,
		0x91, 0x92, 0x94, 0x98, 0xa1, 0xa2, 0xa4, 0xa8, 0xc1, 0xc2, 0xc4, 0xc8, 0x17, 0x1b,
		0x1d, 0x1e, 0x27, 0x2b, 0x2d, 0x2e, 0x33, 0x35, 0x36, 0x39, 0x3a, 0x3c, 0x47, 0x4b,
		0x4d, 0x4e, 0x53, 0x55, 0x56, 0x59, 0x5a, 0x5c, 0x63, 0x65, 0x66, 0x69, 0x6a, 0x6c,
		0x71, 0x72, 0x74, 0x78, 0x87, 0x8b, 0x8d, 0x8e, 0x93, 0x95, 0x96, 0x99, 0x9a, 0x9c,
		0xa3, 0xa5, 0xa6, 0xa9, 0xaa, 0xac, 0xb1, 0xb2, 0xb4, 0xb8, 0xc3, 0xc5, 0xc6, 0xc9,
		0xca, 0xcc, 0xd1, 0xd2, 0xd4, 0xd8, 0xe1, 0xe2, 0xe4, 0xe8, 0x1f, 0x2f, 0x37, 0x3b,
		0x3d, 0x3e, 0x4f, 0x57, 0x5b, 0x5d, 0x5e, 0x67, 0x6b, 0x6d, 0x6e, 0x73, 0x75, 0x76,
		0x79, 0x7a, 0x7c, 0x8f, 0x97, 0x9b, 0x9d, 0x9e, 0xa7, 0xab, 0xad, 0xae, 0xb3, 0xb5,
		0xb6, 0xb9, 0xba, 0xbc, 0xc7, 0xcb, 0xcd, 0xce, 0xd3, 0xd5, 0xd6, 0xd9, 0xda, 0xdc,
		0xe3, 0xe5, 0xe6, 0xe9, 0xea, 0xec, 0xf1, 0xf2, 0xf4, 0xf8, 0x3f, 0x5f, 0x6f, 0x77,
		0x7b, 0x7d, 0x7e, 0x9f, 0xaf, 0xb7, 0xbb, 0xbd, 0xbe, 0xcf, 0xd7, 0xdb, 0xdd, 0xde,
		0xe7, 0xeb, 0xed, 0xee, 0xf3, 0xf5, 0xf6, 0xf9, 0xfa, 0xfc, 0x7f, 0xbf, 0xdf, 0xef,
		0xf7, 0xfb, 0xfd, 0xfe, 0xff}; //8bithamming从小到大排序（不包括前4-bit/后4-bit为零的值）

	memset(Na2RoundNPInput, 0, sizeof(Na2RoundNPInput));
	index = 0;
	for (int tmp_min = Na2MinW; tmp_min <= Na2MaxW; tmp_min++) {
		for (int tmp_max = tmp_min; tmp_max <= Na2MaxW; tmp_max++) {
			for (int i = 0; i < NA2_NUM; i++) {
				if (FWWeightMinandMax[Na2hamming[i] & 0xf][0] + FWWeightMinandMax[(Na2hamming[i] >> 4)][0] == tmp_min
					&& FWWeightMinandMax[Na2hamming[i] & 0xf][1] + FWWeightMinandMax[(Na2hamming[i] >> 4)][1] == tmp_max) {
					Na2RoundNPInput[index][0] = Na2hamming[i];
					Na2RoundNPInput[index][1] = tmp_min;
					Na2RoundNPInput[index][2] = tmp_max;
					Na2RoundNPInput[index][3] = BWWeightMinandMax[Na2hamming[i] & 0xf][0] + BWWeightMinandMax[(Na2hamming[i] >> 4)][0];
					Na2RoundNPInput[index][4] = BWWeightMinandMax[Na2hamming[i] & 0xf][1] + BWWeightMinandMax[(Na2hamming[i] >> 4)][1];
					Na2RoundNPInput[index][5] = Na2hamming[i] & 0xf;
					Na2RoundNPInput[index][6] = Na2hamming[i] >> 4;
					index++;
				}
			}
		}
	}
	memset(Na2InputIndex, 0, sizeof(Na2InputIndex));
	for (int i = 0; i < NA2_NUM; i++) {
		Na2InputIndex[Na2RoundNPInput[i][5]][Na2RoundNPInput[i][6]] = i;
	}

	memset(Na2InOutLink, 0, sizeof(Na2InOutLink));
	memset(Na2OutWeightOrder, 0, sizeof(Na2OutWeightOrder));

	for (int i = 0; i < NA2_NUM; i++) {
		index = 0;
		for (int w = Na2MinW; w <= Na2MaxW; w++) {
			for (int j = 0; j < NA2_NUM; j++) {
				if (DDTorLAT[Na2RoundNPInput[i][5]][Na2RoundNPInput[j][5]] + DDTorLAT[Na2RoundNPInput[i][6]][Na2RoundNPInput[j][6]] == w) {
					Na2InOutLink[i][index] = j;
					Na2OutWeightOrder[i][index] = w;
					index++;
				}
			}
		}
		if (index != NA2_NUM) {
			for (index; index < NA2_NUM; index++) Na2OutWeightOrder[i][index] = INFINITY; //后续全置为INF
		}
	}



	memset(Na2FWOutput, 0, sizeof(Na2FWOutput));
	memset(Na2RoundNPFWInfo, 0, sizeof(Na2RoundNPFWInfo));
	memset(Na2RoundNPFWARRInfo, 0, sizeof(Na2RoundNPFWARRInfo));

	memset(Na2BWOutput, 0, sizeof(Na2BWOutput));
	memset(Na2RoundNPBWInfo, 0, sizeof(Na2RoundNPBWInfo));
	memset(Na2RoundNPBWARRInfo, 0, sizeof(Na2RoundNPBWARRInfo));

	memset(Na2FWMinW, 0, sizeof(Na2FWMinW));
	memset(Na2BWMinW, 0, sizeof(Na2BWMinW));

	memset(Na2FWLB, 0, sizeof(Na2FWLB));
	memset(Na2BWLB, 0, sizeof(Na2BWLB));

	tag = true;
	for (int i = 0; i < NA2_NUM; i++) {
		for (int s = 0; s < (SBox_NUM / 2); s++) {
			for (int k = 0; k < State_NUM; k++) {
				Na2FWOutput[i][s][k] = _mm_xor_si128(   PTable[0][Na2RoundNPInput[i][5]][k],    PTable[s + 1][Na2RoundNPInput[i][6]][k]);
				Na2BWOutput[i][s][k] = _mm_xor_si128(INVPTable[0][Na2RoundNPInput[i][5]][k], INVPTable[s + 1][Na2RoundNPInput[i][6]][k]);
			}
				
			RecordInfoFW(Na2FWOutput[i][s], Na2RoundNPFWInfo[i][s], Na2RoundNPFWARRInfo[i][s], 2);
			RecordInfoBW(Na2BWOutput[i][s], Na2RoundNPBWInfo[i][s], Na2RoundNPBWARRInfo[i][s], 3);

			Na2FWMinW[1][i][s][0] = Na2RoundNPFWInfo[i][s][1];
			Na2FWMinW[1][i][s][1] = 1;

			Na2BWMinW[1][i][s][0] = Na2RoundNPBWInfo[i][s][1];
			Na2BWMinW[1][i][s][1] = 1;
			
			if (tag) { //Na2大子集开始，也需要限制正向的ASN
				Na2FWLB[1] = Na2FWMinW[1][i][s][0];
				Na2BWLB[1] = Na2BWMinW[1][i][s][0];
				tag = false;
			}
			else {
				if (Na2FWMinW[1][i][s][0] < Na2FWLB[1]) Na2FWLB[1] = Na2FWMinW[1][i][s][0];
				if (Na2BWMinW[1][i][s][0] < Na2BWLB[1]) Na2BWLB[1] = Na2BWMinW[1][i][s][0];
			}
		}
	}
	
	memset(Na2FWOutLB, 0, sizeof(Na2FWOutLB));
	for (int i = 0; i < NA2_NUM; i++) { //关心总体的，后续再具体问题具体分析，不再关心不同SBoxIndex下的重量，为了更准确，应该也包括起始轮多余的重量
		for (int s = 0; s < (SBox_NUM / 2); s++) {
			tag = true;
			for (int j = 0; j < NA2_NUM; j++) {
				if (Na2OutWeightOrder[i][j] == INFINITY) break;
				if (tag) {
					Na2FWOutLB[1][i][s][0] = Na2FWMinW[1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]; //包含NP轮的兼容重量减去最小重量
					Na2FWOutLB[1][i][s][1] = 1;
					Na2FWOutLB[1][i][s][2] = j;
					tag = false;
				}
				else if ((Na2FWMinW[1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[1][i][s][0]) {
					Na2FWOutLB[1][i][s][0] = Na2FWMinW[1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j];
					Na2FWOutLB[1][i][s][2] = j;
				}	

			}
		}
	}

}

void GenTables() {
	GenSPTable();
	GenRoundNpInfo();
}

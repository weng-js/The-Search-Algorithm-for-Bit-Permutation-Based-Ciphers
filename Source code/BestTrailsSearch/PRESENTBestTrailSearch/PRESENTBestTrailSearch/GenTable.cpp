#include<iostream>
#include<vector>
#include<bitset>
#include<algorithm>
#include<smmintrin.h>
#include<iomanip>
#include "GlobleVariables.h"
#include "GenTable.h"

#if(TYPE)
double weight[4] = { 0,1,2,INFINITY };
#else
double weight[4] = { 0,2,3,INFINITY }; 
#endif
int WeightLen;

u8 Round1MinIndex[SBox_SIZE];
int Round1MinW[SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) Round1MinSPTable[SBox_NUM][SBox_SIZE];

//forward
int FWWeightMinandMax[SBox_SIZE][2];    
u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; 

//backward
int BWWeightMinandMax[SBox_SIZE][2]; 
u8 BWWeightOrderIndex[SBox_SIZE][SBox_SIZE];
double BWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) BWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; 


int INVSbox_loc[SBox_NUM][2];                   
int Sbox_loc[SBox_NUM];                          
int FWGroup_SBox[Group_NUM][4];
int BWGroup_SBox[Group_NUM][4];
ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE];

double DDTorLAT[SBox_SIZE][SBox_SIZE];

//NA==1 
int Na1RoundNPInput[NA1_NUM];           
int Na1RoundNPInfo[NA1_NUM][2];			
int Na1InputIndex[SBox_SIZE];           
int Na1InOutLink[NA1_NUM][NA1_NUM];     
double Na1OutWeightOrder[NA1_NUM][NA1_NUM]; 
int Na1RoundNPFWMinW[NA1_NUM][SBox_NUM];   
int Na1RoundNPFWASNandG[NA1_NUM][SBox_NUM][2];
u8 Na1RoundNPFWARRInfo[NA1_NUM][SBox_NUM][3][ARR_LEN/2]; 
int Na1FWMinW[RNUM][NA1_NUM][SBox_NUM];	     
bool Na1FWMinWOver[RNUM][NA1_NUM][SBox_NUM];	
int Na1FWLB[RNUM];                   
int Na1FWOutLB[RNUM][NA1_NUM][SBox_NUM];       
int    Na1FWOutLBInfo[RNUM][NA1_NUM][SBox_NUM][2];  
int Na1RoundNPBWMinW[NA1_NUM][SBox_NUM];    
int Na1RoundNPBWASNandG[NA1_NUM][SBox_NUM][2];
u8 Na1RoundNPBWARRInfo[NA1_NUM][SBox_NUM][3][ARR_LEN/2];   
int Na1BWMinW[RNUM][NA1_NUM][SBox_NUM];     
bool Na1BWMinWOver[RNUM][NA1_NUM][SBox_NUM];
int Na1BWLB[RNUM];                

//NA==2 
int Na2RoundNPInput[NA2_NUM][3];  
int Na2RoundNPInfo[NA2_NUM][2]; 
int Na2InputIndex[SBox_SIZE][SBox_SIZE]; 
int Na2InOutLink[NA2_NUM][NA2_NUM];         
double Na2OutWeightOrder[NA2_NUM][NA2_NUM]; 
int Na2SBoxIndex[NA2_SBoxNUM][2]; 
int Na2SBoxInputIndex[SBox_NUM][SBox_NUM];
int Na2RoundNPFWMinW[NA2_NUM][NA2_SBoxNUM];		   
int Na2RoundNPFWASNandG[NA2_NUM][NA2_SBoxNUM][2];	 
u8 Na2RoundNPFWARRInfo[NA2_NUM][NA2_SBoxNUM][3][ARR_LEN/2]; 
int*** Na2FWMinW;     
bool Na2FWMinWOver[RNUM][NA2_NUM][NA2_SBoxNUM]; 
int Na2FWLB[RNUM];							
int*** Na2FWOutLB;  
int**** Na2FWOutLBInfo;  
int Na2RoundNPBWMinW[NA2_NUM][NA2_SBoxNUM];   
int Na2RoundNPBWASNandG[NA2_NUM][NA2_SBoxNUM][2];
u8 Na2RoundNPBWARRInfo[NA2_NUM][NA2_SBoxNUM][3][ARR_LEN/2]; 
int*** Na2BWMinW; 
bool   Na2BWMinWOver[RNUM][NA2_NUM][NA2_SBoxNUM]; 
int Na2BWLB[RNUM + 1];

int NaLB[RNUM + 1][3]; 

int hamming[16] = { 0x0,0x1,0x2,0x4,0x8,0x3,0x5,0x6,0x9,0xa,0xc,0x7,0xb,0xd,0xe,0xf }; 

void WeightOrderTables() { // weightorder_Table
	int Sbox[SBox_SIZE] = { 0xc,0x5,0x6,0xb,0x9,0x0,0xa,0xd,0x3,0xe,0xf,0x8,0x4,0x7,0x1,0x2 };

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
	WeightLen = 4;

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
	WeightLen = 4;
#endif
	
	int index;
	//for round-1  out->in , 
	memset(Round1MinIndex, 0, sizeof(Round1MinIndex));
	memset(Round1MinW, 0, sizeof(Round1MinW));

	for (int i = 0; i < SBox_SIZE; i++) { //output
		Round1MinW[i] = 0xffffff;
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
		for (int w = 0; w < WeightLen; w++) {
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
		FWWeightMinandMax[i][0] = FWWeightOrderW[i][0];
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
		for (int w = 0; w < WeightLen; w++) {
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
		BWWeightMinandMax[i][0] = BWWeightOrderW[i][0]; 
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

void GenPandSearchOrder(int Per[], int INVPer[]) { 
	// Per[block_size]; INVPer[block_size];
	for (int i = 0; i < Block_SIZE; i++) {
		INVPer[Per[i]] = i;
	}

	memset(INVSbox_loc, 0, sizeof(INVSbox_loc));
	int index = 0;

	for (int g = 0; g < (SBox_NUM / 4); g++) {
		for (int i = 0; i < 4; i++) {
			INVSbox_loc[index][0] = g;
			INVSbox_loc[index][1] = (i << 2) | g; 
			index++;
		}
	}

	memset(Sbox_loc, 0, sizeof(Sbox_loc));
	for (int i = 0; i < SBox_NUM; i++) {
		Sbox_loc[i] = i >> 2;
	}

	int tmp_INVSBoxIndex[SBox_NUM] = { 0 };
	for (int i = 0; i < SBox_NUM; i++) tmp_INVSBoxIndex[INVSbox_loc[i][1]] = i;

	for (int g = 0; g < Group_NUM; g++) {
		for (int i = 0; i < 4; i++) {
			FWGroup_SBox[g][i] = (i << 2) | g;
			BWGroup_SBox[g][i] = (g << 2) | i;
		}
	}
}

void GenSPTable() {
	WeightOrderTables();
	int P[Block_SIZE] = { 0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
				  4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
				  8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
				  12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63 };
	int  INVP[Block_SIZE];
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
			for (int k = 0; k < SBox_NUM; k++)
				PTable[i][j].m128i_u8[k] = ((temp2 >> (k * 4)) & mask).to_ulong();
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
			for (int k = 0; k < SBox_NUM; k++)
				INVPTable[i][j].m128i_u8[k] = ((temp2 >> (k * 4)) & mask).to_ulong();
		}
	}

	//creat_SPTable:
	memset(FWSPTable, 0, sizeof(FWSPTable));
	memset(BWSPTable, 0, sizeof(BWSPTable));
	memset(Round1MinSPTable, 0, sizeof(Round1MinSPTable));

	ALIGNED_TYPE_(__m128i, 16) tmp_r1SP[SBox_NUM][SBox_SIZE];
	memset(tmp_r1SP, 0, sizeof(tmp_r1SP));
	
	int index;

	for (int i = 0; i < SBox_NUM; i++) { //sbox
		for (int j = 0; j < SBox_SIZE; j++) { //Input
			//FWSPTable:
			FWSPTable[i][j][0] = PTable[i][FWWeightOrderIndex[j][0]];
			for (int k = 1; k < SBox_SIZE; k++) { //Output
				if (FWWeightOrderW[j][k] == INFINITY) break;
				FWSPTable[i][j][k] = _mm_xor_si128(PTable[i][FWWeightOrderIndex[j][k - 1]], PTable[i][FWWeightOrderIndex[j][k]]);
			}
			//BWSPTable:
			BWSPTable[i][j][0] = INVPTable[INVSbox_loc[i][1]][BWWeightOrderIndex[j][0]];
			for (int k = 1; k < SBox_SIZE; k++) { //Output
				if (BWWeightOrderW[j][k] == INFINITY) break;
				BWSPTable[i][j][k] = _mm_xor_si128(INVPTable[INVSbox_loc[i][1]][BWWeightOrderIndex[j][k - 1]], INVPTable[INVSbox_loc[i][1]][BWWeightOrderIndex[j][k]]);
			}
			tmp_r1SP[i][j] = PTable[i][hamming[j]];
		}
	}
	
	//reorder min1_weightorder 
	u8 temp_index[SBox_SIZE];
	int temp_weight[SBox_SIZE];
	memcpy(temp_index, Round1MinIndex, sizeof(temp_index));
	memcpy(temp_weight, Round1MinW, sizeof(temp_weight));

	index = 0;
	for (int k = 0; k < WeightLen; k++) {	  //weight_len
		for (int i = 0; i < SBox_SIZE; i++) { //del_out
			if (temp_weight[i] == weight[k]) {
				Round1MinIndex[index] = temp_index[i];
				Round1MinW[index] = temp_weight[i] - weight[1];
				for (int j = 0; j < SBox_NUM; j++) {
					Round1MinSPTable[j][index] = tmp_r1SP[j][i];
				}
				index++;
			}
		}
	}

	for (int i = 0; i < SBox_NUM; i++) {
		for (int j = SBox_SIZE - 1; j > 0; j--) {
			Round1MinSPTable[i][j] = _mm_xor_si128(Round1MinSPTable[i][j - 1], Round1MinSPTable[i][j]);
		}
	}
}

void RecordInfoFW(__m128i output, int& FWMinW, int FWASNandG[], u8 FWARRInfo[][ARR_LEN / 2], int asnlb) {
	FWASNandG[0] = 0; //asn
	FWASNandG[1] = 1; //g_num
	FWMinW = 0;		  //min_w
	for (int i = 0; i < SBox_NUM; i++) {
		if (output.m128i_u8[i]) {
			FWARRInfo[0][FWASNandG[0]] = i;
			FWARRInfo[1][FWASNandG[0]] = output.m128i_u8[i];
			FWARRInfo[2][FWASNandG[0]] = Sbox_loc[i];
			FWMinW += FWWeightMinandMax[FWARRInfo[1][FWASNandG[0]]][0];
			if (FWASNandG[0] && (FWARRInfo[2][FWASNandG[0]] != FWARRInfo[2][FWASNandG[0] - 1])) FWASNandG[1]++;
			FWASNandG[0]++;
		}
	}
	if (FWASNandG[0] < asnlb) FWMinW = 0xffff;
}

void RecordInfoBW(__m128i output, int& BWMinW, int BWASNandG[], u8 BWARRInfo[][ARR_LEN / 2], int asnlb) {
	BWASNandG[0] = 0; //asn
	BWASNandG[1] = 1; //asn
	BWMinW = 0;		  //min_w
	for (int i = 0; i < SBox_NUM; i++) {
		if (output.m128i_u8[INVSbox_loc[i][1]]) {
			BWARRInfo[0][BWASNandG[0]] = i;
			BWARRInfo[1][BWASNandG[0]] = output.m128i_u8[INVSbox_loc[i][1]];
			BWARRInfo[2][BWASNandG[0]] = INVSbox_loc[i][0];
			BWMinW += BWWeightMinandMax[BWARRInfo[1][BWASNandG[0]]][0];
			if (BWASNandG[0] && (BWARRInfo[2][BWASNandG[0]] != BWARRInfo[2][BWASNandG[0] - 1])) BWASNandG[1]++;
			BWASNandG[0]++;
		}
	}
	if (BWASNandG[0] < asnlb) BWMinW = 0xffffff;
}


void GenRoundNpInfo() {	
	memset(NaLB, 0, sizeof(NaLB));
	NaLB[1][0] = 1 * weight[1]; NaLB[1][1] = 2 * weight[1];	NaLB[1][2] = 3 * weight[1];
	NaLB[2][0] = 2 * weight[1];	NaLB[2][1] = 4 * weight[1];	NaLB[2][2] = 6 * weight[1];
	int index;

	memset(Na1RoundNPInput, 0, sizeof(Na1RoundNPInput));
	memset(Na1RoundNPInfo, 0, sizeof(Na1RoundNPInfo));
	index = 0;

	for (int i = 1; i < SBox_SIZE; i++) {
		Na1RoundNPInput[index] = hamming[i];
		Na1RoundNPInfo[index][0] = FWWeightMinandMax[hamming[i]][0];
		Na1RoundNPInfo[index][1] = BWWeightMinandMax[hamming[i]][0];
		index++;
	}
	memset(Na1InputIndex, 0, sizeof(Na1InputIndex));
	for (int i = 0; i < NA1_NUM; i++) {
		Na1InputIndex[Na1RoundNPInput[i]] = i;
	}


	memset(Na1InOutLink, 0, sizeof(Na1InOutLink));
	memset(Na1OutWeightOrder, 0, sizeof(Na1OutWeightOrder));

	for (int i = 0; i < NA1_NUM; i++) {
		index = 0;
		for (int w = 1; w < WeightLen; w++) {
			for (int j = 0; j < NA1_NUM; j++) {
				if (DDTorLAT[Na1RoundNPInput[i]][Na1RoundNPInput[j]] == weight[w]) {
					Na1InOutLink[i][index] = j;
					Na1OutWeightOrder[i][index] = weight[w];
					index++;
				}
			}
		}
	}

	memset(Na1RoundNPFWMinW, 0, sizeof(Na1RoundNPFWMinW));	
	memset(Na1RoundNPBWMinW, 0, sizeof(Na1RoundNPBWMinW));	
	memset(Na1RoundNPFWASNandG, 0, sizeof(Na1RoundNPFWASNandG));
	memset(Na1RoundNPBWASNandG, 0, sizeof(Na1RoundNPBWASNandG));
	memset(Na1RoundNPFWARRInfo, 0, sizeof(Na1RoundNPFWARRInfo));
	memset(Na1RoundNPBWARRInfo, 0, sizeof(Na1RoundNPBWARRInfo));	
	
	memset(Na1FWMinW, 0, sizeof(Na1FWMinW)); 
	memset(Na1BWMinW, 0, sizeof(Na1BWMinW)); 

	memset(Na1FWMinWOver, 0, sizeof(Na1FWMinWOver)); 
	memset(Na1BWMinWOver, 0, sizeof(Na1BWMinWOver)); 

	memset(Na1FWLB, 0, sizeof(Na1FWLB));
	memset(Na1BWLB, 0, sizeof(Na1BWLB));


	bool tag = true;
	for (int s = 0; s < SBox_NUM; s++) {
		for (int i = 0; i < NA1_NUM; i++) {
			RecordInfoFW(   PTable[s][Na1RoundNPInput[i]], Na1RoundNPFWMinW[i][s], Na1RoundNPFWASNandG[i][s], Na1RoundNPFWARRInfo[i][s], 1);   
			RecordInfoBW(INVPTable[s][Na1RoundNPInput[i]], Na1RoundNPBWMinW[i][s], Na1RoundNPBWASNandG[i][s], Na1RoundNPBWARRInfo[i][s], 2); 
			
			Na1FWMinW[1][i][s] = Na1RoundNPFWMinW[i][s];
			Na1FWMinWOver[1][i][s] = true;

			Na1BWMinW[1][i][s] = Na1RoundNPBWMinW[i][s];
			Na1BWMinWOver[1][i][s] = true;

			if (tag) {
				Na1FWLB[1] = Na1FWMinW[1][i][s];
				Na1BWLB[1] = Na1BWMinW[1][i][s];
				tag = false;
			}
			else {
				if (Na1FWMinW[1][i][s] < Na1FWLB[1]) Na1FWLB[1] = Na1FWMinW[1][i][s];
				if (Na1BWMinW[1][i][s] < Na1BWLB[1]) Na1BWLB[1] = Na1BWMinW[1][i][s];
			}
		}
	}
	

	memset(Na1FWOutLB, 0, sizeof(Na1FWOutLB)); 
	memset(Na1FWOutLBInfo, 0, sizeof(Na1FWOutLBInfo)); 

	for (int s = 0; s < SBox_NUM; s++) {
		for (int i = 0; i < NA1_NUM; i++) {
			Na1FWOutLB[1][i][s] = Na1FWMinW[1][Na1InOutLink[i][0]][s] + Na1OutWeightOrder[i][0];
			Na1FWOutLBInfo[1][i][s][0] = 1;
			Na1FWOutLBInfo[1][i][s][1] = 0;
			for (int j = 1; j < NA1_NUM; j++) {
				if (Na1OutWeightOrder[i][j] == INFINITY) break;
				if (Na1FWMinW[1][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j] < Na1FWOutLB[1][i][s]) {
					Na1FWOutLB[1][i][s] = Na1FWMinW[1][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j];
					Na1FWOutLBInfo[1][i][s][1] = j;
				}
			}
		}
	}



	//NA2
	double Na2Weight[6] = { 0 }; //weightLen == 6 || 3
	int Na2WeightLen = 0;
	for (int i = WeightLen - 2; i > 0; i--)  Na2WeightLen += i;
	index = 0; 
	for (int i = 1; i < WeightLen - 1; i++) {
		for (int j = i; j < WeightLen - 1; j++) {
			if (index) {
				int k; for (k = index; k > 0; k--) if (Na2Weight[k - 1] > weight[i] + weight[j]) Na2Weight[k] = Na2Weight[k - 1]; else break;
				Na2Weight[k] = weight[i] + weight[j]; index++;
			}
			else {
				Na2Weight[index] = weight[i] + weight[j]; index++;
			}
		}
	}
	
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
		0xf7, 0xfb, 0xfd, 0xfe, 0xff}; //8bits hamming

	memset(Na2RoundNPInput, 0, sizeof(Na2RoundNPInput));
	index = 0;
	for (int i = 0; i < NA2_NUM; i++) {
		Na2RoundNPInput[index][0] = Na2hamming[i];
		Na2RoundNPInput[index][1] = Na2hamming[i] & 0xf;
		Na2RoundNPInput[index][2] = Na2hamming[i] >> 4;
		Na2RoundNPInfo[index][0] = FWWeightMinandMax[Na2hamming[i] & 0xf][0] + FWWeightMinandMax[(Na2hamming[i] >> 4)][0];
		Na2RoundNPInfo[index][1] = BWWeightMinandMax[Na2hamming[i] & 0xf][0] + BWWeightMinandMax[(Na2hamming[i] >> 4)][0];
		index++;
	}
	memset(Na2InputIndex, 0, sizeof(Na2InputIndex));
	for (int i = 0; i < NA2_NUM; i++) {
		Na2InputIndex[Na2RoundNPInput[i][1]][Na2RoundNPInput[i][2]] = i;
	}

	memset(Na2InOutLink, 0, sizeof(Na2InOutLink));
	memset(Na2OutWeightOrder, 0, sizeof(Na2OutWeightOrder));

	for (int i = 0; i < NA2_NUM; i++) {
		index = 0;
		for (int w = 0; w < Na2WeightLen; w++) {
			for (int j = 0; j < NA2_NUM; j++) {
				if (DDTorLAT[Na2RoundNPInput[i][1]][Na2RoundNPInput[j][1]] + DDTorLAT[Na2RoundNPInput[i][2]][Na2RoundNPInput[j][2]] == Na2Weight[w]) {
					Na2InOutLink[i][index] = j;
					Na2OutWeightOrder[i][index] = Na2Weight[w];
					index++;
				}
			}
		}
		if (index != NA2_NUM) {
			for (index; index < NA2_NUM; index++) Na2OutWeightOrder[i][index] = INFINITY; 
		}
	}


	memset(Na2SBoxIndex, 0, sizeof(Na2SBoxIndex));
	memset(Na2SBoxInputIndex, 0, sizeof(Na2SBoxInputIndex));
	index = 0;
	for (int i = 0; i < SBox_NUM - 1; i++) {
		for (int j = i + 1; j < SBox_NUM; j++) {
			Na2SBoxIndex[index][0] = i; Na2SBoxIndex[index][1] = j; index++;
		}
	}
	for (int i = 0; i < NA2_SBoxNUM; i++) {
		Na2SBoxInputIndex[Na2SBoxIndex[i][0]][Na2SBoxIndex[i][1]] = i;
	}

	memset(Na2RoundNPFWMinW, 0, sizeof(Na2RoundNPFWMinW));
	memset(Na2RoundNPFWASNandG, 0, sizeof(Na2RoundNPFWASNandG));
	memset(Na2RoundNPFWARRInfo, 0, sizeof(Na2RoundNPFWARRInfo));

	memset(Na2RoundNPBWMinW, 0, sizeof(Na2RoundNPBWMinW));
	memset(Na2RoundNPBWASNandG, 0, sizeof(Na2RoundNPBWASNandG));
	memset(Na2RoundNPBWARRInfo, 0, sizeof(Na2RoundNPBWARRInfo));

	Na2FWMinW = new int** [RNUM];
	Na2BWMinW = new int** [RNUM];
	for (int i = 0; i < RNUM; i++) {
		Na2FWMinW[i] = new int* [NA2_NUM];
		Na2BWMinW[i] = new int* [NA2_NUM];
		for (int j = 0; j < NA2_NUM; j++) {
			Na2FWMinW[i][j] = new int[NA2_SBoxNUM]();
			Na2BWMinW[i][j] = new int[NA2_SBoxNUM]();
		}
	}


	memset(Na2FWMinWOver, 0, sizeof(Na2FWMinWOver));
	memset(Na2BWMinWOver, 0, sizeof(Na2BWMinWOver));

	memset(Na2FWLB, 0, sizeof(Na2FWLB));
	memset(Na2BWLB, 0, sizeof(Na2BWLB));


	tag = true;
	__m128i Na2FWOutput, Na2BWOutput;

	for (int i = 0; i < NA2_NUM; i++) {
		for (int s = 0; s < NA2_SBoxNUM; s++) {
			Na2FWOutput = _mm_xor_si128(   PTable[Na2SBoxIndex[s][0]][Na2RoundNPInput[i][1]],    PTable[Na2SBoxIndex[s][1]][Na2RoundNPInput[i][2]]);
			Na2BWOutput = _mm_xor_si128(INVPTable[Na2SBoxIndex[s][0]][Na2RoundNPInput[i][1]], INVPTable[Na2SBoxIndex[s][1]][Na2RoundNPInput[i][2]]);

			RecordInfoFW(Na2FWOutput, Na2RoundNPFWMinW[i][s], Na2RoundNPFWASNandG[i][s], Na2RoundNPFWARRInfo[i][s], 2);
			RecordInfoBW(Na2BWOutput, Na2RoundNPBWMinW[i][s], Na2RoundNPBWASNandG[i][s], Na2RoundNPBWARRInfo[i][s], 3);

			Na2FWMinW[1][i][s] = Na2RoundNPFWMinW[i][s];
			Na2FWMinWOver[1][i][s] = true;

			Na2BWMinW[1][i][s] = Na2RoundNPBWMinW[i][s];
			Na2BWMinWOver[1][i][s] = true;
			
			if (tag) { 
				Na2FWLB[1] = Na2FWMinW[1][i][s];
				Na2BWLB[1] = Na2BWMinW[1][i][s];
				tag = false;
			}
			else {
				if (Na2FWMinW[1][i][s] < Na2FWLB[1]) Na2FWLB[1] = Na2FWMinW[1][i][s];
				if (Na2BWMinW[1][i][s] < Na2BWLB[1]) Na2BWLB[1] = Na2BWMinW[1][i][s];
			}			
		}
	}
	
	Na2FWOutLB = new int** [RNUM];
	Na2FWOutLBInfo = new int*** [RNUM];
	for (int i = 0; i < RNUM; i++) {
		Na2FWOutLB[i] = new int* [NA2_NUM];
		Na2FWOutLBInfo[i] = new int** [NA2_NUM];
		for (int j = 0; j < NA2_NUM; j++) {
			Na2FWOutLB[i][j] = new int[NA2_SBoxNUM]();
			Na2FWOutLBInfo[i][j] = new int* [NA2_SBoxNUM];
			for (int k = 0; k < NA2_SBoxNUM; k++) Na2FWOutLBInfo[i][j][k] = new int[2]();
		}
	}
	for (int i = 0; i < NA2_NUM; i++) { 
		for (int s = 0; s < NA2_SBoxNUM; s++) {
			tag = true;
			for (int j = 0; j < NA2_NUM; j++) {
				if (Na2OutWeightOrder[i][j] == INFINITY) break;
				if (tag) {
					Na2FWOutLB[1][i][s] = Na2FWMinW[1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]; 
					Na2FWOutLBInfo[1][i][s][0] = 1;
					Na2FWOutLBInfo[1][i][s][1] = j;
					tag = false;
				}
				else if ((Na2FWMinW[1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[1][i][s]) {
					Na2FWOutLB[1][i][s] = Na2FWMinW[1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j];
					Na2FWOutLBInfo[1][i][s][1] = j;
				}	
			}
		}
	}
}

void GenTables() {
	GenSPTable();
	GenRoundNpInfo();
}

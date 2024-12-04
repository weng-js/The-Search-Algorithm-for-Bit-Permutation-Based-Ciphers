#include<iostream>
#include<vector>
#include<bitset>
#include<algorithm>
#include<smmintrin.h>
#include<iomanip>
#include "GlobleVariables.h"
#include "GenTable.h"

#if(TYPE)
double weight[4] = { 0,2,4,INFINITY };
#else
double weight[4] = { 0,2,3,INFINITY };
#endif
int WeightLen;

int FWWeightMinandMax[SBox_SIZE][2];    
u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE]; // sbox*input*output

int Sbox_loc[SBox_NUM];                        
int FWGroup_SBox[Group_NUM][4];
ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE];
double DDTorLAT[SBox_SIZE][SBox_SIZE];
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
			DDTorLAT[i][j] = -log(abs(DDTorLAT[i][j] / (SBox_SIZE / 2) - 1)) / log(2) * 2;
		}
	}
	WeightLen = 4;
#endif
	int index;
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
}

void GenPandSearchOrder(int Per[], int INVPer[]) { 
	// Per[block_size]; INVPer[block_size];
	for (int i = 0; i < Block_SIZE; i++) {
		INVPer[Per[i]] = i;
	}

	memset(Sbox_loc, 0, sizeof(Sbox_loc));
	for (int i = 0; i < SBox_NUM; i++) {
		Sbox_loc[i] = i >> 2;
	}
	
	for (int g = 0; g < Group_NUM; g++) {
		for (int i = 0; i < 4; i++) {
			FWGroup_SBox[g][i] = (i << 2) | g;
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
	for (int i = 0; i < SBox_NUM; i++) { //sbox
		for (int j = 0; j < SBox_SIZE; j++) { //Input
			//FWSPTable:
			FWSPTable[i][j][0] = PTable[i][FWWeightOrderIndex[j][0]];
			for (int k = 1; k < SBox_SIZE; k++) { //Output
				FWSPTable[i][j][k] = _mm_xor_si128(PTable[i][FWWeightOrderIndex[j][k]], PTable[i][FWWeightOrderIndex[j][k - 1]]);
				if (FWWeightOrderW[j][k + 1] == INFINITY) break;
			}
		}
	}	
}

void GenTables() {
	GenSPTable();
}

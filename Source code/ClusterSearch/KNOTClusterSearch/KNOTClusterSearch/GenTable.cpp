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
double weight[4] = { 0,2,4,INFINITY };
#endif

#if(Block_SIZE==256)
int SR_offsets[4] = { 0,1,8,25 };
#elif(Block_SIZE==384)
int SR_offsets[4] = { 0,1,8,55 };
#else
int SR_offsets[4] = { 0,1,16,25 };
#endif

double FWWeightMinandMax[SBox_SIZE][2];     
u8 FWWeightOrderIndex[SBox_SIZE][SBox_SIZE]; 
double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
ALIGNED_TYPE_(__m128i, 16) FWSPTable[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM];	// sbox*input*output*state_NUM
ALIGNED_TYPE_(__m128i, 16) FWSPTableXor[SBox_NUM][SBox_SIZE][SBox_SIZE][State_NUM]; // sbox*input*output*state_NUM

//SBox
int Sbox_loc[SBox_NUM][3];

ALIGNED_TYPE_(__m128i, 16) PTable[SBox_NUM][SBox_SIZE][State_NUM];
ALIGNED_TYPE_(__m128i, 16) INVPTable[SBox_NUM][SBox_SIZE][State_NUM];

int hamming[16] = { 0x0,0x1,0x2,0x4,0x8,0x3,0x5,0x6,0x9,0xa,0xc,0x7,0xb,0xd,0xe,0xf }; 
double DDTorLAT[SBox_SIZE][SBox_SIZE];

void WeightOrderTables() { 
	int Sbox[SBox_SIZE] = { 0x4,0x0,0xa,0x7,0xb,0xe,0x1,0xd,0x9,0xf,0x6,0x8,0x5,0x2,0xc,0x3 };
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
#elif(TYPE == 1)
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
#endif

	int index;
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
	// 2d->1d
	int tmp_P1[SBox_BITSIZE][SBox_NUM];
	int index = 0;
	for (int j = 0; j < SBox_NUM; j++) {
		for (int i = 0; i < SBox_BITSIZE; i++) {
			tmp_P1[i][j] = index;
			index++;
		}
	}
	int tmp_P2[SBox_BITSIZE][SBox_NUM];
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
	//Gen INVP  2d->1d
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

	memset(Sbox_loc, 0, sizeof(Sbox_loc));
	for (int i = 0; i < SBox_NUM; i++) {
		Sbox_loc[i][0] = i >> 5;
		Sbox_loc[i][1] = (i >> 1) & 0xf;
		Sbox_loc[i][2] = (i & 0x1) * 4;
	}
   
}

void GenSPTable() {
	WeightOrderTables();
	int P[Block_SIZE], INVP[Block_SIZE];
	GenPandSearchOrder(P, INVP);
	bitset<SBox_BITSIZE>  temp1;
	bitset<Block_SIZE>    temp2, mask = 0xff;

	//PTable
	memset(PTable, 0, sizeof(PTable));
	for (int i = 0; i < SBox_NUM; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			temp1 = j; //input  0 ~ f
			temp2 = 0;
			for (int k = 0; k < SBox_BITSIZE; k++)
				temp2[P[i * SBox_BITSIZE + k]] = temp1[k];
			for (int k = 0; k < (SBox_NUM / 2); k++)
				PTable[i][j][k >> 4].m128i_u8[k & 0xf] = ((temp2 >> (k * 8)) & mask).to_ulong();
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
			for (int k = 0; k < (SBox_NUM / 2); k++)
				INVPTable[i][j][k >> 4].m128i_u8[k & 0xf] = ((temp2 >> (k * 8)) & mask).to_ulong();
		}
	}

	//creat SPTable:
	memset(FWSPTable, 0, sizeof(FWSPTable));
	memset(FWSPTableXor, 0, sizeof(FWSPTableXor));	
	__m128i judge = _mm_set1_epi8(0xff);

	for (int i = 0; i < SBox_NUM; i++) { //sbox
		for (int j = 0; j < SBox_SIZE; j++) { //Input
			//FWSPTable:
			memcpy(FWSPTable[i][j][0], PTable[i][(FWWeightOrderIndex[j][0])], sizeof(__m128i) * State_NUM);
			memcpy(FWSPTableXor[i][j][0], FWSPTable[i][j][0], sizeof(__m128i) * State_NUM);
			for (int k = 1; k < SBox_SIZE; k++) { //Output
				if (FWWeightOrderW[j][k] == INFINITY) break;
				memcpy(FWSPTable[i][j][k], PTable[i][(FWWeightOrderIndex[j][k])], sizeof(__m128i) * State_NUM);
				for (int s = 0; s < State_NUM; s++) {
					FWSPTableXor[i][j][k][s] = _mm_xor_si128(FWSPTable[i][j][k][s], FWSPTable[i][j][k - 1][s]);
				}
			}
		}
	}	
}


void GenTables() {
	GenSPTable();
}

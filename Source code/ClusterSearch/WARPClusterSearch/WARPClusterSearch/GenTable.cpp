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
int WeightLen = 4;

int SBoxPermutation[SBox_NUM];   
int INVSBoxPermutation[SBox_NUM]; 
int State1Permutation[SBox_NUM];	
int INVState1Permutation[SBox_NUM];
int State2Permutation[SBox_NUM];    //P1
int INVState2Permutation[SBox_NUM]; //P1

__m128i State1PermutationSSE;
__m128i INVState1PermutationSSE;
__m128i SBoxPermutationSSE;
__m128i INVSBoxPermutationSSE;


int FWWeightMinandMax[SBox_SIZE][2];         
u8 FWWeightOrderValue[SBox_SIZE][SBox_SIZE];
double FWWeightOrderW[SBox_SIZE][SBox_SIZE];
double FWWeightOrderWMinusMin[SBox_SIZE][SBox_SIZE];

double DDTorLAT[SBox_SIZE][SBox_SIZE];
double DDTorLATMinusMin[SBox_SIZE][SBox_SIZE];

void WeightOrderTables() { 
	// weightorder_Table
	int NibblePermutation[SBox_NUM * 2] = { 31, 6, 29, 14, 1, 12, 21, 8, 27, 2, 3, 0, 25,4, 23, 10, 15, 22, 13, 30, 17, 28, 5, 24, 11, 18, 19, 16, 9, 20, 7, 26 };
	int Permutation1[SBox_NUM] = { 0 };
	memset(SBoxPermutation, 0, sizeof(SBoxPermutation));
	memset(State1Permutation, 0, sizeof(State1Permutation));
	memset(State2Permutation, 0, sizeof(State2Permutation));
	for (int i = 0; i < SBox_NUM; i++) {
		Permutation1[i] = (NibblePermutation[i << 1] / 2);
		State2Permutation[i] = (NibblePermutation[i << 1] / 2);
		SBoxPermutation[i] = (NibblePermutation[i << 1 | 1] / 2);
	}
	for (int i = 0; i < SBox_NUM; i++) {
		State1Permutation[i] = SBoxPermutation[Permutation1[i]];
	}
	for (int i = 0; i < SBox_NUM; i++) {
		INVSBoxPermutation[SBoxPermutation[i]] = i;
		INVState1Permutation[State1Permutation[i]] = i;
		INVState2Permutation[State2Permutation[i]] = i;
	}

	int Sbox[SBox_SIZE] = { 0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7, 0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6 };

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
	int tmp_Permutation[SBox_NUM] = {0};
	memcpy(tmp_Permutation, SBoxPermutation, sizeof(SBoxPermutation));
	memcpy(SBoxPermutation, INVSBoxPermutation, sizeof(INVSBoxPermutation));
	memcpy(INVSBoxPermutation, tmp_Permutation, sizeof(INVSBoxPermutation));
	memcpy(tmp_Permutation, State1Permutation, sizeof(State1Permutation));
	memcpy(State1Permutation, INVState1Permutation, sizeof(INVState1Permutation));
	memcpy(INVState1Permutation, tmp_Permutation, sizeof(INVState1Permutation));
	//LAT£º out -> in
	bitset<SBox_BITSIZE> mask_in, mask_out;
	for (int i = 0; i < SBox_SIZE; i++) {
		mask_out = i;
		for (int j = 0; j < SBox_SIZE; j++) {
			mask_in = j;
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
	State1PermutationSSE = _mm_setzero_si128();
	INVState1PermutationSSE = _mm_setzero_si128();
	SBoxPermutationSSE = _mm_setzero_si128();
	INVSBoxPermutationSSE = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		State1PermutationSSE.m128i_u8[i] = INVState1Permutation[i];
		INVState1PermutationSSE.m128i_u8[i] = State1Permutation[i];
		SBoxPermutationSSE.m128i_u8[i] = INVSBoxPermutation[i];
		INVSBoxPermutationSSE.m128i_u8[i] = SBoxPermutation[i];
	}

	int index; int k;
	//for round-i  in->out 
	memset(FWWeightOrderValue, 0, sizeof(FWWeightOrderValue));
	memset(FWWeightOrderW, 0, sizeof(FWWeightOrderW));
	memset(FWWeightOrderWMinusMin, 0, sizeof(FWWeightOrderWMinusMin));
	memset(FWWeightMinandMax, 0, sizeof(FWWeightMinandMax));
	for (int i = 0; i < SBox_SIZE; i++) {
		index = 0;
		for (int w = 0; w < WeightLen; w++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				if (DDTorLAT[i][j] == weight[w]) {
					FWWeightOrderValue[i][index] = j;
					FWWeightOrderW[i][index] = weight[w];
					FWWeightOrderWMinusMin[i][index] = weight[w];
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
			FWWeightOrderWMinusMin[i][j] -= FWWeightOrderW[i][0];
		}
	}
	memset(DDTorLATMinusMin, 0, sizeof(DDTorLATMinusMin));
	for (int i = 0; i < SBox_SIZE; i++) {
		for (int j = 0; j < SBox_SIZE; j++) {
			DDTorLATMinusMin[i][j] = DDTorLAT[i][j] - FWWeightMinandMax[i][0];
		}
	}
}


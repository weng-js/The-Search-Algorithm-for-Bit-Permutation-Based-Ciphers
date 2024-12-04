#include<iostream>
#include<vector>
#include<bitset>
#include<algorithm>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<iomanip>
#include "GlobleVariables.h"
#include "GenTable.h"

#if(TYPE)
double weight[4] = { 0,1,2,INFINITY };
#else
double weight[4] = { 0,2,3,INFINITY };
#endif
int WeightLen = 4;

int FWSBoxPermutation[SBox_NUM] = { 2,0,3,1,6,4,7,5 };    
int FWINVSBoxPermutation[SBox_NUM]; 

int BWSBoxPermutation[SBox_NUM];   

int FWSBoxROT[SBox_NUM];
int BWSBoxROT[SBox_NUM]; 

u8  IRFWMinV[SBox_NUM][SBox_SIZE];
int IRFWMinW[SBox_NUM][SBox_SIZE];


int FWWeightMinandMax[SBox_NUM][SBox_SIZE][2];        
u8	   FWWeightOrderV[SBox_NUM][SBox_SIZE][SBox_SIZE];
double FWWeightOrderW[SBox_NUM][SBox_SIZE][SBox_SIZE];
double FWWeightOrderWMinusMin[SBox_NUM][SBox_SIZE][SBox_SIZE];

double DDTorLAT[SBox_NUM][SBox_SIZE][SBox_SIZE];
double DDTorLATMinusMinW[SBox_NUM][SBox_SIZE][SBox_SIZE];

//DC Diff
int LB[RNUM + 1][2]; //NA = 0, NA = 1
//ASP
u8 ASP_FW_Info[ASP_NUM][ARR_LEN];
u8 ASP_BW_Info[ASP_NUM][ARR_LEN];
int ASPInfo[ASP_NUM]; 

//LB[D_r_0_i_ASP] = LBNA0[r,i,ASP] = max(BW[i-1,ASP]+FW[r-i,ASP],LBNA0[r,i,ASP])
int LBNA0[RNUM + 1][RNUM + 1][ASP_NUM];
int BWLB[RNUM][ASP_NUM];     
int FWLB[RNUM][ASP_NUM];

int ASNLB[RNUM + 1][2];
int ASNBWLB[RNUM][ASP_NUM];
int ASNFWLB[RNUM][ASP_NUM];
bool ASNBWLBOver[RNUM][ASP_NUM]; 
bool ASNFWLBOver[RNUM][ASP_NUM];

//int hamming[16] = { 0x0,0x1,0x2,0x4,0x8,0x3,0x5,0x6,0x9,0xa,0xc,0x7,0xb,0xd,0xe,0xf }; 

void WeightOrderTables() { 
	// weightorder_Table

	for (int i = 0; i < SBox_NUM; i++) {
		FWINVSBoxPermutation[FWSBoxPermutation[i]] = i;
		FWSBoxROT[i] = (i + 2) % SBox_NUM;
		BWSBoxROT[i] = (i - 2 + SBox_NUM) % SBox_NUM;
	}

	int Sbox[SBox_NUM][SBox_SIZE] = { 
		{14, 9, 15, 0, 13, 4, 10, 11, 1, 2, 8, 3, 7, 6, 12, 5},
		{4, 11, 14, 9, 15, 13, 0, 10, 7, 12, 5, 6, 2, 8, 1, 3},
		{1, 14, 7, 12, 15, 13, 0, 6, 11, 5, 9, 3, 2, 4, 8, 10},
		{7, 6, 8, 11, 0, 15, 3, 14, 9, 10, 12, 13, 5, 2, 4, 1},
		{14, 5, 15, 0, 7, 2, 12, 13, 1, 8, 4, 9, 11, 10, 6, 3},
		{2, 13, 11, 12, 15, 14, 0, 9, 7, 10, 6, 3, 1, 8, 4, 5},
		{11, 9, 4, 14, 0, 15, 10, 13, 6, 12, 5, 7, 3, 8, 1, 2},
		{13, 10, 15, 0, 14, 4, 9, 11, 2, 1, 8, 3, 7, 5, 12, 6},
	};

	memset(DDTorLAT, 0, sizeof(DDTorLAT));
	memset(DDTorLATMinusMinW, 0, sizeof(DDTorLATMinusMinW));

#if(TYPE == 0)
	for (int i = 0; i < SBox_NUM; i++) {
		BWSBoxPermutation[i] = (FWSBoxPermutation[i] - 2 + SBox_NUM) % SBox_NUM;
	}

	//DDT
	for (int sbx = 0; sbx < SBox_NUM; sbx++) {
		for (int del_in = 0; del_in < SBox_SIZE; del_in++) {
			for (int i = 0; i < SBox_SIZE; i++) {
				DDTorLAT[sbx][del_in][Sbox[sbx][i] ^ Sbox[sbx][i ^ del_in]] ++;
			}
		}

		for (int i = 0; i < SBox_SIZE; i++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				DDTorLAT[sbx][i][j] = -log(DDTorLAT[sbx][i][j] / SBox_SIZE) / log(2);
			}
		}
	}
#else
	int tmp_Sbox[SBox_NUM][SBox_SIZE] = { 0 };
	for (int i = 0; i < SBox_NUM; i++) {
		memcpy(tmp_Sbox[i], Sbox[i], sizeof(int) * SBox_SIZE);
	}
	for (int i = 0; i < SBox_NUM; i++) {
		memcpy(Sbox[i], tmp_Sbox[FWINVSBoxPermutation[i]], sizeof(int) * SBox_SIZE);
	}
	int tmp_Permutation[SBox_NUM] = { 0 };
	memcpy(tmp_Permutation, FWSBoxPermutation, sizeof(FWSBoxPermutation));
	memcpy(FWSBoxPermutation, FWINVSBoxPermutation, sizeof(FWINVSBoxPermutation));
	memcpy(FWINVSBoxPermutation, tmp_Permutation, sizeof(FWINVSBoxPermutation));

	for (int i = 0; i < SBox_NUM; i++) {
		BWSBoxPermutation[i] = (FWSBoxPermutation[i] + 2) % SBox_NUM;
		FWSBoxROT[i] = (i - 2 + SBox_NUM) % SBox_NUM; 
		BWSBoxROT[i] = (i + 2) % SBox_NUM;		     
	}

	//LAT£º out -> in
	for (int sbx = 0; sbx < SBox_NUM; sbx++) {
		bitset<SBox_BITSIZE> mask_in, mask_out;
		for (int i = 0; i < SBox_SIZE; i++) {
			mask_out = i;
			for (int j = 0; j < SBox_SIZE; j++) {
				mask_in = j;
				for (int k = 0; k < SBox_SIZE; k++) {
					if (((bitset<SBox_BITSIZE>(k) & mask_in).count() & 0x1) == ((bitset<SBox_BITSIZE>(Sbox[sbx][k]) & mask_out).count() & 0x1))
						DDTorLAT[sbx][i][j]++;
				}
			}
		}
		for (int i = 0; i < SBox_SIZE; i++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				DDTorLAT[sbx][i][j] = -log(abs(DDTorLAT[sbx][i][j] / (SBox_SIZE / 2) - 1)) / log(2);
			}
		}
	}
#endif	
	int index;
	//for round-i  in->out 
	memset(FWWeightOrderV, 0, sizeof(FWWeightOrderV));
	memset(FWWeightOrderW, 0, sizeof(FWWeightOrderW));
	memset(FWWeightOrderWMinusMin, 0, sizeof(FWWeightOrderWMinusMin));
	memset(FWWeightMinandMax, 0, sizeof(FWWeightMinandMax));

	for (int sbx = 0; sbx < SBox_NUM; sbx++) {
		for (int i = 0; i < SBox_SIZE; i++) {
			index = 0;
			for (int w = 0; w < WeightLen; w++) {
				for (int j = 0; j < SBox_SIZE; j++) {
					if (DDTorLAT[sbx][i][j] == weight[w]) {
						FWWeightOrderV[sbx][i][index] = j;
						FWWeightOrderW[sbx][i][index] = weight[w];
						FWWeightOrderWMinusMin[sbx][i][index] = weight[w];
						index++;
					}
				}
			}
		}
		bool tag;
		for (int i = 0; i < SBox_SIZE; i++) {
			FWWeightMinandMax[sbx][i][0] = FWWeightOrderW[sbx][i][0]; 
			tag = true;
			for (int j = SBox_SIZE - 1; j >= 0; j--) {
				if (tag && FWWeightOrderW[sbx][i][j] != INFINITY) {
					FWWeightMinandMax[sbx][i][1] = FWWeightOrderW[sbx][i][j];
					tag = false;
				}
				FWWeightOrderWMinusMin[sbx][i][j] -= FWWeightOrderW[sbx][i][0];
			}
		}
	}

	for (int sbx = 0; sbx < SBox_NUM; sbx++) {
		for (int i = 0; i < SBox_SIZE; i++) {
			for (int j = 0; j < SBox_SIZE; j++) {
				DDTorLATMinusMinW[sbx][i][j] = DDTorLAT[sbx][i][j] - FWWeightMinandMax[sbx][i][0];
			}
		}
	}

	memset(IRFWMinV, 0, sizeof(IRFWMinV));
	memset(IRFWMinW, 0, sizeof(IRFWMinW));
	for (int sbx = 0; sbx < SBox_NUM; sbx++) {
		index = 0;
		for (int tmp_min = 0; tmp_min <= 2; tmp_min++) {
			for (int tmp_max = tmp_min; tmp_max <= 2; tmp_max++) {
				for (int i = 0; i < SBox_SIZE; i++) {
					if (FWWeightMinandMax[sbx][i][0] == weight[tmp_min] && FWWeightMinandMax[sbx][i][1] == weight[tmp_max]) {
						IRFWMinV[sbx][index] = i;
						IRFWMinW[sbx][index] = weight[tmp_min] - weight[1];
						index++;
					}
				}
			}
		}		
	}
}

void GenASP(__m128i input, int last_asn, int index, int len, int& ArrIndex) {
	if (index == len) {
		int index1 = 0, index2 = 0;
#if(TYPE)
		__m128i tmp_input = _mm_xor_si128(_mm_slli_epi64(input, 16), _mm_srli_epi64(input, 48));
#else
		__m128i tmp_input = _mm_xor_si128(_mm_slli_epi64(input, 48), _mm_srli_epi64(input, 16));
#endif		
		for (int i = 0; i < SBox_NUM; i++) {
			if (input.m128i_u8[i]) {
				ASP_FW_Info[ArrIndex][index1] = i;
				index1++;
			}
			if (tmp_input.m128i_u8[i]) {
				ASP_BW_Info[ArrIndex][index2] = i;
				index2++;
			}
		}
		for (index1; index1 < SBox_NUM; index1++) {
			ASP_FW_Info[ArrIndex][index1] = 0;
			ASP_BW_Info[ArrIndex][index1] = 0;
		}
		ASPInfo[ArrIndex] = len;
		ArrIndex++;
		return;
	}
	else {
		for (int i = last_asn; i < SBox_NUM - len + index + 1; i++) {
			input.m128i_u8[i] = 1;
			GenASP(input, i + 1, index + 1, len, ArrIndex);
			input.m128i_u8[i] = 0;
		}
	}
}

void GenASPbyInt(__m128i input, int aspIndex) {
	bitset<8> bitASP = aspIndex;
	for (int i = 0; i < SBox_NUM; i++) input.m128i_u8[i] = bitASP[i];
	int index1 = 0, index2 = 0;
#if(TYPE)
	__m128i tmp_input = _mm_xor_si128(_mm_slli_epi64(input, 16), _mm_srli_epi64(input, 48));
#else
	__m128i tmp_input = _mm_xor_si128(_mm_slli_epi64(input, 48), _mm_srli_epi64(input, 16));
#endif		
	for (int i = 0; i < SBox_NUM; i++) {
		if (input.m128i_u8[i]) {
			ASP_FW_Info[aspIndex - 1][index1] = i;
			index1++;
		}
		if (tmp_input.m128i_u8[i]) {
			ASP_BW_Info[aspIndex - 1][index2] = i;
			index2++;
		}
	}
	for (index1; index1 < SBox_NUM; index1++) {
		ASP_FW_Info[aspIndex - 1][index1] = 0;
		ASP_BW_Info[aspIndex - 1][index1] = 0;
	}
	ASPInfo[aspIndex - 1] = index2;
}

void GenDCArr() {
	memset(ASP_FW_Info, 0, sizeof(ASP_FW_Info));
	memset(ASP_BW_Info, 0, sizeof(ASP_BW_Info));
	memset(ASPInfo, 0, sizeof(ASPInfo));
	__m128i tmp_out = _mm_setzero_si128();
	for (int i = 1; i <= ASP_NUM; i++) {
		GenASPbyInt(tmp_out, i);
	}

	memset(LB, 0, sizeof(LB));
	memset(LBNA0, 0, sizeof(LBNA0));
	memset(BWLB, 0, sizeof(BWLB));
	memset(FWLB, 0, sizeof(FWLB));

	memset(ASNLB, 0, sizeof(ASNLB));
	memset(ASNBWLB, 0, sizeof(ASNBWLB));
	memset(ASNFWLB, 0, sizeof(ASNFWLB));
	memset(ASNBWLBOver, 0, sizeof(ASNBWLBOver));
	memset(ASNFWLBOver, 0, sizeof(ASNFWLBOver));

	__m128i tmp_sbx1fw, tmp_sbx2fw, tmp_sbx1bw, tmp_sbx2bw;
	__m128i Mask = _mm_setzero_si128();
	int asnFw, asnBw;

	LB[1][0] = 0; LB[1][1] = weight[1]; LB[2][0] = weight[1]; LB[2][1] = 2 * weight[1];
	ASNLB[1][0] = 0; ASNLB[1][1] = 1; ASNLB[2][0] = 1; ASNLB[2][1] = 2;
	

	for (int i = 0; i < ASP_NUM; i++) {		

		FWLB[1][i] = ASPInfo[i] * weight[1];
		BWLB[1][i] = ASPInfo[i] * weight[1]; 
		LBNA0[2][1][i] = FWLB[1][i]; LBNA0[2][2][i] = BWLB[1][i];
		
		FWLB[2][i] = 2 * FWLB[1][i]; BWLB[2][i] = 2 * BWLB[1][i];
		LBNA0[3][1][i] = FWLB[2][i]; LBNA0[3][2][i] = FWLB[1][i] + BWLB[1][i]; LBNA0[3][3][i] = BWLB[2][i]; 

		ASNBWLB[1][i] = ASPInfo[i]; ASNBWLBOver[1][i] = true;
		ASNFWLB[1][i] = ASPInfo[i]; ASNFWLBOver[1][i] = true;
		ASNBWLB[2][i] = ASPInfo[i] * 2; ASNBWLBOver[2][i] = true;
		ASNFWLB[2][i] = ASPInfo[i] * 2; ASNFWLBOver[2][i] = true;
	}

}

void GenTables() {
	WeightOrderTables();
	GenDCArr();
}



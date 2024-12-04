#include<iostream>
#include<emmintrin.h>
#include<nmmintrin.h>
#include<ctime>
#include<string>
#include<fstream>
#include<sstream>
#include<iomanip>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
using namespace std;


ALIGNED_TYPE_(__m128i, 16) Trail_BW[RNUM + 1];        
ALIGNED_TYPE_(__m128i, 16) Trail_FW[RNUM + 1];        
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];   
ALIGNED_TYPE_(__m128i, 16) NxtBestTrail[RNUM + 1];
ALIGNED_TYPE_(__m128i, 16) TMPX[RNUM + 1];      
ALIGNED_TYPE_(__m128i, 16) TMPX2R[RNUM + 1];
ALIGNED_TYPE_(__m128i, 16) TMPX2R_ASN[RNUM + 1]; 
int T_W_BW[RNUM];
int T_W_FW[RNUM];
int Best_W[RNUM];
int NxtBest_W[RNUM];
int Extern2RMask[RNUM + 1];


STATE R2STATE;

int BASN_PC_FW[RNUM + 1][SBox_NUM];
int BASN_PC_BW[RNUM + 1][SBox_NUM];


ALIGNED_TYPE_(__m128i, 16) GenBnTrail[2];
int GenBn_W[3];  
int ExternRound;
int ExternBnRound; 
bool ExternDir;  
bool GenBnTag;
int BnNAIndex;   
int BNPRnum;

bool FindNA0, FindNA1; 
ALIGNED_TYPE_(__m128i, 16) TMPNA0_TRAIL[RNUM + 1];
int TMPNA0_W[RNUM];
int NxtNA0Bn;
bool PreSearchTag; 


int BASN;
bool FindBASN;

int Bn, BWBn; 
int BestB[RNUM + 1] = { 0 };
bool FindBn; 
bool FindSub; 
bool BWSearchOver;
int FWASN; 
int R2ASNW;
int NPRnum, ASP_INDEX; 
int ASP_Value;
int Rnum;

__m128i Mask = _mm_setzero_si128();


map<pair<u8, u8>, RecordLBForValue_ASN> ASPandValueMapLB_ASN_FW;
map<pair<u8, u8>, RecordLBForValue_ASN> ASPandValueMapLB_ASN_BW;


map<pair<pair<u8, u8>, pair<u8, u8>>, RecordLBForValue_ASN> ASPandValueMapLB_ASN_FW_j;
map<pair<pair<u8, u8>, pair<u8, u8>>, RecordLBForValue_ASN> ASPandValueMapLB_ASN_BW_j; 


void logToFile(const string& fileNameR, const string& message) {
	ofstream file(fileNameR, ios::app);
	if (!file.is_open()) {
		cerr << "Unable to open file: " << fileNameR << endl;
		return;
	}
	file << message;
	file.close();
}

void FileOutputTrail() {
#if(TYPE)
	string fileName = "LBlock_Linear_Trail.txt";
#else
	string fileName = "LBlock_Diff_Trail.txt";
#endif
#if(TYPE)
	//Linear
	ALIGNED_TYPE_(__m128i, 16) SI[RNUM];
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM];
	memset(SI, 0, sizeof(SI));
	memset(SO, 0, sizeof(SO));

	for (int r = 0; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			SO[r].m128i_u8[i] = BestTrail[r + 1].m128i_u8[FWINVSBoxPermutation[i]];
		}
	}

	for (int i = 0; i < SBox_NUM; i++) {
		if (SO[0].m128i_u8[i]) {
			SI[0].m128i_u8[i] = FWWeightOrderV[i][SO[0].m128i_u8[i]][0];
		}
		if (SO[Rnum - 1].m128i_u8[i]) {
			SI[Rnum - 1].m128i_u8[i] = FWWeightOrderV[i][SO[Rnum - 1].m128i_u8[i]][0];
		}
	}

	__m128i tmp;
	for (int r = 1; r < Rnum - 1; r++) {
		SI[r] = _mm_xor_si128(BestTrail[r + 2], _mm_xor_si128(_mm_slli_epi64(BestTrail[r], 48), _mm_srli_epi64(BestTrail[r], 16)));
	}

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Bn << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0x7; k >= 0; k--) {
			message << hex << static_cast<int>(SI[Rnum - 1 - r].m128i_u8[k]);
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0x7; k >= 0; k--) {
			message << hex << static_cast<int>(SO[Rnum - 1 - r].m128i_u8[k]);
		}
		message << "  w: " << dec << Best_W[Rnum - 1 - r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());

#else
	//Diff
	ALIGNED_TYPE_(__m128i, 16) SI[RNUM];
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM];
	memset(SI, 0, sizeof(SI));
	memset(SO, 0, sizeof(SO));

	memcpy(SI, &BestTrail[1], Rnum * STATE_LEN);

	for (int i = 0; i < SBox_NUM; i++) {
		if (BestTrail[1].m128i_u8[i]) {
			SO[0].m128i_u8[i] = FWWeightOrderV[i][BestTrail[1].m128i_u8[i]][0];
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (BestTrail[Rnum].m128i_u8[i]) {
			SO[Rnum - 1].m128i_u8[i] = FWWeightOrderV[i][BestTrail[Rnum].m128i_u8[i]][0];
		}
	}

	__m128i tmp;
	for (int r = 1; r < Rnum - 1; r++) {
		tmp = _mm_xor_si128(SI[r + 1], _mm_xor_si128(_mm_slli_epi64(SI[r - 1], 16), _mm_srli_epi64(SI[r - 1], 48)));
		for (int i = 0; i < SBox_NUM; i++) {
			if (tmp.m128i_u8[i]) {
				SO[r].m128i_u8[FWINVSBoxPermutation[i]] = tmp.m128i_u8[i];
			}
		}
	}

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Bn << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0x7; k >= 0; k--) {
			message << hex << static_cast<int>(SI[r].m128i_u8[k]);

		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0x7; k >= 0; k--) {
			message << hex << static_cast<int>(SO[r].m128i_u8[k]);
		}
		message << "  w: " << dec << Best_w[r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());
#endif
}


inline void UpdateBWLBandLBNA0(int w) {
	BWLB[NPRnum - 1][ASP_INDEX] = w;
	for (int r1 = NPRnum; r1 <= Rnum; r1++) {
		if (LBNA0[r1][NPRnum][ASP_INDEX] < BWLB[NPRnum - 1][ASP_INDEX] + FWLB[r1 - NPRnum][ASP_INDEX]) {
			LBNA0[r1][NPRnum][ASP_INDEX] = BWLB[NPRnum - 1][ASP_INDEX] + FWLB[r1 - NPRnum][ASP_INDEX];
			for (int r2 = r1 + 1; r2 <= Rnum; r2++) {
				LBNA0[r2][NPRnum][ASP_INDEX] = (LBNA0[r2][NPRnum][ASP_INDEX] > LBNA0[r1][NPRnum][ASP_INDEX] + LB[r2 - r1][0]) ?
					LBNA0[r2][NPRnum][ASP_INDEX] : LBNA0[r1][NPRnum][ASP_INDEX] + LB[r2 - r1][0];
			}
		}
	}

	bool tag;
	for (int i = NPRnum; i < Rnum; i++) {
		tag = true;
		for (int i2 = NPRnum - 1; i2 < i; i2++) {
			if (BWLB[i][ASP_INDEX] < (BWLB[i2][ASP_INDEX] + LB[i - i2][1])) {
				BWLB[i][ASP_INDEX] = BWLB[i2][ASP_INDEX] + LB[i - i2][1];
				tag = false;
			}
		}
		if (tag) continue;
		for (int r1 = i + 1; r1 <= Rnum; r1++) {
			if (LBNA0[r1][i + 1][ASP_INDEX] < BWLB[i][ASP_INDEX] + FWLB[r1 - i - 1][ASP_INDEX]) {
				LBNA0[r1][i + 1][ASP_INDEX] = BWLB[i][ASP_INDEX] + FWLB[r1 - i - 1][ASP_INDEX];
				for (int r2 = r1 + 1; r2 <= Rnum; r2++) {
					LBNA0[r2][i + 1][ASP_INDEX] = (LBNA0[r2][i + 1][ASP_INDEX] > LBNA0[r1][i + 1][ASP_INDEX] + LB[r2 - r1][0]) ?
						LBNA0[r2][i + 1][ASP_INDEX] : LBNA0[r1][i + 1][ASP_INDEX] + LB[r2 - r1][0];
				}
			}
		}
	}
}

inline void UpdateASNBWLB() {
	if (FindBASN) {
		ASNBWLB[NPRnum - 1][ASP_INDEX] = BASN; ASNBWLBOver[NPRnum - 1][ASP_INDEX] = true;
	}
	else ASNBWLB[NPRnum - 1][ASP_INDEX] = BASN + 1;
	for (int r1 = NPRnum; r1 < Rnum; r1++) {
		for (int r2 = NPRnum - 1; r2 < r1; r2++) {
			ASNBWLB[r1][ASP_INDEX] = (ASNBWLB[r1][ASP_INDEX] > (ASNBWLB[r2][ASP_INDEX] + ASNLB[r1 - r2][1])) ?
				ASNBWLB[r1][ASP_INDEX] : ASNBWLB[r2][ASP_INDEX] + ASNLB[r1 - r2][1];
		}
	}

	if (BWLB[NPRnum - 1][ASP_INDEX] < (ASNBWLB[NPRnum - 1][ASP_INDEX] * weight[1])) UpdateBWLBandLBNA0(ASNBWLB[NPRnum - 1][ASP_INDEX] * weight[1]);
}

inline void UpdateASNFWLB() {
	if (FindBASN) {
		ASNFWLB[Rnum - NPRnum][ASP_INDEX] = BASN; ASNFWLBOver[Rnum - NPRnum][ASP_INDEX] = true;
	}
	else ASNFWLB[Rnum - NPRnum][ASP_INDEX] = BASN + 1;
	for (int r1 = Rnum - NPRnum + 1; r1 < Rnum; r1++) {
		for (int r2 = Rnum - NPRnum; r2 < r1; r2++) {
			ASNFWLB[r1][ASP_INDEX] = (ASNFWLB[r1][ASP_INDEX] > (ASNFWLB[r2][ASP_INDEX] + ASNLB[r1 - r2][0])) ?
				ASNFWLB[r1][ASP_INDEX] : ASNFWLB[r2][ASP_INDEX] + ASNLB[r1 - r2][0];
		}
	}

	FWLB[Rnum - NPRnum][ASP_INDEX] = (FWLB[Rnum - NPRnum][ASP_INDEX] > (ASNFWLB[Rnum - NPRnum][ASP_INDEX] * weight[1])) ?
		FWLB[Rnum - NPRnum][ASP_INDEX] : (ASNFWLB[Rnum - NPRnum][ASP_INDEX] * weight[1]);
	for (int r1 = Rnum - NPRnum + 1; r1 < Rnum; r1++) {
		for (int r2 = Rnum - NPRnum; r2 < r1; r2++) {
			FWLB[r1][ASP_INDEX] = (FWLB[r1][ASP_INDEX] > (FWLB[r2][ASP_INDEX] + LB[r1 - r2][0])) ?
				FWLB[r1][ASP_INDEX] : FWLB[r2][ASP_INDEX] + LB[r1 - r2][0];
			FWLB[r1][ASP_INDEX] = (FWLB[r1][ASP_INDEX] > (ASNFWLB[r1][ASP_INDEX] * weight[1])) ?
				FWLB[r1][ASP_INDEX] : ASNFWLB[r1][ASP_INDEX] * weight[1];
		}
	}
}

// search for active sboxes
void FWRound_n_ASN(STATE s) {
	s.W += s.w;
	BASN = s.W;
	FindBASN = true;
	return;
}

void FWRound_i_ASN(STATE s, __m128i sbx_out) {
	if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + ASNLB[Rnum - s.rnum - 1][FWASN] > BASN))
		|| (FindBASN && (s.W + s.w + s.nr_sbx_num + ASNLB[Rnum - s.rnum - 1][FWASN] >= BASN))) return;
	if (s.sbx_tag[s.j]) {
		for (s.j; s.j <= s.sbx_num; s.j++)
			sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]] = 1;

		if (s.rnum + 1 == Rnum) {
			STATE nxt_s = UpdateStateRoundN_ASN(s, 0);
			FWRound_n_ASN(nxt_s);				
		}
		else {
			Trail_FW[s.rnum + 1] = sbx_out;
#if(TYPE)
			TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[s.rnum + 1], 16));
#else
			TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[s.rnum + 1], 48));
#endif
			Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask));

			STATE nxt_s = FWUpdateStateRoundI_ASN(s, 0);
			if ((!FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[Rnum - nxt_s.rnum - 1][FWASN] > BASN))
				|| (FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[Rnum - nxt_s.rnum - 1][FWASN] >= BASN))) return;
			FWRound_i_ASN(nxt_s, TMPX[s.rnum]);
		}
	}
	else {
		int sbx_nr_w = 0;
		u8 tmp2R = TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]];
		for (int i = 0; i <= 1; i++) {
			sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]] = i;
			if (i) {
				sbx_nr_w = 1;
				TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = 1;
				if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + ASNLB[Rnum - s.rnum - 1][FWASN] > BASN))
					|| (FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + ASNLB[Rnum - s.rnum - 1][FWASN] >= BASN))) continue;
			}
			else {
				TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = 0;
			}

			if (s.rnum + 1 == Rnum) {
				if (s.j == s.sbx_num) {
					STATE nxt_s = UpdateStateRoundN_ASN(s, sbx_nr_w);
					if (nxt_s.w >= FWASN)
						FWRound_n_ASN(nxt_s);
				}
				else {
					FWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				}
			}
			else {
				int tmp_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX2R_ASN[s.rnum], Mask));
				int asn = _mm_popcnt_u32(tmp_mask ^ Extern2RMask[s.rnum]); asn = (asn > FWASN) ? asn : FWASN;
				if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + asn + ASNLB[Rnum - s.rnum - 2][FWASN] > BASN))
					|| (FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + asn + ASNLB[Rnum - s.rnum - 2][FWASN] >= BASN))) continue;
				if (s.j == s.sbx_num) {
					Trail_FW[s.rnum + 1] = sbx_out;
#if(TYPE)
					TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[s.rnum + 1], 16));
#else
					TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[s.rnum + 1], 48));
#endif
					Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask));

					STATE nxt_s = FWUpdateStateRoundI_ASN(s, sbx_nr_w);

					if ((nxt_s.sbx_num + 1 < FWASN)
						|| (!FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[Rnum - nxt_s.rnum - 1][FWASN] > BASN))
						|| (FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[Rnum - nxt_s.rnum - 1][FWASN] >= BASN))) continue;

					if (nxt_s.w) FWRound_i_ASN(nxt_s, TMPX[s.rnum]);
					else { 			
						Trail_FW[nxt_s.rnum + 1] = TMPX[s.rnum];
#if(TYPE)
						TMPX[nxt_s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[nxt_s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[nxt_s.rnum + 1], 16));
#else
						TMPX[nxt_s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[nxt_s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[nxt_s.rnum + 1], 48));
#endif
						Extern2RMask[nxt_s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[nxt_s.rnum + 1], Mask));

						STATE nxtnxt_s = FWUpdateStateRoundI_ASN(nxt_s, 0);

						if (nxtnxt_s.rnum == Rnum) FWRound_n_ASN(nxtnxt_s);
						else if ((!FindBASN && (nxtnxt_s.W + nxtnxt_s.w + nxtnxt_s.nr_sbx_num + ASNLB[Rnum - nxtnxt_s.rnum - 1][FWASN] <= BASN))
							|| (FindBASN && (nxtnxt_s.W + nxtnxt_s.w + nxtnxt_s.nr_sbx_num + ASNLB[Rnum - nxtnxt_s.rnum - 1][FWASN] < BASN))) {
							FWRound_i_ASN(nxtnxt_s, TMPX[nxt_s.rnum]);
						}
					}

				}
				else {
					FWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				}
			}
		}
		TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
	}	
	return;
}

void BWRound_n_ASN(STATE s) {
	s.W += s.w;
	BASN = s.W;
	FindBASN = true;
	return;
}

void BWRound_i_ASN(STATE s, __m128i sbx_out) {
	if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + ASNLB[s.rnum - 2][1] > BASN))
		|| (FindBASN && (s.W + s.w + s.nr_sbx_num + ASNLB[s.rnum - 2][1] >= BASN))) return;

	if (s.sbx_tag[s.j]) {
		if (s.rnum == 2) {
			STATE nxt_s = UpdateStateRoundN_ASN(s, 0);
			BWRound_n_ASN(nxt_s);
		}
		else {
			for (s.j; s.j <= s.sbx_num; s.j++)
				sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]] = 1;

			Trail_BW[s.rnum - 1] = sbx_out;
#if(TYPE)
			TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 16), _mm_srli_epi64(Trail_BW[s.rnum - 1], 48));
#else
			TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 48), _mm_srli_epi64(Trail_BW[s.rnum - 1], 16));
#endif
			Extern2RMask[s.rnum - 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum - 1], Mask));
			STATE nxt_s = BWUpdateStateRoundI_ASN(s, 0);
			if ((!FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[nxt_s.rnum - 2][1] > BASN))
				|| (FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[nxt_s.rnum - 2][1] >= BASN))) return;
			BWRound_i_ASN(nxt_s, TMPX[s.rnum]);
		}	
	}
	else {
		int sbx_nr_w = 0; u8 tmp2R = TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]];
		for (int i = 0; i <= 1; i++) {
			sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]] = i;
			if (i) {
				sbx_nr_w = 1;
				TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = 1;
				if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + ASNLB[s.rnum - 2][1] > BASN))
					|| (FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + ASNLB[s.rnum - 2][1] >= BASN))) continue;
			}
			else {
				TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = 0;
			}			

			if (s.rnum == 2) {
				if (s.j == s.sbx_num) {
					STATE nxt_s = UpdateStateRoundN_ASN(s, sbx_nr_w);
					if (nxt_s.w)
						BWRound_n_ASN(nxt_s);
				}
				else {
					BWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				}
			}
			else {
				int tmp_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX2R_ASN[s.rnum], Mask));
				int asn = _mm_popcnt_u32(tmp_mask ^ Extern2RMask[s.rnum]); asn = (asn > 1) ? asn : 1;
				if ((!FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + asn + ASNLB[s.rnum - 3][1] > BASN))
					|| (FindBASN && (s.W + s.w + s.nr_sbx_num + sbx_nr_w + asn + ASNLB[s.rnum - 3][1] >= BASN))) continue;

				if (s.j == s.sbx_num) {
					Trail_BW[s.rnum - 1] = sbx_out;
#if(TYPE)
					TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 16), _mm_srli_epi64(Trail_BW[s.rnum - 1], 48));
#else
					TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 48), _mm_srli_epi64(Trail_BW[s.rnum - 1], 16));
#endif
					Extern2RMask[s.rnum - 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum - 1], Mask)); 
					STATE nxt_s = BWUpdateStateRoundI_ASN(s, sbx_nr_w);
					if ((nxt_s.sbx_num < 0)
						|| (!FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[nxt_s.rnum - 2][1] > BASN))
						|| (FindBASN && (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + ASNLB[nxt_s.rnum - 2][1] >= BASN))) continue;
					BWRound_i_ASN(nxt_s, TMPX[s.rnum]);
				}
				else {
					BWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				}
			}
		}
		TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
	}
	return;
}


void BWRound_n(STATE s) {
	s.W               += s.w;
	T_W_BW[s.rnum - 1] = s.w;
	BWBn               = s.W;
	BWSearchOver       = true;
	if (GenBnTag) {
		Bn			  = BWBn;
		FindBn		  = true;
		ExternDir	  = false;
		ExternBnRound = ExternRound;
		memcpy(GenBnTrail, &Trail_BW[1], ExternRound * STATE_LEN);
		memcpy(GenBn_W, T_W_BW, (ExternRound + 1) * sizeof(int));
	
	}
	else {
		if (NPRnum == Rnum) {
			Bn		= BWBn;
			FindBn  = true;
			FindSub = true;
			BNPRnum = NPRnum;
			memcpy(BestTrail, Trail_BW, (Rnum + 1) * STATE_LEN);
			memcpy(Best_W, T_W_BW, Rnum * sizeof(int));
		}
		else {			
			memcpy(Trail_FW, &Trail_BW[0], NPRnum * STATE_LEN);
			memcpy(T_W_FW, &T_W_BW[0], (NPRnum - 1) * sizeof(int));
		}
	}	
	
	return;
}

void BWRound_i(STATE s, __m128i sbx_out) {
	int tmp_sbxout = sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]];
	int sbx_nr_w = 0, sbx_nr_num = 0, nr_sbx_w = 0;
	u8 tmp_asp1, tmp_asp2;
	u8 tmp_searchnum, tmp_searchOutput; 
	u8 tmp2R = TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]];
	if (tmp_sbxout) {
		nr_sbx_w = weight[1];
		if (s.j != s.sbx_num) {
			tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum + 1], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum], Mask));
			tmp_searchnum = s.j; tmp_searchOutput = 0;
			for (int ASB_Output = 0; ASB_Output < s.j; ASB_Output++) if (sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[ASB_Output]]]) tmp_searchOutput ^= (1 << ASB_Output);
		}
		if ((!(FindBn || BWSearchOver) && (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw + BASN_PC_BW[s.rnum][s.j] > BWBn))
			|| ((FindBn || BWSearchOver) && (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw + BASN_PC_BW[s.rnum][s.j] >= BWBn))) goto SBOXOUTNOZERO_BW;
		sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]] = 0; TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = 0;		
		if (s.j == s.sbx_num && s.nr_sbx_num > 0) {
			if (s.rnum == 2) {
				Trail_BW[s.rnum - 1] = sbx_out;
				STATE nxt_s = BWUpdateStateRoundN(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w);
				BWRound_n(nxt_s);
			}
			else {
				Trail_BW[s.rnum - 1] = sbx_out;
#if(TYPE)
				TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 16), _mm_srli_epi64(Trail_BW[s.rnum - 1], 48));
#else
				TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 48), _mm_srli_epi64(Trail_BW[s.rnum - 1], 16));
#endif
				Extern2RMask[s.rnum - 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum - 1], Mask)); 

				STATE nxt_s = BWUpdateStateRoundI(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w);

				if ((!(FindBn || BWSearchOver) && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[nxt_s.rnum - 2][1] > BWBn))
					|| ((FindBn || BWSearchOver) && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[nxt_s.rnum - 2][1] >= BWBn)))  goto SBOXOUTNOZERO_BW;

				tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum - 1], Mask));				
				auto itor_ASN = ASPandValueMapLB_ASN_BW.find(make_pair(tmp_asp1, tmp_asp2));
				int asn_lb = 0; bool searchasn_tag = true;
				if (itor_ASN != ASPandValueMapLB_ASN_BW.end()) {
					asn_lb = itor_ASN->second.ReturnLB(nxt_s.rnum, FWASN);
					if (itor_ASN->second.RdLBOver[nxt_s.rnum - 1]) searchasn_tag = false;
				}
				
				if (!(FindBn || BWSearchOver)) BASN = ((BWBn - nxt_s.W - nxt_s.w - nxt_s.nr_minw) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
				else BASN = ((BWBn - nxt_s.W - nxt_s.w - nxt_s.nr_minw - 1) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;

				if (BASN < asn_lb) goto SBOXOUTNOZERO_BW;
				if (searchasn_tag) {
					FindBASN = false;
					BWRound_i_ASN(GenStateRI_ASN(nxt_s), TMPX[s.rnum]);
					if (itor_ASN != ASPandValueMapLB_ASN_BW.end()) {
						if (FindBASN) {
							itor_ASN->second.UpdateOrInsertLB(nxt_s.rnum, BASN); itor_ASN->second.RdLBOver[nxt_s.rnum - 1] = true;							
						}
						else itor_ASN->second.UpdateOrInsertLB(nxt_s.rnum, BASN + 1);
					}
					else {
						RecordLBForValue_ASN newValueLB_ASN(nxt_s.rnum, BASN);
						if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(nxt_s.rnum, BASN + 1);
						else newValueLB_ASN.RdLBOver[nxt_s.rnum - 1] = true;
						ASPandValueMapLB_ASN_BW.insert(make_pair(make_pair(tmp_asp1, tmp_asp2), newValueLB_ASN));
					}
					if (!FindBASN) goto SBOXOUTNOZERO_BW;
				}
				else BASN = asn_lb;

				BASN_PC_BW[nxt_s.rnum][0] = (BASN - s.nr_sbx_num - nxt_s.nr_sbx_num) * weight[1];
				BASN_PC_BW[nxt_s.rnum][0] = (BASN_PC_BW[nxt_s.rnum][0] > LB[nxt_s.rnum - 2][1]) ? BASN_PC_BW[nxt_s.rnum][0] : LB[nxt_s.rnum - 2][1];
				BWRound_i(nxt_s, TMPX[s.rnum]);
			}
		}
		else if (s.j != s.sbx_num) {
			if (!(FindBn || BWSearchOver)) BASN = ((BWBn - (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
			else BASN = ((BWBn - (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;

			auto itor_ASN_j = ASPandValueMapLB_ASN_BW_j.find(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)));
			bool searchasn_tag = true; int asn_lb = 0;
			if (itor_ASN_j != ASPandValueMapLB_ASN_BW_j.end()) {
				asn_lb = itor_ASN_j->second.ReturnLB(s.rnum, 1);
				if (itor_ASN_j->second.RdLBOver[s.rnum - 1]) searchasn_tag = false; 
			}
			if (BASN < asn_lb) goto SBOXOUTNOZERO_BW;

			if (searchasn_tag) {
				FindBASN = false;
				BWRound_i_ASN(GenStateRI_j_ASN(s, sbx_nr_num), sbx_out);
				if (itor_ASN_j != ASPandValueMapLB_ASN_BW_j.end()) {
					if (FindBASN) { itor_ASN_j->second.UpdateOrInsertLB(s.rnum, BASN); itor_ASN_j->second.RdLBOver[s.rnum - 1] = true; }
					else itor_ASN_j->second.UpdateOrInsertLB(s.rnum, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(s.rnum, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(s.rnum, BASN + 1); else newValueLB_ASN.RdLBOver[s.rnum - 1] = true;
					ASPandValueMapLB_ASN_BW_j.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)), newValueLB_ASN));
				}
				if (!FindBASN) goto SBOXOUTNOZERO_BW;
			}
			else BASN = asn_lb;

			BASN_PC_BW[s.rnum][s.j + 1] = (BASN - s.sbx_num - s.nr_sbx_num - 1) * weight[1];
			BASN_PC_BW[s.rnum][s.j + 1] = (BASN_PC_BW[s.rnum][s.j + 1] > LB[s.rnum - 2][1]) ? BASN_PC_BW[s.rnum][s.j + 1] : LB[s.rnum - 2][1];
			BWRound_i(UpdateStateRoundI_j(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w), sbx_out);
		}

		SBOXOUTNOZERO_BW:
		int tmp_BASN_PC = BASN_PC_BW[s.rnum][s.j] - weight[1];
		BASN_PC_BW[s.rnum][s.j] = (tmp_BASN_PC > LB[s.rnum - 2][1]) ? tmp_BASN_PC : LB[s.rnum - 2][1];
	}


	sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]] = 1;
	if (!s.sbx_tag[s.j]) { 
		sbx_nr_num = 1;
		TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = 1;
	}
	int RecordMinAsn = 0;
	FindBASN = false;
	if (!(FindBn || BWSearchOver)) BASN = ((BWBn - (s.W + s.w + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
	else BASN = ((BWBn - (s.W + s.w + s.nr_minw) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;

	if (s.j != s.sbx_num && !s.sbx_tag[s.j]) {
		tmp_searchOutput ^= (1 << s.j);
		auto itor_ASN_j = ASPandValueMapLB_ASN_BW_j.find(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)));
		bool searchasn_tag = true; int asn_lb = 0;
		if (itor_ASN_j != ASPandValueMapLB_ASN_BW_j.end()) {
			asn_lb = itor_ASN_j->second.ReturnLB(s.rnum, 1);
			if (itor_ASN_j->second.RdLBOver[s.rnum - 1]) searchasn_tag = false; 
		}
		if (BASN >= asn_lb) {
			if (searchasn_tag) {
				BWRound_i_ASN(GenStateRI_j_ASN(s, sbx_nr_num), sbx_out);
				if (itor_ASN_j != ASPandValueMapLB_ASN_BW_j.end()) {
					if (FindBASN) { itor_ASN_j->second.UpdateOrInsertLB(s.rnum, BASN); itor_ASN_j->second.RdLBOver[s.rnum - 1] = true; }
					else itor_ASN_j->second.UpdateOrInsertLB(s.rnum, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(s.rnum, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(s.rnum, BASN + 1); else newValueLB_ASN.RdLBOver[s.rnum - 1] = true;
					ASPandValueMapLB_ASN_BW_j.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)), newValueLB_ASN));
				}
			}
			else {
				FindBASN = true; BASN = asn_lb;
			}
		}
	}
	else if (s.j != s.sbx_num) {
		FindBASN = true; BASN = BASN_PC_BW[s.rnum][s.j] / weight[1] + s.sbx_num + 1 + s.nr_sbx_num;
	}
	else if (s.rnum != 2) { 
		Trail_BW[s.rnum - 1] = sbx_out;
#if(TYPE)
		TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 16), _mm_srli_epi64(Trail_BW[s.rnum - 1], 48));
#else
		TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 48), _mm_srli_epi64(Trail_BW[s.rnum - 1], 16));
#endif
		Extern2RMask[s.rnum - 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum - 1], Mask)); 

		BASN -= (s.sbx_num + 1);

		tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_BW[s.rnum - 1], Mask));		
		auto itor_ASN = ASPandValueMapLB_ASN_BW.find(make_pair(tmp_asp1, tmp_asp2));
		int asn_lb = 0; bool searchasn_tag = true;
		if (itor_ASN != ASPandValueMapLB_ASN_BW.end()) {
			asn_lb = itor_ASN->second.ReturnLB(s.rnum - 1, 1);
			if (itor_ASN->second.RdLBOver[s.rnum - 2]) searchasn_tag = false;
		}
		if (BASN >= asn_lb) {
			if (searchasn_tag) {
				BWRound_i_ASN(GenStateRI_ASN_BW(s), TMPX[s.rnum]);
				if (itor_ASN != ASPandValueMapLB_ASN_BW.end()) {
					if (FindBASN) {
						itor_ASN->second.UpdateOrInsertLB(s.rnum - 1, BASN); itor_ASN->second.RdLBOver[s.rnum - 2] = true;
					}
					else itor_ASN->second.UpdateOrInsertLB(s.rnum - 1, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(s.rnum - 1, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(s.rnum - 1, BASN + 1); else newValueLB_ASN.RdLBOver[s.rnum - 2] = true;					
					ASPandValueMapLB_ASN_BW.insert(make_pair(make_pair(tmp_asp1, tmp_asp2), newValueLB_ASN));
				}
			}
			else {
				FindBASN = true; BASN = asn_lb;			
			}			
		}		
	}
	else FindBASN = true;

	if (FindBASN) RecordMinAsn = BASN;
	else {
		TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
		return;
	}
	int AllowMaxASN = 0;
	if (s.j != s.sbx_num && !s.sbx_tag[s.j]) {
		BASN_PC_BW[s.rnum][s.j + 1] = (RecordMinAsn - s.sbx_num - s.nr_sbx_num - sbx_nr_num - 1) * weight[1];
		BASN_PC_BW[s.rnum][s.j + 1] = (BASN_PC_BW[s.rnum][s.j + 1] > LB[s.rnum - 2][1]) ? BASN_PC_BW[s.rnum][s.j + 1] : LB[s.rnum - 2][1];
	}
	else if (s.j != s.sbx_num) BASN_PC_BW[s.rnum][s.j + 1] = BASN_PC_BW[s.rnum][s.j]; 

	for (int i = 0; i < SBox_SIZE; i++) {
		if ((!(FindBn || BWSearchOver) && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + nr_sbx_w + s.nr_minw + BASN_PC_BW[s.rnum][s.j] > BWBn))
			|| ((FindBn || BWSearchOver) && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + nr_sbx_w + s.nr_minw + BASN_PC_BW[s.rnum][s.j] >= BWBn))) break;
		sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]] = (FWWeightOrderV[s.sbx_a[s.j]][s.sbx_in[s.j]][i] ^ tmp_sbxout);
		if (!sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]]) continue;
		sbx_nr_w = FWWeightMinandMax[BWSBoxPermutation[s.sbx_a[s.j]]][sbx_out.m128i_u8[BWSBoxPermutation[s.sbx_a[s.j]]]][0]; 
		if (s.sbx_tag[s.j]) sbx_nr_w -= weight[1];

		if (((sbx_nr_num && sbx_nr_w != weight[1]) || (s.sbx_tag[s.j] && sbx_nr_w)) 
			&& ((!(FindBn || BWSearchOver) && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w + BASN_PC_BW[s.rnum][s.j] > BWBn))
				|| ((FindBn || BWSearchOver) && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w + BASN_PC_BW[s.rnum][s.j] >= BWBn))))
			continue;

		if (s.j == s.sbx_num) {
			if (s.rnum == 2) {
				Trail_BW[s.rnum - 1] = sbx_out;
				STATE nxt_s = BWUpdateStateRoundN(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w);
				BWRound_n(nxt_s);
			}
			else {
				Trail_BW[s.rnum - 1] = sbx_out;
#if(TYPE)
				TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 16), _mm_srli_epi64(Trail_BW[s.rnum - 1], 48));
#else
				TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum - 1], 48), _mm_srli_epi64(Trail_BW[s.rnum - 1], 16));
#endif
				Extern2RMask[s.rnum - 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum - 1], Mask));

				STATE nxt_s = BWUpdateStateRoundI(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w);

				if ((!(FindBn || BWSearchOver) && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[nxt_s.rnum - 2][1] > BWBn))
					|| ((FindBn || BWSearchOver) && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[nxt_s.rnum - 2][1] >= BWBn))) continue;

				if (!(FindBn || BWSearchOver)) AllowMaxASN = ((BWBn - nxt_s.W - nxt_s.w - nxt_s.nr_minw) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
				else AllowMaxASN = ((BWBn - nxt_s.W - nxt_s.w - nxt_s.nr_minw - 1) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
				if (AllowMaxASN >= RecordMinAsn) {
					BASN_PC_BW[nxt_s.rnum][0] = (RecordMinAsn - nxt_s.sbx_num - nxt_s.nr_sbx_num - 1) * weight[1];
					BASN_PC_BW[nxt_s.rnum][0] = (BASN_PC_BW[nxt_s.rnum][0] > LB[nxt_s.rnum - 2][1]) ? BASN_PC_BW[nxt_s.rnum][0] : LB[nxt_s.rnum - 2][1];
					BWRound_i(nxt_s, TMPX[s.rnum]);
				}
			}
		}
		else {
			if (!(FindBn || BWSearchOver)) AllowMaxASN = ((BWBn - (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w)) / weight[1]) + s.sbx_num + s.nr_sbx_num + sbx_nr_num + 1;
			else AllowMaxASN = ((BWBn - (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + sbx_nr_num + 1;
			if (AllowMaxASN < RecordMinAsn) break;
			BWRound_i(UpdateStateRoundI_j(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w), sbx_out);
		}
	}
	TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
	return;
}

void BWRound_NP_GenInput(STATE s, __m128i sbx_in) {
	for (int i = 1; i < SBox_SIZE; i++) {
		if ((!(FindBn || BWSearchOver) && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_BW[0][0] > BWBn))
			|| ((FindBn || BWSearchOver) && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_BW[0][0] >= BWBn))) break;
		s.sbx_in[s.j] = IRFWMinV[s.sbx_a[s.j]][i];  
		sbx_in.m128i_u8[s.sbx_a[s.j]] = IRFWMinV[s.sbx_a[s.j]][i];
		if (s.j == s.sbx_num) {
			Trail_BW[s.rnum] = sbx_in;
#if(TYPE)
			TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum], 16), _mm_srli_epi64(Trail_BW[s.rnum], 48));
#else
			TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[s.rnum], 48), _mm_srli_epi64(Trail_BW[s.rnum], 16));
#endif
			Extern2RMask[s.rnum] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum], Mask)); 
			STATE nxt_s = UpdateStateFWR2andBWInput(s, IRFWMinW[s.sbx_a[s.j]][i]);
			BASN_PC_BW[nxt_s.rnum][0] = BASN_PC_BW[0][0];
			BWRound_i(nxt_s, TMPX[s.rnum + 1]);

		}
		else {
			BWRound_NP_GenInput(UpdateStateRoundI_j(s, IRFWMinW[s.sbx_a[s.j]][i], 0), sbx_in);
		}
	}
	return;
}

void FWRound_n(STATE s) { 
	s.W				  += s.w;
	T_W_FW[s.rnum - 1] = s.w;
	Bn				   = s.W;
	FindBn			   = true;
	FindSub			   = true;	
	if (GenBnTag) {
		ExternDir	  = true;
		ExternBnRound = ExternRound;
		memcpy(GenBnTrail, &Trail_FW[Rnum - ExternBnRound + 1], ExternRound * STATE_LEN);
		memcpy(GenBn_W, &T_W_FW[Rnum - ExternBnRound - 1], (ExternRound + 1) * sizeof(int));

	}
	else {
		BNPRnum = NPRnum;
		memcpy(BestTrail, Trail_FW, (Rnum + 1) * STATE_LEN);
		memcpy(Best_W, T_W_FW, Rnum * sizeof(int));
	}	
	return;
}

void FWRound_i(STATE s, __m128i sbx_out) {
	int tmp_sbxout = sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]];
	int sbx_nr_w = 0, sbx_nr_num = 0, nr_sbx_w = 0;
	u8 tmp_asp1, tmp_asp2; u8 tmp_searchnum, tmp_searchOutput;	
	u8 tmp2R = TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]];
	if (tmp_sbxout) {
		nr_sbx_w = weight[1];
		if (s.j != s.sbx_num) {
			tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[s.rnum - 1], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[s.rnum], Mask));
			tmp_searchnum = s.j; tmp_searchOutput = 0;
			for (int ASB_Output = 0; ASB_Output < s.j; ASB_Output++) if (sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[ASB_Output]]]) tmp_searchOutput ^= (1 << ASB_Output);
		}

		if ((!FindBn && (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw + BASN_PC_FW[s.rnum][s.j] > Bn))
			|| (FindBn && (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw + BASN_PC_FW[s.rnum][s.j] >= Bn))) goto SBOXOUTNOZERO_FW;
		sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]] = 0; TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = 0;
		if (s.j == s.sbx_num && s.nr_sbx_num >= FWASN) {
			if (s.rnum + 1 == Rnum) {
				Trail_FW[s.rnum + 1] = sbx_out;
				STATE nxt_s = FWUpdateStateRoundN(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w);
				FWRound_n(nxt_s);
			}
			else {
				Trail_FW[s.rnum + 1] = sbx_out;
#if(TYPE)
				TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[s.rnum + 1], 16));
#else
				TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[s.rnum + 1], 48));
#endif
				Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
				STATE nxt_s = FWUpdateStateRoundI(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w);

				if (((!FindBn && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[Rnum - nxt_s.rnum - 1][FWASN] > Bn))
					|| (FindBn && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[Rnum - nxt_s.rnum - 1][FWASN] >= Bn))))  goto SBOXOUTNOZERO_FW;

				if (nxt_s.w) { 
					tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[nxt_s.rnum], Mask));					
					auto itor_ASN = ASPandValueMapLB_ASN_FW.find(make_pair(tmp_asp1, tmp_asp2));
					int asn_lb = 0; int searchasn_tag = true;
					if (itor_ASN != ASPandValueMapLB_ASN_FW.end()) {
						asn_lb = itor_ASN->second.ReturnLB(Rnum - s.rnum, FWASN);
						if (itor_ASN->second.RdLBOver[Rnum - s.rnum - 1]) searchasn_tag = false; 
					}					
					if (!FindBn) BASN = ((Bn - nxt_s.W - nxt_s.w - nxt_s.nr_minw) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
					else BASN = ((Bn - nxt_s.W - nxt_s.w - nxt_s.nr_minw - 1) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
					if (BASN < asn_lb) goto SBOXOUTNOZERO_FW;
					if (searchasn_tag) {
						FindBASN = false;
						FWRound_i_ASN(GenStateRI_ASN(nxt_s), TMPX[s.rnum]);
						if (itor_ASN != ASPandValueMapLB_ASN_FW.end()) {
							if (FindBASN) { itor_ASN->second.UpdateOrInsertLB(Rnum - s.rnum, BASN); itor_ASN->second.RdLBOver[Rnum - s.rnum - 1] = true; }
							else itor_ASN->second.UpdateOrInsertLB(Rnum - s.rnum, BASN + 1);
						}
						else {
							RecordLBForValue_ASN newValueLB_ASN(Rnum - s.rnum, BASN);
							if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(Rnum - s.rnum, BASN + 1); else newValueLB_ASN.RdLBOver[Rnum - s.rnum - 1] = true;
							ASPandValueMapLB_ASN_FW.insert(make_pair(make_pair(tmp_asp1, tmp_asp2), newValueLB_ASN));
						}
						if (!FindBASN) goto SBOXOUTNOZERO_FW;
					}
					else BASN = asn_lb;
					
					BASN_PC_FW[nxt_s.rnum][0] = (BASN - s.nr_sbx_num - nxt_s.nr_sbx_num) * weight[1];
					BASN_PC_FW[nxt_s.rnum][0] = (BASN_PC_FW[nxt_s.rnum][0] > LB[Rnum - nxt_s.rnum - 1][FWASN]) ? BASN_PC_FW[nxt_s.rnum][0] : LB[Rnum - nxt_s.rnum - 1][FWASN];
					FWRound_i(nxt_s, TMPX[s.rnum]);
				}
				else {
					Trail_FW[nxt_s.rnum + 1] = TMPX[s.rnum];
#if(TYPE)
					TMPX[nxt_s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[nxt_s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[nxt_s.rnum + 1], 16));
#else
					TMPX[nxt_s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[nxt_s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[nxt_s.rnum + 1], 48));
#endif
					Extern2RMask[nxt_s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[nxt_s.rnum + 1], Mask));
					STATE nxtnxt_s = FWUpdateStateRoundI(nxt_s, 0, 0);
					if (nxtnxt_s.rnum == Rnum) FWRound_n(nxtnxt_s);

					int tmp_aspIndex = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[nxt_s.rnum + 1], Mask)) - 1; 
					int tmp_lb = (nxtnxt_s.w + nxtnxt_s.nr_minw + LB[Rnum - nxtnxt_s.rnum - 1][FWASN] > FWLB[Rnum - nxt_s.rnum][tmp_aspIndex]) ?
						nxtnxt_s.w + nxtnxt_s.nr_minw + LB[Rnum - nxtnxt_s.rnum - 1][FWASN] : FWLB[Rnum - nxt_s.rnum][tmp_aspIndex];
					if ((!FindBn && (nxtnxt_s.W + tmp_lb > Bn)) || (FindBn && (nxtnxt_s.W + tmp_lb >= Bn))) goto SBOXOUTNOZERO_FW;			

					int asn_ub = 0;
					if (!FindBn) asn_ub = ((Bn - nxtnxt_s.W - nxtnxt_s.w - nxtnxt_s.nr_minw) / weight[1]) + nxtnxt_s.sbx_num + nxtnxt_s.nr_sbx_num + 1;
					else asn_ub = ((Bn - nxtnxt_s.W - nxtnxt_s.w - nxtnxt_s.nr_minw - 1) / weight[1]) + nxtnxt_s.sbx_num + nxtnxt_s.nr_sbx_num + 1;
					if (ASNFWLB[Rnum - nxt_s.rnum][tmp_aspIndex] > asn_ub) goto SBOXOUTNOZERO_FW;

					if (ASNFWLBOver[Rnum - nxt_s.rnum][tmp_aspIndex]) {
						BASN_PC_FW[nxtnxt_s.rnum][0] = (ASNFWLB[Rnum - nxt_s.rnum][tmp_aspIndex] - nxtnxt_s.nr_sbx_num * 2) * weight[1];
						BASN_PC_FW[nxtnxt_s.rnum][0] = (BASN_PC_FW[nxtnxt_s.rnum][0] > LB[Rnum - nxtnxt_s.rnum - 1][FWASN]) ? BASN_PC_FW[nxtnxt_s.rnum][0] : LB[Rnum - nxtnxt_s.rnum - 1][FWASN];							
						FWRound_i(nxtnxt_s, TMPX[nxt_s.rnum]);
					}
					else if (!ASNFWLBOver[Rnum - nxt_s.rnum][tmp_aspIndex]) {
						FindBASN = false; BASN = asn_ub;
						FWRound_i_ASN(GenStateRI_ASN(nxtnxt_s), TMPX[nxt_s.rnum]);
						if (FindBASN) {
							ASNFWLB[Rnum - nxt_s.rnum][tmp_aspIndex] = BASN; ASNFWLBOver[Rnum - nxt_s.rnum][tmp_aspIndex] = true;							
							BASN_PC_FW[nxtnxt_s.rnum][0] = (BASN - nxtnxt_s.nr_sbx_num * 2) * weight[1];
							BASN_PC_FW[nxtnxt_s.rnum][0] = (BASN_PC_FW[nxtnxt_s.rnum][0] > LB[Rnum - nxtnxt_s.rnum - 1][FWASN]) ? BASN_PC_FW[nxtnxt_s.rnum][0] : LB[Rnum - nxtnxt_s.rnum - 1][FWASN];								
							FWRound_i(nxtnxt_s, TMPX[nxt_s.rnum]);

						}
						else ASNFWLB[Rnum - nxt_s.rnum][tmp_aspIndex] = BASN + 1;
					}
				}
			}
		}
		else if (s.j != s.sbx_num) {	
			if (!FindBn) BASN = ((Bn - (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
			else BASN = ((Bn - (s.W + s.w + DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
			auto itor_ASN_j = ASPandValueMapLB_ASN_FW_j.find(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)));
			bool searchasn_tag = true; int asn_lb = 0;
			if (itor_ASN_j != ASPandValueMapLB_ASN_FW_j.end()) {
				asn_lb = itor_ASN_j->second.ReturnLB(Rnum - s.rnum + 1, FWASN);
				if (itor_ASN_j->second.RdLBOver[Rnum - s.rnum]) searchasn_tag = false; 
			}
			if (BASN < asn_lb) goto SBOXOUTNOZERO_FW;

			if (searchasn_tag) {
				FindBASN = false;
				FWRound_i_ASN(GenStateRI_j_ASN(s, sbx_nr_num), sbx_out);
				if (itor_ASN_j != ASPandValueMapLB_ASN_FW_j.end()) {
					if (FindBASN) { itor_ASN_j->second.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN); itor_ASN_j->second.RdLBOver[Rnum - s.rnum] = true; }
					else itor_ASN_j->second.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(Rnum - s.rnum + 1, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN + 1); else newValueLB_ASN.RdLBOver[Rnum - s.rnum] = true;
					ASPandValueMapLB_ASN_FW_j.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)), newValueLB_ASN));
				}
				if (!FindBASN) goto SBOXOUTNOZERO_FW;
			}
			else BASN = asn_lb;

			BASN_PC_FW[s.rnum][s.j + 1] = (BASN - s.sbx_num - s.nr_sbx_num - 1) * weight[1];
			BASN_PC_FW[s.rnum][s.j + 1] = (BASN_PC_FW[s.rnum][s.j + 1] > LB[Rnum - s.rnum - 1][FWASN]) ? BASN_PC_FW[s.rnum][s.j + 1] : LB[Rnum - s.rnum - 1][FWASN];
			FWRound_i(UpdateStateRoundI_j(s, DDTorLATMinusMinW[s.sbx_a[s.j]][s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w), sbx_out); 
		}

	SBOXOUTNOZERO_FW:
		int tmp_BASN_PC_R = BASN_PC_FW[s.rnum][s.j] - weight[1];
		BASN_PC_FW[s.rnum][s.j] = (tmp_BASN_PC_R > LB[Rnum - s.rnum - 1][FWASN]) ? tmp_BASN_PC_R : LB[Rnum - s.rnum - 1][FWASN];
	}

	sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]] = 1;
	if (!s.sbx_tag[s.j]) { 
		sbx_nr_num = 1;
		TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = 1;		
	}
	int RecordMinAsn = 0; FindBASN = false;	
	if (!FindBn) BASN = ((Bn - (s.W + s.w + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
	else BASN = ((Bn - (s.W + s.w + s.nr_minw) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;

	if (s.j != s.sbx_num && !s.sbx_tag[s.j]) {
		tmp_searchOutput ^= (1 << s.j);
		auto itor_ASN_j = ASPandValueMapLB_ASN_FW_j.find(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)));
		bool searchasn_tag = true; int asn_lb = 0;
		if (itor_ASN_j != ASPandValueMapLB_ASN_FW_j.end()) {
			asn_lb = itor_ASN_j->second.ReturnLB(Rnum - s.rnum + 1, FWASN);
			if (itor_ASN_j->second.RdLBOver[Rnum - s.rnum]) searchasn_tag = false; 
		}
		if (BASN >= asn_lb) {
			if (searchasn_tag) {
				FWRound_i_ASN(GenStateRI_j_ASN(s, sbx_nr_num), sbx_out);
				if (itor_ASN_j != ASPandValueMapLB_ASN_FW_j.end()) {
					if (FindBASN) { itor_ASN_j->second.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN); itor_ASN_j->second.RdLBOver[Rnum - s.rnum] = true; }
					else itor_ASN_j->second.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(Rnum - s.rnum + 1, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(Rnum - s.rnum + 1, BASN + 1); else newValueLB_ASN.RdLBOver[Rnum - s.rnum] = true;
					ASPandValueMapLB_ASN_FW_j.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(tmp_searchnum, tmp_searchOutput)), newValueLB_ASN));
				}
			}
			else {
				FindBASN = true; BASN = asn_lb;
			}
		}		
	}
	else if (s.j != s.sbx_num) {
		FindBASN = true; BASN = BASN_PC_FW[s.rnum][s.j] / weight[1] + s.sbx_num + 1 + s.nr_sbx_num;
	}
	else if (s.rnum + 1 != Rnum) { 
		Trail_FW[s.rnum + 1] = sbx_out;
#if(TYPE)
		TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(sbx_out, 48), _mm_srli_epi64(sbx_out, 16));
#else
		TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(sbx_out, 16), _mm_srli_epi64(sbx_out, 48));
#endif
		Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
		BASN -= (s.sbx_num + 1);

		tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail_FW[s.rnum + 1], Mask));		
		auto itor_ASN = ASPandValueMapLB_ASN_FW.find(make_pair(tmp_asp1, tmp_asp2));
		int asn_lb = 0; int searchasn_tag = true;
		if (itor_ASN != ASPandValueMapLB_ASN_FW.end()) {
			asn_lb = itor_ASN->second.ReturnLB(Rnum - s.rnum, FWASN);
			if (itor_ASN->second.RdLBOver[Rnum - s.rnum - 1]) searchasn_tag = false;
		} 

		if (BASN >= asn_lb) {
			if (searchasn_tag) {
				FWRound_i_ASN(GenStateRI_ASN_FW(s), TMPX[s.rnum]);
				if (itor_ASN != ASPandValueMapLB_ASN_FW.end()) {
					if (FindBASN) {
						itor_ASN->second.UpdateOrInsertLB(Rnum - s.rnum, BASN); itor_ASN->second.RdLBOver[Rnum - s.rnum - 1] = true;
					}
					else itor_ASN->second.UpdateOrInsertLB(Rnum - s.rnum, BASN + 1);
				}
				else {
					RecordLBForValue_ASN newValueLB_ASN(Rnum - s.rnum, BASN);
					if (!FindBASN) newValueLB_ASN.UpdateOrInsertLB(Rnum - s.rnum, BASN + 1);
					else newValueLB_ASN.RdLBOver[Rnum - s.rnum - 1] = true;
					ASPandValueMapLB_ASN_FW.insert(make_pair(make_pair(tmp_asp1, tmp_asp2), newValueLB_ASN));
				}
			}
			else {
				FindBASN = true; BASN = asn_lb;
			}
		}		
	}
	else FindBASN = true; 
	
	if (FindBASN) RecordMinAsn = BASN;
	else {
		TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
		return;
	}
	int AllowMaxASN = 0;
	if (s.j != s.sbx_num && !s.sbx_tag[s.j]) {
		BASN_PC_FW[s.rnum][s.j + 1] = (RecordMinAsn - s.sbx_num - s.nr_sbx_num - sbx_nr_num - 1) * weight[1];
		BASN_PC_FW[s.rnum][s.j + 1] = (BASN_PC_FW[s.rnum][s.j + 1] > LB[Rnum - s.rnum - 1][FWASN]) ? BASN_PC_FW[s.rnum][s.j + 1] : LB[Rnum - s.rnum - 1][FWASN];
	}
	else if (s.j != s.sbx_num) {
		BASN_PC_FW[s.rnum][s.j + 1] = BASN_PC_FW[s.rnum][s.j];
	}

	for (int i = 0; i < SBox_SIZE; i++) {
		if ((!FindBn && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + nr_sbx_w + s.nr_minw + BASN_PC_FW[s.rnum][s.j] > Bn))
			|| (FindBn && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + nr_sbx_w + s.nr_minw + BASN_PC_FW[s.rnum][s.j] >= Bn))) break;
		sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]] = (FWWeightOrderV[s.sbx_a[s.j]][s.sbx_in[s.j]][i] ^ tmp_sbxout);
		if (!sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]]) continue; 
		sbx_nr_w = FWWeightMinandMax[FWSBoxPermutation[s.sbx_a[s.j]]][sbx_out.m128i_u8[FWSBoxPermutation[s.sbx_a[s.j]]]][0]; 
		if (s.sbx_tag[s.j]) sbx_nr_w -= weight[1];  

		if (((sbx_nr_num && sbx_nr_w != weight[1]) || (s.sbx_tag[s.j] && sbx_nr_w))
			&& ((!FindBn && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w + BASN_PC_FW[s.rnum][s.j] > Bn))
				|| (FindBn && (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w + BASN_PC_FW[s.rnum][s.j] >= Bn))))  continue; 

		if (s.j == s.sbx_num && s.rnum + 1 == Rnum) {
			Trail_FW[s.rnum + 1] = sbx_out;
			STATE nxt_s = FWUpdateStateRoundN(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w);
			FWRound_n(nxt_s);
		}
		else if (s.j == s.sbx_num) {
			Trail_FW[s.rnum + 1] = sbx_out;
#if(TYPE)
			TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 48), _mm_srli_epi64(Trail_FW[s.rnum + 1], 16));
#else
			TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 1], 16), _mm_srli_epi64(Trail_FW[s.rnum + 1], 48));
#endif
			Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 

			STATE nxt_s = FWUpdateStateRoundI(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w);
			if ((!FindBn && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[Rnum - nxt_s.rnum - 1][FWASN] > Bn))
				|| (FindBn && (nxt_s.W + nxt_s.w + nxt_s.nr_minw + LB[Rnum - nxt_s.rnum - 1][FWASN] >= Bn))) continue;

			if (!FindBn) AllowMaxASN = ((Bn - nxt_s.W - nxt_s.w - nxt_s.nr_minw) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
			else AllowMaxASN = ((Bn - nxt_s.W - nxt_s.w - nxt_s.nr_minw - 1) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
			if (AllowMaxASN < RecordMinAsn) continue;

			BASN_PC_FW[nxt_s.rnum][0] = (RecordMinAsn - nxt_s.sbx_num - 1 - nxt_s.nr_sbx_num) * weight[1];
			BASN_PC_FW[nxt_s.rnum][0] = (BASN_PC_FW[nxt_s.rnum][0] > LB[Rnum - nxt_s.rnum - 1][FWASN]) ? BASN_PC_FW[nxt_s.rnum][0] : LB[Rnum - nxt_s.rnum - 1][FWASN];
			FWRound_i(nxt_s, TMPX[s.rnum]);
					
		}
		else {
			if (!FindBn) AllowMaxASN = ((Bn - (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w)) / weight[1]) + s.sbx_num + s.nr_sbx_num + sbx_nr_num + 1;
			else AllowMaxASN = ((Bn - (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i] + s.nr_minw + sbx_nr_w) - 1) / weight[1]) + s.sbx_num + s.nr_sbx_num + sbx_nr_num + 1;
			if (AllowMaxASN < RecordMinAsn) break;
			FWRound_i(UpdateStateRoundI_j(s, FWWeightOrderWMinusMin[s.sbx_a[s.j]][s.sbx_in[s.j]][i], sbx_nr_w), sbx_out);		
		}
	}
	TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[s.sbx_a[s.j]]]] = tmp2R;
	return;
}


void FWRound_2_GenInput(STATE s, __m128i sbx_in) {
	for (int i = 1; i < SBox_SIZE; i++) {
		if ((!FindBn && (s.W + s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_FW[0][0] > Bn))
			|| (FindBn && (s.W + s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_FW[0][0] >= Bn))) break;
		s.sbx_in[s.j] = IRFWMinV[s.sbx_a[s.j]][i]; sbx_in.m128i_u8[s.sbx_a[s.j]] = IRFWMinV[s.sbx_a[s.j]][i];			
		if (s.j == s.sbx_num) {
			Trail_FW[s.rnum] = sbx_in;
#if(TYPE)
			TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum], 48), _mm_srli_epi64(Trail_FW[s.rnum], 16));
#else
			TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum], 16), _mm_srli_epi64(Trail_FW[s.rnum], 48));
#endif		
			Extern2RMask[s.rnum] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum], Mask));

			STATE nxt_s = UpdateStateFWR2andBWInput(s, IRFWMinW[s.sbx_a[s.j]][i]);
			BASN_PC_FW[nxt_s.rnum][0] = BASN_PC_FW[0][0];
			FWRound_i(nxt_s, TMPX[1]);
		}
		else FWRound_2_GenInput(UpdateStateRoundI_j(s, IRFWMinW[s.sbx_a[s.j]][i], 0), sbx_in);
	}
}


void FWRound_1(STATE s, __m128i sbx_in) {
	int sbx_r3_w = 0;
	for (int i = 1; i < SBox_SIZE; i++) {
		if ((!FindBn && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_FW[1][0] > Bn))
			|| (FindBn && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + BASN_PC_FW[1][0] >= Bn))) break;
		sbx_in.m128i_u8[s.sbx_a[s.j]] = IRFWMinV[s.sbx_a[s.j]][i];
		if (s.sbx_tag[s.j]) {
			sbx_r3_w = FWWeightMinandMax[FWSBoxROT[s.sbx_a[s.j]]][sbx_in.m128i_u8[s.sbx_a[s.j]]][0] - weight[1];
			if ((!FindBn && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + sbx_r3_w + BASN_PC_FW[1][0] > Bn))
				|| (FindBn && (s.w + IRFWMinW[s.sbx_a[s.j]][i] + s.nr_minw + sbx_r3_w + BASN_PC_FW[1][0] >= Bn))) continue;
		}		

		if (s.j == s.sbx_num) {
			Trail_FW[1] = sbx_in;
#if(TYPE)
			TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 48), _mm_srli_epi64(Trail_FW[1], 16));
#else
			TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 16), _mm_srli_epi64(Trail_FW[1], 48));
#endif			
			STATE nxt_s = UpdateStateR2Input(R2STATE, s.w + IRFWMinW[s.sbx_a[s.j]][i], s.nr_minw + sbx_r3_w);
			BASN_PC_FW[0][0] = BASN_PC_FW[1][0];
			FWRound_2_GenInput(nxt_s, TMPX[1]);

		}
		else FWRound_1(UpdateStateRoundI_j(s, IRFWMinW[s.sbx_a[s.j]][i], sbx_r3_w), sbx_in);
	}
	return;
}


void Round_NP(STATE s, __m128i sbx_in, int NPPlusOneMinW) {	
	for (int i = 1; i < SBox_SIZE; i++) { 
		if ((!FindBn && (s.W + s.w + IRFWMinW[s.sbx_a[s.j]][i] + NPPlusOneMinW + FWWeightMinandMax[FWSBoxROT[s.sbx_a[s.j]]][IRFWMinV[s.sbx_a[s.j]][i]][0] - weight[1] + BASN_PC_BW[s.rnum + 1][0] > Bn))
			|| (FindBn && (s.W + s.w + IRFWMinW[s.sbx_a[s.j]][i] + NPPlusOneMinW + FWWeightMinandMax[FWSBoxROT[s.sbx_a[s.j]]][IRFWMinV[s.sbx_a[s.j]][i]][0] - weight[1] + BASN_PC_BW[s.rnum + 1][0] >= Bn))) break;
		s.sbx_in[s.j] = IRFWMinV[s.sbx_a[s.j]][i]; sbx_in.m128i_u8[s.sbx_a[s.j]] = IRFWMinV[s.sbx_a[s.j]][i];	
		if (s.j == s.sbx_num) {			
			Trail_BW[s.rnum] = sbx_in; 
#if(TYPE)
			Trail_FW[s.rnum + 2] = _mm_xor_si128(_mm_slli_epi64(sbx_in, 48), _mm_srli_epi64(sbx_in, 16));
#else
			Trail_FW[s.rnum + 2] = _mm_xor_si128(_mm_slli_epi64(sbx_in, 16), _mm_srli_epi64(sbx_in, 48));
#endif
			BWSearchOver = false; BWBn = Bn - s.W - NPPlusOneMinW - FWWeightMinandMax[FWSBoxROT[s.sbx_a[s.j]]][IRFWMinV[s.sbx_a[s.j]][i]][0] + weight[1];
			
			STATE nxt_s_bw = UpdateStateRoundNP_BW(s, IRFWMinW[s.sbx_a[s.j]][i]);
			if (s.rnum == 1) BWRound_n(nxt_s_bw);
			else {
#if(TYPE)
				TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(sbx_in, 16), _mm_srli_epi64(sbx_in, 48));
#else
				TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(sbx_in, 48), _mm_srli_epi64(sbx_in, 16));
#endif	
				Extern2RMask[s.rnum] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum], Mask)); 

				if ((!FindBn && (nxt_s_bw.W + nxt_s_bw.w + nxt_s_bw.nr_minw + LB[s.rnum - 2][1] > BWBn))
					|| (FindBn && (nxt_s_bw.W + nxt_s_bw.w + nxt_s_bw.nr_minw + LB[s.rnum - 2][1] >= BWBn))) continue;

				BASN_PC_BW[nxt_s_bw.rnum][0] = (ASNBWLB[s.rnum][ASP_INDEX] - nxt_s_bw.nr_sbx_num * 2) * weight[1];
				BASN_PC_BW[nxt_s_bw.rnum][0] = (BASN_PC_BW[nxt_s_bw.rnum][0] > LB[nxt_s_bw.rnum - 2][1]) ? BASN_PC_BW[nxt_s_bw.rnum][0] : LB[nxt_s_bw.rnum - 2][1];
				BWRound_i(nxt_s_bw, TMPX[s.rnum + 1]);
			}

			ASP_Value = (ASP_Value < BWBn) ? ASP_Value : BWBn;

			if (!BWSearchOver) continue;

#if(TYPE)
			TMPX[s.rnum + 2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 2], 48), _mm_srli_epi64(Trail_FW[s.rnum + 2], 16));
#else
			TMPX[s.rnum + 2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[s.rnum + 2], 16), _mm_srli_epi64(Trail_FW[s.rnum + 2], 48));
#endif
			Extern2RMask[s.rnum + 2] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 2], Mask)); 

			STATE nxt_s_fw = UpdateStateRoundNP_FW(BWBn, NPRnum + 1);
			if (NPRnum + 1 == Rnum) {
				if ((!FindBn && (nxt_s_fw.W + nxt_s_fw.w <= Bn))
					|| (FindBn && (nxt_s_fw.W + nxt_s_fw.w < Bn))) FWRound_n(nxt_s_fw);
			}
			else if ((!FindBn && (nxt_s_fw.W + nxt_s_fw.w + nxt_s_fw.nr_minw + LB[Rnum - nxt_s_fw.rnum - 1][0] <= Bn))
				|| (FindBn && (nxt_s_fw.W + nxt_s_fw.w + nxt_s_fw.nr_minw + LB[Rnum - nxt_s_fw.rnum - 1][0] < Bn))) {

				
				BASN_PC_FW[nxt_s_fw.rnum][0] = (ASNFWLB[Rnum - NPRnum][ASP_INDEX] - nxt_s_fw.nr_sbx_num * 2) * weight[1];
				BASN_PC_FW[nxt_s_fw.rnum][0] = (BASN_PC_FW[nxt_s_fw.rnum][0] > LB[Rnum - nxt_s_fw.rnum - 1][0]) ? BASN_PC_FW[nxt_s_fw.rnum][0] : LB[Rnum - nxt_s_fw.rnum - 1][0];
				FWRound_i(nxt_s_fw, TMPX[s.rnum + 1]);
			}
		}
		else Round_NP(UpdateStateRoundNP_j(s, IRFWMinW[s.sbx_a[s.j]][i]), sbx_in, NPPlusOneMinW + FWWeightMinandMax[FWSBoxROT[s.sbx_a[s.j]]][IRFWMinV[s.sbx_a[s.j]][i]][0] - weight[1]);

	}
	return;

}

void Round_12_GenASNPattern_ASN(int index1, int asn1, int index2, int asn2, u8 arr1_sbx[], u8 arr2_sbx[]) {
	int lb = 0;
	if (index1 == asn1) {
		if (index2) lb = arr1_sbx[index2 - 1] + 1;
		for (int i = lb; i < (SBox_NUM + index2 + 1 - asn2); i++) {
			arr2_sbx[index2] = i;
			if (index2 + 1 == asn2) {
				STATE s_asn = GenR2OrR3State_ASN(asn1, asn2, arr1_sbx, arr2_sbx);
				FWRound_i_ASN(s_asn, TMPX[s_asn.rnum - 1]);
			}
			else Round_12_GenASNPattern_ASN(index1, asn1, index2 + 1, asn2, arr1_sbx, arr2_sbx);
		}
	}
	else {
		if (index1) lb = arr1_sbx[index1 - 1] + 1;
		for (int i = lb; i < (SBox_NUM + index1 + 1 - asn1); i++) {
			arr1_sbx[index1] = i;
			if (!asn2 && index1 + 1 == asn1) {
				STATE s_asn = GenR2OrR3State_ASN(asn1, asn2, arr1_sbx, arr2_sbx);
				if (Rnum == 3 && s_asn.rnum == 3) {
					if ((FindBASN && s_asn.W + s_asn.w < BASN) || (!FindBASN && s_asn.W + s_asn.w <= BASN))
						FWRound_n_ASN(s_asn); 
				}
				else FWRound_i_ASN(s_asn, TMPX[s_asn.rnum - 1]);
			}
			else Round_12_GenASNPattern_ASN(index1 + 1, asn1, index2, asn2, arr1_sbx, arr2_sbx);
		}
	}
}

void Round_12_GenASNPattern(int index1, int asn1, int index2, int asn2, u8 arr1_sbx[], u8 arr2_sbx[]) {
	int lb = 0;
	if (index1 == asn1) {
		if (index2) lb = arr1_sbx[index2 - 1] + 1;
		for (int i = lb; i < (SBox_NUM + index2 + 1 - asn2); i++) {
			arr2_sbx[index2] = i;
			if (index2 + 1 == asn2) {
				FindBASN = false;
				if (!FindBn) BASN = (Bn / weight[1]);
				else BASN = ((Bn - 1) / weight[1]);
				STATE s_asn = GenR2OrR3State_ASN(asn1, asn2, arr1_sbx, arr2_sbx);
				FWRound_i_ASN(s_asn, TMPX[s_asn.rnum - 1]);
				if (!FindBASN) continue;  

				BASN_PC_FW[1][0] = (BASN - s_asn.W - s_asn.w - s_asn.nr_sbx_num) * weight[1];
				BASN_PC_FW[1][0] = (BASN_PC_FW[1][0] > LB[Rnum - 3][FWASN]) ? BASN_PC_FW[1][0] : LB[Rnum - 3][FWASN];

				STATE s_r1(1, 0); STATE s_r2(2, 0);
				GenR1R2StateForNa1(s_r1, s_r2, asn1, asn2, arr1_sbx, arr2_sbx);
				R2STATE = s_r2; 
				__m128i sbx_in = _mm_setzero_si128();
				if (s_r1.sbx_num + 1) {
					FWRound_1(s_r1, sbx_in);
				}
				else {
					s_r2 = UpdateStateR2Input(s_r2, 0, s_r1.nr_minw);
					BASN_PC_FW[0][0] = BASN_PC_FW[1][0];
					FWRound_2_GenInput(s_r2, sbx_in);
				}

			}
			else Round_12_GenASNPattern(index1, asn1, index2 + 1, asn2, arr1_sbx, arr2_sbx);
		}
	}
	else {
		if (index1) lb = arr1_sbx[index1 - 1] + 1;
		for (int i = lb; i < (SBox_NUM + index1 + 1 - asn1); i++) {
			arr1_sbx[index1] = i;
			if (!asn2 && index1 + 1 == asn1) {
				FindBASN = false;
				if (!FindBn) BASN = (Bn / weight[1]);
				else BASN = ((Bn - 1) / weight[1]);
				STATE s_asn = GenR2OrR3State_ASN(asn1, asn2, arr1_sbx, arr2_sbx);
				if (Rnum == 3 && s_asn.rnum == 3) {
					if (s_asn.W + s_asn.w <= BASN)
						FWRound_n_ASN(s_asn);
				}
				else FWRound_i_ASN(s_asn, TMPX[s_asn.rnum - 1]);
				if (!FindBASN) continue;  

				if (s_asn.rnum == 3) BASN_PC_FW[1][0] = (BASN - s_asn.W - s_asn.w) * weight[1];
				else BASN_PC_FW[1][0] = (BASN - s_asn.W - s_asn.w - s_asn.nr_sbx_num) * weight[1];
				BASN_PC_FW[1][0] = (BASN_PC_FW[1][0] > LB[Rnum - 3][FWASN]) ? BASN_PC_FW[1][0] : LB[Rnum - 3][FWASN];

				STATE s_r1(1, 0); STATE s_r2(2, 0);
				GenR1R2StateForNa1(s_r1, s_r2, asn1, asn2, arr1_sbx, arr2_sbx);
				R2STATE = s_r2; 
				__m128i sbx_in = _mm_setzero_si128();

				FWRound_1(s_r1, sbx_in); 	
			}
			else Round_12_GenASNPattern(index1 + 1, asn1, index2, asn2, arr1_sbx, arr2_sbx);
		}
	}
}

void RoundASN() {
	int asn_ub = BASN - ASNLB[Rnum - 2][FWASN];
	if (FWASN && asn_ub < 2) return; 
	u8 arr1_sbx[SBox_NUM] = { 0 }; u8 arr2_sbx[SBox_NUM] = { 0 };	
	int asn_lb = 1, asn_r1_lb = 0, asn_r1_ub;
	if (FWASN) {
		asn_lb = 2; asn_r1_lb = 1; 
	}

	for (int i = asn_lb; i <= asn_ub; i++) {
		if (FWASN) asn_r1_ub = i - 1;
		else asn_r1_ub = i;
		for (int j = asn_r1_lb; j <= asn_r1_ub; j++) {
			memset(arr1_sbx, 0, sizeof(arr1_sbx));
			memset(arr2_sbx, 0, sizeof(arr2_sbx));
			Round_12_GenASNPattern_ASN(0, j, 0, i - j, arr1_sbx, arr2_sbx);
			if (FindBASN) {
				asn_ub = BASN - ASNLB[Rnum - 2][FWASN]; 
			}
		}	
	}
	return;
}

void RoundNA0() {
	FindBASN = false; BASN = Bn / weight[1]; FWASN = 0;
	RoundASN();
	if (FindBASN) ASNLB[Rnum][0] = BASN;
	else ASNLB[Rnum][0] = BASN + 1;
	if (!FindBASN || (!FindBn && LB[Rnum][0] > Bn) || (FindBn && LB[Rnum][0] >= Bn)) return;

	__m128i tmp_in = _mm_setzero_si128(); 

	NPRnum = Rnum; Trail_BW[Rnum] = tmp_in; TMPX[Rnum] = tmp_in; T_W_BW[Rnum - 1] = 0;
	for (ASP_INDEX = 0; ASP_INDEX < ASP_NUM; ASP_INDEX++) {
		if ((FindBn && (LBNA0[Rnum][NPRnum][ASP_INDEX] >= Bn)) || (!FindBn && (LBNA0[Rnum][NPRnum][ASP_INDEX] > Bn))) continue;
		FindBASN = false; BASN = Bn / weight[1];
		STATE s_asn = GenNRBWState_ASN(NPRnum - 1);
		BWRound_i_ASN(s_asn, TMPX[NPRnum]);

		if (!FindBASN) {
			ASNBWLB[Rnum - 1][ASP_INDEX] = BASN + 1;
			BWLB[Rnum - 1][ASP_INDEX] = (BASN + 1) * weight[1];
			LBNA0[Rnum][Rnum][ASP_INDEX] = (BASN + 1) * weight[1];
			continue;
		}

		ASNBWLB[Rnum - 1][ASP_INDEX] = BASN; ASNBWLBOver[Rnum - 1][ASP_INDEX] = true;
		int tmpBASN = Bn / weight[1]; if (FindBn) tmpBASN = (Bn - 1) / weight[1];
		if (tmpBASN < BASN) {
			BWLB[Rnum - 1][ASP_INDEX] = BASN * weight[1];
			LBNA0[Rnum][Rnum][ASP_INDEX] = BWLB[Rnum - 1][ASP_INDEX];
			continue;
		}

		FindSub = false; BWBn = Bn;
		STATE state = GenStateNRBWForNa0(NPRnum - 1);
		BASN_PC_BW[0][0] = (BASN - s_asn.w - s_asn.nr_sbx_num) * weight[1];
		BASN_PC_BW[0][0] = (BASN_PC_BW[0][0] > LB[Rnum - 3][FWASN]) ? BASN_PC_BW[0][0] : LB[Rnum - 3][FWASN];

		BWRound_NP_GenInput(state, TMPX[Rnum]);

		if (FindSub) {
			FindNA0 = true; BWLB[Rnum - 1][ASP_INDEX] = Bn;
		}
		else if (FindBn) {
			BWLB[Rnum - 1][ASP_INDEX] = (Bn > ASNBWLB[Rnum - 1][ASP_INDEX] * weight[1]) ? Bn : (ASNBWLB[Rnum - 1][ASP_INDEX] * weight[1]);
		}
		else {
			BWLB[Rnum - 1][ASP_INDEX] = ((Bn + 1) > ASNBWLB[Rnum - 1][ASP_INDEX] * weight[1]) ? (Bn + 1) : (ASNBWLB[Rnum - 1][ASP_INDEX] * weight[1]);
		}
		LBNA0[Rnum][Rnum][ASP_INDEX] = BWLB[Rnum - 1][ASP_INDEX];
	}

	for (NPRnum = Rnum - 1; NPRnum > 1; NPRnum--) {
		Trail_FW[NPRnum] = tmp_in; Trail_BW[NPRnum] = tmp_in; TMPX[NPRnum] = tmp_in; T_W_FW[NPRnum - 1] = 0;
		for (ASP_INDEX = 0; ASP_INDEX < ASP_NUM; ASP_INDEX++) {
			if ((FindBn && (LBNA0[Rnum][NPRnum][ASP_INDEX] >= Bn)) || (!FindBn && (LBNA0[Rnum][NPRnum][ASP_INDEX] > Bn))) continue;
			FindBASN = false; BASN = Bn / weight[1] - ASNFWLB[Rnum - NPRnum][ASP_INDEX];
			if (!ASNBWLBOver[NPRnum - 1][ASP_INDEX] && ASNBWLB[NPRnum - 1][ASP_INDEX] <= BASN) { 
				STATE s_asn = GenNRBWState_ASN(NPRnum - 1);
				BWRound_i_ASN(s_asn, TMPX[NPRnum]);
				UpdateASNBWLB();
			}
			else if (ASNBWLB[NPRnum - 1][ASP_INDEX] <= BASN) {
				FindBASN = true; BASN = ASNBWLB[NPRnum - 1][ASP_INDEX];
			}

			if (!FindBASN) {
				LBNA0[Rnum][NPRnum][ASP_INDEX] = (BASN + ASNFWLB[Rnum - NPRnum][ASP_INDEX] + 1) * weight[1];
				continue;
			}


			FindBASN = false; BASN = Bn / weight[1] - BASN;
			if (!ASNFWLBOver[Rnum - NPRnum][ASP_INDEX] && ASNFWLB[Rnum - NPRnum][ASP_INDEX] <= BASN) {
				STATE s_asn = GenNRFWState_ASN(NPRnum + 1);
				FWRound_i_ASN(s_asn, tmp_in);
				UpdateASNFWLB();
			}
			else if(ASNFWLB[Rnum - NPRnum][ASP_INDEX] <= BASN) {
				FindBASN = true;
				BASN = ASNFWLB[Rnum - NPRnum][ASP_INDEX];
			}

			if (!FindBASN) { 
				LBNA0[Rnum][NPRnum][ASP_INDEX] = (BASN + ASNBWLB[NPRnum - 1][ASP_INDEX] + 1) * weight[1];
				continue;
			}

			int tmpBASN = Bn / weight[1]; if (FindBn) tmpBASN = (Bn - 1) / weight[1]; BASN += ASNBWLB[NPRnum - 1][ASP_INDEX];
			if (tmpBASN < BASN) {
				LBNA0[Rnum][NPRnum][ASP_INDEX] = BASN * weight[1]; continue;				
			}


			FindSub = false; ASP_Value = Bn; 	
			STATE state = GenStateNRForNa0(NPRnum - 1); int NPPlus1MinW = state.w;
			state.W = (LB[Rnum - state.rnum - 2][0] > (FWLB[Rnum - NPRnum][ASP_INDEX] - ASPInfo[ASP_INDEX] * weight[1])) ? 
				LB[Rnum - state.rnum - 2][0] : (FWLB[Rnum - NPRnum][ASP_INDEX] - ASPInfo[ASP_INDEX] * weight[1]);
			BASN_PC_BW[NPRnum][0] = (ASNBWLB[NPRnum - 1][ASP_INDEX] - state.sbx_num - 1) * weight[1];
			BASN_PC_BW[NPRnum][0] = (BASN_PC_BW[NPRnum][0] > LB[state.rnum - 1][FWASN]) ? BASN_PC_BW[NPRnum][0] : LB[state.rnum - 1][FWASN];
			Round_NP(state, tmp_in, NPPlus1MinW);
			
			if (BWLB[NPRnum - 1][ASP_INDEX] < ASP_Value) UpdateBWLBandLBNA0(ASP_Value);
			if (FindSub) {
				FindNA0 = true; LBNA0[Rnum][NPRnum][ASP_INDEX] = Bn;		
			}
			else if (FindBn) {
				LBNA0[Rnum][NPRnum][ASP_INDEX] = (Bn > ((ASNBWLB[NPRnum - 1][ASP_INDEX] + ASNFWLB[Rnum - NPRnum][ASP_INDEX]) * weight[1])) ?
					Bn : ((ASNBWLB[NPRnum - 1][ASP_INDEX] + ASNFWLB[Rnum - NPRnum][ASP_INDEX]) * weight[1]);
			}
			else {
				LBNA0[Rnum][NPRnum][ASP_INDEX] = ((Bn + 1) > ((ASNBWLB[NPRnum - 1][ASP_INDEX] + ASNFWLB[Rnum - NPRnum][ASP_INDEX]) * weight[1])) ?
					(Bn + 1) : ((ASNBWLB[NPRnum - 1][ASP_INDEX] + ASNFWLB[Rnum - NPRnum][ASP_INDEX]) * weight[1]);
			}
		}
	}	


	Trail_FW[1] = tmp_in; TMPX[1] = tmp_in; T_W_FW[0] = 0; NPRnum = 1;	
	for (ASP_INDEX = 0; ASP_INDEX < ASP_NUM; ASP_INDEX++) {
		if ((FindBn && (LBNA0[Rnum][1][ASP_INDEX] >= Bn)) || (!FindBn && (LBNA0[Rnum][1][ASP_INDEX] > Bn))) continue;
		FindBASN = false;
		if (FindBn)	BASN = (Bn - 1) / weight[1]; else BASN = Bn / weight[1];		
		STATE s_asn = GenNRFWState_ASN(2);
		FWRound_i_ASN(s_asn, tmp_in);

		if (!FindBASN) {
			ASNFWLB[Rnum - 1][ASP_INDEX] = BASN + 1;
			FWLB[Rnum - 1][ASP_INDEX] = (BASN + 1) * weight[1];
			LBNA0[Rnum][1][ASP_INDEX] = (BASN + 1) * weight[1];
			continue;
		}

		ASNFWLB[Rnum - 1][ASP_INDEX] = BASN; ASNFWLBOver[Rnum - 1][ASP_INDEX] = true;
		int tmpBASN = Bn / weight[1];
		if (FindBn) tmpBASN = (Bn - 1) / weight[1];
		if (tmpBASN < BASN) {
			FWLB[Rnum - 1][ASP_INDEX] = BASN * weight[1];
			LBNA0[Rnum][1][ASP_INDEX] = FWLB[Rnum - 1][ASP_INDEX];
			continue;
		}	
		FindSub = false;
		STATE state = s_asn; state.w = s_asn.w * weight[1]; state.nr_minw = s_asn.nr_sbx_num * weight[1];
		TMPX2R[state.rnum] = TMPX2R_ASN[state.rnum];
		BASN_PC_FW[0][0] = (BASN - s_asn.w - s_asn.nr_sbx_num) * weight[1];
		BASN_PC_FW[0][0] = (BASN_PC_FW[0][0] > LB[Rnum - 3][FWASN]) ? BASN_PC_FW[0][0] : LB[Rnum - 3][FWASN];
		FWRound_2_GenInput(state, TMPX[1]);
		if (FindSub) {
			FindNA0 = true; FWLB[Rnum - 1][ASP_INDEX] = Bn;
		}
		else if (FindBn) {
			FWLB[Rnum - 1][ASP_INDEX] = (Bn > ASNFWLB[Rnum - 1][ASP_INDEX] * weight[1]) ? Bn : (ASNFWLB[Rnum - 1][ASP_INDEX] * weight[1]);
		}
		else {
			FWLB[Rnum - 1][ASP_INDEX] = ((Bn + 1) > ASNFWLB[Rnum - 1][ASP_INDEX] * weight[1]) ? (Bn + 1) : (ASNFWLB[Rnum - 1][ASP_INDEX] * weight[1]);
		}
		LBNA0[Rnum][1][ASP_INDEX] = FWLB[Rnum - 1][ASP_INDEX];
	}

	if (FindNA0 && Rnum < PreRange) {
		memcpy(TMPNA0_TRAIL, BestTrail, (Rnum + 1) * STATE_LEN);
		memcpy(TMPNA0_W, Best_W, Rnum * sizeof(int));
	}


	if (FindBn) LB[Rnum][0] = Bn; 
	else {
		LB[Rnum][0] = ((ASNLB[Rnum][0] * weight[1]) > (Bn + 1)) ? (ASNLB[Rnum][0] * weight[1]) : (Bn + 1);
	}
}

void RoundNA1() {
	FindBASN = false;
	if ((!FindBn && LB[Rnum][1] > Bn) || (FindBn && LB[Rnum][1] >= Bn)) return;
	BASN = Bn / weight[1]; FWASN = 1;
	RoundASN();

	if (PreSearchTag && FindBASN) {
		ASNLB[Rnum][1] = BASN;
		LB[Rnum][1] = BASN * weight[1];
		return;
	}
	else if (FindBASN) {
		ASNLB[Rnum][1] = BASN;
		int tmpBASN = Bn / weight[1]; if (FindBn) tmpBASN = (Bn - 1) / weight[1];		
		if (tmpBASN >= BASN) {
			FindSub = false;	
			int asn_ub = (Bn / weight[1]) - ASNLB[Rnum - 2][FWASN];
			if (asn_ub >= 2) {
				u8 arr1_sbx[SBox_NUM] = { 0 }; u8 arr2_sbx[SBox_NUM] = { 0 };
				for (int i = 2; i <= asn_ub; i++) {
					for (int j = 1; j <= asn_ub - 1; j++) {
						memset(arr1_sbx, 0, sizeof(arr1_sbx));
						memset(arr2_sbx, 0, sizeof(arr2_sbx));
						Round_12_GenASNPattern(0, j, 0, i - j, arr1_sbx, arr2_sbx);
						if (FindBn) asn_ub = ((Bn - 1) / weight[1]) - ASNLB[Rnum - 2][FWASN];
					}
				}			
			}			
		}
		if (FindSub) {
			FindNA1 = true;
			LB[Rnum][1] = Bn;
		}
		else if (FindBn) {
			LB[Rnum][1] = (Bn > (BASN * weight[1])) ? Bn : (BASN * weight[1]);
		}
		else LB[Rnum][1] = ((Bn + 1) > (BASN * weight[1])) ? (Bn + 1) : (BASN * weight[1]);

	}
	else {
		ASNLB[Rnum][1] = BASN + 1;
		LB[Rnum][1] = (BASN + 1) * weight[1];
	}

	return;
}

void InitialLB() {
	for (int i = 0; i < ASP_NUM; i++) {
		ASNBWLB[Rnum - 1][i] = ASNBWLB[Rnum - 2][i] + ASNLB[1][1];
		ASNFWLB[Rnum - 1][i] = ASNFWLB[Rnum - 2][i] + ASNLB[1][0];
		BWLB[Rnum - 1][i] = BWLB[Rnum - 2][i] + LB[1][1];
		FWLB[Rnum - 1][i] = FWLB[Rnum - 2][i] + LB[1][0];
		for (int r = 2; r < Rnum - 1; r++) {
			ASNBWLB[Rnum - 1][i] = (ASNBWLB[Rnum - 1][i] > ASNBWLB[Rnum - 1 - r][i] + ASNLB[r][1]) ? ASNBWLB[Rnum - 1][i] : ASNBWLB[Rnum - 1 - r][i] + ASNLB[r][1];
			ASNFWLB[Rnum - 1][i] = (ASNFWLB[Rnum - 1][i] > ASNFWLB[Rnum - 1 - r][i] + ASNLB[r][0]) ? ASNFWLB[Rnum - 1][i] : ASNFWLB[Rnum - 1 - r][i] + ASNLB[r][0];
			BWLB[Rnum - 1][i] = (BWLB[Rnum - 1][i] > BWLB[Rnum - 1 - r][i] + LB[r][1]) ? BWLB[Rnum - 1][i] : BWLB[Rnum - 1 - r][i] + LB[r][1];
			FWLB[Rnum - 1][i] = (FWLB[Rnum - 1][i] > FWLB[Rnum - 1 - r][i] + LB[r][0]) ? FWLB[Rnum - 1][i] : FWLB[Rnum - 1 - r][i] + LB[r][0];
		}
	}
	bool tag = true;
	for (int np = 1; np <= Rnum; np++) {
		for (int i = 0; i < ASP_NUM; i++) {
			LBNA0[Rnum][np][i] = BWLB[np - 1][i] + FWLB[Rnum - np][i];
			for (int r1 = 1; r1 <= np - 1; r1++) {
				LBNA0[Rnum][np][i] = (LBNA0[Rnum][np][i] > (LB[r1][1] + LBNA0[Rnum - r1][np - r1][i])) ? LBNA0[Rnum][np][i] : LB[r1][1] + LBNA0[Rnum - r1][np - r1][i];
			}
			for (int r2 = 1; r2 <= Rnum - np; r2++) {
				LBNA0[Rnum][np][i] = (LBNA0[Rnum][np][i] > (LB[r2][0] + LBNA0[Rnum - r2][np][i])) ? LBNA0[Rnum][np][i] : LB[r2][0] + LBNA0[Rnum - r2][np][i];
			}
			if (tag) { LB[Rnum][0] = LBNA0[Rnum][np][i]; tag = false; }
			else LB[Rnum][0] = (LB[Rnum][0] < LBNA0[Rnum][np][i]) ? LB[Rnum][0] : LBNA0[Rnum][np][i];
		}
	}
	ASNLB[Rnum][0] = ASNLB[Rnum - 1][0] + ASNLB[1][0];
	ASNLB[Rnum][1] = ASNLB[Rnum - 1][1] + ASNLB[1][1];
	LB[Rnum][1] = LB[Rnum - 1][1] + LB[1][1];
	for (int r = 2; r <= Rnum / 2; r++) {
		ASNLB[Rnum][0] = (ASNLB[Rnum][0] > (ASNLB[Rnum - r][0] + ASNLB[r][0])) ? ASNLB[Rnum][0] : ASNLB[Rnum - r][0] + ASNLB[r][0];
		ASNLB[Rnum][1] = (ASNLB[Rnum][1] > (ASNLB[Rnum - r][1] + ASNLB[r][1])) ? ASNLB[Rnum][1] : ASNLB[Rnum - r][1] + ASNLB[r][1];
		LB[Rnum][1] = (LB[Rnum][1] > (LB[Rnum - r][1] + LB[r][1])) ? LB[Rnum][1] : LB[Rnum - r][1] + LB[r][1];
	}
}

void GenBn(int Tag) {
	FWASN = 0; 
	bool fwTag = true;
	int recordWeight[2]; recordWeight[0] = Best_W[Rnum - 2]; recordWeight[1] = Best_W[Rnum - 3]; int recordFWExt;
	STATE s_fw = GenBn_STATE_FW();
	if (Tag || (!Tag && Best_W[Rnum - 2])) {
		fwTag = false;
		if (s_fw.rnum == Rnum - 1) ExternRound = 1;
		else ExternRound = 2;
		recordFWExt = ExternRound;
		BASN_PC_FW[s_fw.rnum][0] = LB[Rnum - s_fw.rnum - 1][FWASN];
		FWRound_i(s_fw, TMPX[s_fw.rnum - 1]);
	}
	
	NPRnum = Rnum; BWBn = Bn;
	STATE s_bw = GenBn_STATE_BW(BestB[Rnum - 1]); int recordBn = Bn;
	if (Tag || (!Tag && (Best_W[0] || fwTag))) {
		if (s_bw.rnum == 2) ExternRound = 1;
		else ExternRound = 2;
		BASN_PC_BW[s_bw.rnum][0] = LB[s_bw.rnum - 2][1];
		BWRound_i(s_bw, TMPX[s_bw.rnum + 1]);
		if (Bn != recordBn && !fwTag) {
			Best_W[Rnum - 2] = recordWeight[0];
			if (recordFWExt == 2) { Best_W[Rnum - 3] = recordWeight[1]; }
		}
	}
}

void matsui() {
	GenTables();
	clock_t start, End;
#if(TYPE)
	string fileName = "LBlock_Linear.txt";
#else	
	string fileName = "LBlock_Diff.txt";
#endif

	stringstream message;
	message << "Pre-Seach Round: " << PreRange << endl;
	logToFile(fileName, message.str());

	PreSearchTag = false;
	BestB[1] = 0;
	BestB[2] = weight[1];
	initial_AllTrail();
	Bn = 0;
	double RecordTotalTime = 0;

	for (int i = 3; i <= RNUM; i++) {
		Rnum = i;
		FindNA0 = false; FindNA1 = false;
		if (i == 3) Bn = BestB[i - 1] + weight[1];

		cout << "Round NUM: " << dec << i << endl;
		message.str("");
		message << "RNUM_" << Rnum << " :\nBeginBn:" << Bn << endl;
		
		if (i == 3) {
			FindBn = false;
			start = clock();			
			while (!FindBn) {
				cout << "Bn: " << Bn << endl;
				RoundNA0();
				RoundNA1();
				if (!FindBn) Bn += weight[WeightLen - 2];
			}
			End = clock();
		}
		else {
			InitialLB();
			FindBn = true;
			start = clock();
			RoundNA0();
			RoundNA1();
			End = clock();
		}
		
		printf("Final Bn:%f\ntime: %fs, %fmin\n",(double) Bn, ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
		FileOutputTrail();
		message << "BestBn:" << Bn << "\nSearch Time: " << ((double)End - (double)start) / CLOCKS_PER_SEC << " s,  " << (((double)End - (double)start) / CLOCKS_PER_SEC) / 60 << " min\n";
		RecordTotalTime += (double)End - (double)start;

		BestB[i] = Bn;
		if (i <= PreRange) {
			if (!FindNA0 && !FindNA1) {
				if (BnNAIndex == 0) FindNA0 = true;
				else FindNA1 = true;
			}
		}

		if (i < RNUM) {
			GenBnTag = true; FindBn = false;
			Rnum++;
			GenBn(1);
			if (ExternDir) {
				memcpy(NxtBestTrail, BestTrail, (Rnum + 1 - ExternBnRound) * STATE_LEN);
				memcpy(NxtBest_W, Best_W, (Rnum - ExternBnRound - 1) * sizeof(int));
				memcpy(&NxtBestTrail[Rnum - ExternBnRound + 1], GenBnTrail, ExternBnRound * STATE_LEN);
				memcpy(&NxtBest_W[Rnum - ExternBnRound - 1], GenBn_W, (ExternBnRound + 1) * sizeof(int));
			}
			else {
				memcpy(&NxtBestTrail[1], GenBnTrail, ExternBnRound * STATE_LEN);
				memcpy(NxtBest_W, GenBn_W, (ExternBnRound + 1) * sizeof(int));
				memcpy(&NxtBestTrail[ExternBnRound + 1], &BestTrail[ExternBnRound], (Rnum - ExternBnRound) * STATE_LEN);
				memcpy(&NxtBest_W[ExternBnRound + 1], &Best_W[ExternRound], (Rnum - ExternBnRound - 1) * sizeof(int));
			}
		
			BnNAIndex = 1;
			for (int r = 0; r < Rnum; r++) {
				if (NxtBest_W[r] == 0) {
					BnNAIndex = 0; break;
				}
			}
			Rnum--;
			GenBnTag = false;
		}

		//Pre-search
		if (i <= PreRange) {
			int NxtBn = Bn;
			start = clock();
			if (!FindNA0) {
				if (i == 3) {
					FindBn = false;
					Bn = BestB[i];
					while (!FindBn) {
						RoundNA0();
						if (!FindBn) Bn += weight[WeightLen - 2];
					}
				}
				else {
					Bn = NxtNA0Bn; FindBn = false;
					RoundNA0();
				}
				if (i < PreRange) {
					GenBnTag = true; FindBn = false;
					Rnum++;
					GenBn(0); 
					NxtNA0Bn = Bn;
					Rnum--;
					GenBnTag = false;
				}
			}
			else if (i < PreRange && BnNAIndex == 1) {
				memcpy(BestTrail, TMPNA0_TRAIL, (Rnum + 1) * STATE_LEN);
				memcpy(Best_W, TMPNA0_W, Rnum * sizeof(int));
				GenBnTag = true; FindBn = false;
				Rnum++;
				GenBn(0); 
				NxtNA0Bn = Bn;
				Rnum--;
				GenBnTag = false;
			}

			if (!FindNA1) {
				PreSearchTag = true;
				FindBn = false;
				Bn = BestB[i];
				while (!FindBn) {
					RoundNA1();
					if (FindBASN) break; 
					if (!FindBn) Bn += weight[WeightLen - 2];
				}
				PreSearchTag = false;
			}			

			End = clock();
			printf("PreSearch time: %fs, %fmin\n", ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
			message << "PreSearch time: " << ((double)End - (double)start) / CLOCKS_PER_SEC << " s,  " << (((double)End - (double)start) / CLOCKS_PER_SEC) / 60 << " min\n";
			RecordTotalTime += (double)End - (double)start;
			Bn = NxtBn;
		}

		printf("\nTotal Time: %fs, %fmin\n\n", (RecordTotalTime) / CLOCKS_PER_SEC, ((RecordTotalTime) / CLOCKS_PER_SEC) / 60);
		message << "\nTotal Time: " << (RecordTotalTime) / CLOCKS_PER_SEC << " s,  " << (RecordTotalTime / CLOCKS_PER_SEC) / 60 << " min\n\n";
		logToFile(fileName, message.str());

		if (i < RNUM) {
			memcpy(BestTrail, NxtBestTrail, (i + 2) * STATE_LEN);
			memcpy(Best_W, NxtBest_W, (i + 1) * sizeof(int));
		}

	}


	printf("B: ");
	message.str("");
	message << "BestB:\n";
	for (int i = 1; i <= RNUM; i++) {
		printf("%f ", (double)BestB[i]);
		message << BestB[i] << ", ";
	}
	logToFile(fileName, message.str());
	printf("\nTotal Time: %fs, %fmin\n", (RecordTotalTime) / CLOCKS_PER_SEC, ((RecordTotalTime) / CLOCKS_PER_SEC) / 60);

	return;
}

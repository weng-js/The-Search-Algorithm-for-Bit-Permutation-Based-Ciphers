#include<emmintrin.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<algorithm>
#include<bitset>
#include "GlobleVariables.h"
#include "State.h"

int RecordLBForValue::ReturnLB(int r, int asn_tag) {
	if (RecordRdLB[r - 1]) return RecordRdLB[r - 1];
	else if (RecordRnum[0] > r) return 0;
	else if (RecordRnum[Round_NUM - 1] <= r) return RecordRdLB[RecordRnum[Round_NUM - 1] - 1] + LB[r - RecordRnum[Round_NUM - 1]][asn_tag];
	int low = 0, high = Round_NUM - 1, mid = (low + high) / 2;
	while (RecordRnum[mid] != r && low < high) {
		if (RecordRnum[mid] > r) {
			high = mid - 1; 
		}
		else {
			low = mid + 1; if (RecordRnum[low] > r) break;					
		}
		mid = (low + high) / 2;
	}
	return RecordRdLB[RecordRnum[mid] - 1] + LB[r - RecordRnum[mid]][asn_tag];	
}

void RecordLBForValue::UpdateOrInsertLB(int r, int lb) {
	if (RecordRdLB[r - 1]) {
		RecordRdLB[r - 1] = lb; return;
	}	
	int index;
	for (index = Round_NUM; index > 0; index--) {
		if (RecordRnum[index - 1] > r) {
			RecordRnum[index] = RecordRnum[index - 1];
		}
		else  break;
	}	
	RecordRnum[index] = r; RecordRdLB[r - 1] = lb; Round_NUM++; 
	return;	
}

int RecordLBForValue_ASN::ReturnLB(int r, int asn) {
	if (RecordRdLB[r - 1]) return RecordRdLB[r - 1];
	else if (RecordRnum[0] > r) return 0; 
	else if (RecordRnum[Round_NUM - 1] <= r) return RecordRdLB[RecordRnum[Round_NUM - 1] - 1] + ASNLB[r - RecordRnum[Round_NUM - 1]][asn];
	int low = 0, high = Round_NUM - 1, mid = (low + high) / 2;
	while (RecordRnum[mid] != r && low < high) {
		if (RecordRnum[mid] > r) {
			high = mid - 1; 
		}
		else {
			low = mid + 1; if (RecordRnum[low] > r) break;					
		}
		mid = (low + high) / 2;
	}
	return RecordRdLB[RecordRnum[mid] - 1] + ASNLB[r - RecordRnum[mid]][asn];
	
}

void RecordLBForValue_ASN::UpdateOrInsertLB(int r, int lb) {
	if (RecordRdLB[r - 1]) {
		RecordRdLB[r - 1] = lb; return;
	}	
	int index;
	for (index = Round_NUM; index > 0; index--) {
		if (RecordRnum[index - 1] > r) {
			RecordRnum[index] = RecordRnum[index - 1];
		}
		else  break;
	}	
	RecordRnum[index] = r; RecordRdLB[r - 1] = lb; Round_NUM++; 
	return;	
}

// for trail
void initial_AllTrail() {
	memset(Trail_FW, 0, (RNUM + 1) * STATE_LEN);
	memset(Trail_BW, 0, (RNUM + 1) * STATE_LEN);
	memset(BestTrail, 0, (RNUM + 1) * STATE_LEN);
	memset(T_W_FW, 0, sizeof(T_W_FW));
	memset(T_W_BW, 0, sizeof(T_W_BW));
	memset(Best_W, 0, sizeof(Best_W));
	memset(TMPX, 0, sizeof(TMPX));
}

STATE UpdateStateRoundN_ASN(STATE s, int sbx_nr_w) {
	s.W += s.w;
	s.w = s.nr_sbx_num + sbx_nr_w;
	return s;
}

STATE UpdateStateRoundI_j_ASN(STATE s, int nr_w) { 
	s.nr_sbx_num += nr_w;
	s.j++;
	return s;
}

STATE FWUpdateStateRoundI_ASN(STATE s, int sbx_nr_w) {
	s.W += s.w;
	s.w = s.nr_sbx_num + sbx_nr_w;
	s.j = 0;
	s.nr_sbx_num = 0;
	s.sbx_num = 0;
	s.rnum++;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i]&& TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i]&&!TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++; 
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE BWUpdateStateRoundI_ASN(STATE s, int sbx_nr_w) {
	s.W += s.w;
	s.w = s.nr_sbx_num + sbx_nr_w;
	s.j = 0;
	s.nr_sbx_num = 0;
	s.sbx_num = 0;
	s.rnum--;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && !TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE FWUpdateStateRoundN(STATE s, int w, int sbx_nr_w) {
	s.w += w;
	T_W_FW[s.rnum - 1] = s.w;
	s.W += s.w;
	s.w = s.nr_minw + sbx_nr_w;
	s.rnum++;
	return s;
}

STATE BWUpdateStateRoundN(STATE s, int w, int sbx_nr_w) { 
	s.w += w;
	T_W_BW[s.rnum - 1] = s.w;
	s.W += s.w;
	s.w = s.nr_minw + sbx_nr_w;
	s.rnum--;
	return s;
}

STATE FWUpdateStateRoundI(STATE s, int w, int sbx_nr_w) {
	s.w += w;
	T_W_FW[s.rnum - 1] = s.w;
	s.W += s.w;
	s.w = s.nr_minw + sbx_nr_w;
	s.j = 0; s.sbx_num = 0; s.nr_minw = 0; s.nr_sbx_num = 0;
	s.rnum++;
	TMPX2R[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i]&& TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_FW[s.rnum].m128i_u8[i];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.nr_minw += FWWeightMinandMax[FWSBoxPermutation[i]][TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_FW[s.rnum].m128i_u8[i];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1]; 
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE BWUpdateStateRoundI(STATE s, int w, int sbx_nr_w) { 
	s.w += w;
	T_W_BW[s.rnum - 1] = s.w;
	s.W += s.w;
	s.w = s.nr_minw + sbx_nr_w;
	s.j = 0; s.sbx_num = 0; s.nr_minw = 0; s.nr_sbx_num = 0;
	s.rnum--;
	TMPX2R[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_BW[s.rnum].m128i_u8[i];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]]) {
				TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.nr_minw += FWWeightMinandMax[BWSBoxPermutation[i]][TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && !TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_BW[s.rnum].m128i_u8[i];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1]; 
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE UpdateStateRoundI_j(STATE s, int w, int nr_w) { 
	s.w += w;
	s.nr_minw += nr_w;
	if (!s.sbx_tag[s.j] && nr_w) s.nr_sbx_num++;
	s.j++;
	return s;
}

STATE UpdateStateRoundNP_j(STATE s, int w) { 
	s.w += w;
	s.j++;
	return s;
}

STATE UpdateStateRoundNP_BW(STATE s, int w) {
	s.W = 0;
	s.w += w;
	s.j = 0;
	return s;
}

STATE UpdateStateRoundNP_FW(int w, int Round) {
	STATE s(Round, w);
	TMPX2R[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_FW[s.rnum].m128i_u8[i];
			s.w += FWWeightMinandMax[i][Trail_FW[s.rnum].m128i_u8[i]][0];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1]; 
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;

}

STATE GenNRFWState_ASN(int r) {  
	STATE s(r, 0); __m128i Mask = _mm_setzero_si128();
	Trail_FW[r] = _mm_setzero_si128();
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < ASPInfo[ASP_INDEX]; i++) {
		Trail_FW[r].m128i_u8[ASP_FW_Info[ASP_INDEX][i]] = 1;
		s.sbx_a[i] = ASP_FW_Info[ASP_INDEX][i];
		s.sbx_tag[i] = true;
		TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[ASP_FW_Info[ASP_INDEX][i]]]] = 1;
	}
	s.W = 0; s.w = ASPInfo[ASP_INDEX]; s.nr_sbx_num = ASPInfo[ASP_INDEX]; s.sbx_num = s.nr_sbx_num - 1;

#if(TYPE)
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[r], 48), _mm_srli_epi64(Trail_FW[r], 16));
#else
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[r], 16), _mm_srli_epi64(Trail_FW[r], 48));
#endif
	Extern2RMask[r] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[r], Mask));
	return s;
}

STATE GenNRBWState_ASN(int r) {  
	STATE s(r, 0); __m128i Mask = _mm_setzero_si128();
	Trail_BW[r] = _mm_setzero_si128();
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < ASPInfo[ASP_INDEX]; i++) {
		Trail_BW[r].m128i_u8[ASP_BW_Info[ASP_INDEX][i]] = 1;
		s.sbx_a[i] = ASP_BW_Info[ASP_INDEX][i];
		s.sbx_tag[i] = true;
		TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[ASP_BW_Info[ASP_INDEX][i]]]] = 1;
	}
	s.W = 0; s.w = ASPInfo[ASP_INDEX]; s.nr_sbx_num = ASPInfo[ASP_INDEX]; s.sbx_num = s.nr_sbx_num - 1;

#if(TYPE)
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[r], 16), _mm_srli_epi64(Trail_BW[r], 48));
#else
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[r], 48), _mm_srli_epi64(Trail_BW[r], 16));
#endif
	Extern2RMask[r] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[r], Mask)); 
	return s;
}

STATE GenStateNPFWForNa0() {
	STATE s(2, 0);
	s.w = FWLB[1][ASP_INDEX];
	TMPX2R[2] = _mm_setzero_si128(); 
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[2].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			TMPX2R[2].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.nr_minw += weight[1];
			s.nr_sbx_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE GenStateNRBWForNa0(int r) {
	STATE s(r, 0);
	s.w = BWLB[1][ASP_INDEX];
	TMPX2R[r] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.nr_minw += weight[1];
			s.nr_sbx_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE GenStateNRForNa0(int r) {
	STATE s(r, 0);
	__m128i Mask = _mm_setzero_si128();
	Trail_BW[r] = _mm_setzero_si128();
	for (int i = 0; i < ASPInfo[ASP_INDEX]; i++) {
		Trail_BW[r].m128i_u8[ASP_BW_Info[ASP_INDEX][i]] = 1;
	}
#if(TYPE)
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[r], 16), _mm_srli_epi64(Trail_BW[r], 48));
#else
	TMPX[r] = _mm_xor_si128(_mm_slli_epi64(Trail_BW[r], 48), _mm_srli_epi64(Trail_BW[r], 16));
#endif
	Extern2RMask[r] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[r], Mask)); 

	s.w = BWLB[1][ASP_INDEX];
	TMPX2R[r] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.nr_minw += weight[1];
			s.nr_sbx_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE GenStateRI_ASN(STATE s) {
	s.W = 0;
	s.w = s.sbx_num + 1;
	TMPX2R_ASN[s.rnum] = TMPX2R[s.rnum];
	return s;
}

STATE GenStateRI_ASN_FW(STATE s) {
	s.W = 0; s.j = 0; s.sbx_num = 0; s.nr_sbx_num = 0;
	s.rnum++;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.w = s.sbx_num;
	s.sbx_num--;
	return s;
}

STATE GenStateRI_ASN_BW(STATE s) {
	s.W = 0; s.j = 0; s.sbx_num = 0; s.nr_sbx_num = 0;
	s.rnum--;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i; s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && !TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i; s.sbx_tag[s.sbx_num] = true; s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.w = s.sbx_num;
	s.sbx_num--;
	return s;
}

STATE GenStateRI_j_ASN(STATE s, int nr_num) {
	s.W = 0;
	s.w = s.sbx_num + 1;
	s.nr_sbx_num += nr_num;
	TMPX2R_ASN[s.rnum] = TMPX2R[s.rnum];
	s.j++;
	return s;
}

STATE UpdateStateR2Input(STATE s, int w, int nr_w) {
	T_W_FW[0] = w;
	s.W = w;
	s.nr_minw = nr_w - (s.sbx_num + 1) * weight[1]; 
	return s;
}

STATE UpdateStateFWR2andBWInput(STATE s, int w) {
	s.w += w;        
	s.j = 0;
	return s;
}

STATE GenR2OrR3State_ASN(int asn1, int asn2, u8 sbx1_info[], u8 sbx2_info[]) {
	STATE s(2, 0); __m128i Mask = _mm_setzero_si128();
	Trail_FW[1] = _mm_setzero_si128(); Trail_FW[2] = _mm_setzero_si128();
	for (int i = 0; i < asn1; i++) {
		Trail_FW[1].m128i_u8[sbx1_info[i]] = 1;
	}
	for (int i = 0; i < asn2; i++) {
		Trail_FW[2].m128i_u8[sbx2_info[i]] = 1;
	}
	s.W = asn1; s.w = asn2;

#if(TYPE)
	TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 48), _mm_srli_epi64(Trail_FW[1], 16));
	TMPX[2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[2], 48), _mm_srli_epi64(Trail_FW[2], 16));
#else
	TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 16), _mm_srli_epi64(Trail_FW[1], 48));
	TMPX[2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[2], 16), _mm_srli_epi64(Trail_FW[2], 48));
#endif
	Extern2RMask[2] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[2], Mask)); 

	if (!asn2) {
		Trail_FW[3] = TMPX[1];
#if(TYPE)
		TMPX[3] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[3], 48), _mm_srli_epi64(Trail_FW[3], 16));
#else		
		TMPX[3] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[3], 16), _mm_srli_epi64(Trail_FW[3], 48));
#endif
		Extern2RMask[3] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[3], Mask));
		s.w = s.W; s.rnum++;
	}

	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

void GenR1R2StateForNa1(STATE& s_r1, STATE& s_r2, int asn1, int asn2, u8 sbx1_info[], u8 sbx2_info[]) {
	__m128i Mask = _mm_setzero_si128();
	Trail_FW[1] = _mm_setzero_si128(); Trail_FW[2] = _mm_setzero_si128(); __m128i TMPR2Out = _mm_setzero_si128();
	for (int i = 0; i < asn1; i++) {
		Trail_FW[1].m128i_u8[sbx1_info[i]] = 1;
	}
	for (int i = 0; i < asn2; i++) {
		Trail_FW[2].m128i_u8[sbx2_info[i]] = 1;
		TMPR2Out.m128i_u8[FWSBoxPermutation[sbx2_info[i]]] = 1;
	}
	s_r1.w = asn1 * weight[1]; s_r2.w = asn2 * weight[1];

#if(TYPE)
	TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 48), _mm_srli_epi64(Trail_FW[1], 16));
	TMPX[2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[2], 48), _mm_srli_epi64(Trail_FW[2], 16));
#else
	TMPX[1] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[1], 16), _mm_srli_epi64(Trail_FW[1], 48));
	TMPX[2] = _mm_xor_si128(_mm_slli_epi64(Trail_FW[2], 16), _mm_srli_epi64(Trail_FW[2], 48));
#endif
	Extern2RMask[2] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[2], Mask)); 

	TMPX2R[2] = _mm_setzero_si128(); 
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[1].m128i_u8[i] && !TMPR2Out.m128i_u8[FWSBoxROT[i]]) {
			s_r1.sbx_a[s_r1.sbx_num] = i;
			s_r1.sbx_tag[s_r1.sbx_num] = true;
			s_r1.nr_minw += weight[1]; 
			s_r1.sbx_num++;
		}
		else if (!Trail_FW[1].m128i_u8[i] && TMPR2Out.m128i_u8[FWSBoxROT[i]]) {
			s_r1.nr_minw += weight[1];
		}

		if (Trail_FW[2].m128i_u8[i] && TMPX[1].m128i_u8[FWSBoxPermutation[i]]) {
			s_r2.sbx_a[s_r2.sbx_num] = i;
			s_r2.sbx_tag[s_r2.sbx_num] = false;
			if (TMPX[2].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R[2].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1; 
			}
			s_r2.sbx_num++;
		}
		else if (!Trail_FW[2].m128i_u8[i] && TMPX[1].m128i_u8[FWSBoxPermutation[i]]) {
			TMPX2R[2].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s_r2.nr_sbx_num++;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[1].m128i_u8[i] && TMPR2Out.m128i_u8[FWSBoxROT[i]]) {
			s_r1.sbx_a[s_r1.sbx_num] = i;
			s_r1.sbx_tag[s_r1.sbx_num] = false;
			s_r1.sbx_num++;
		}

		if (Trail_FW[2].m128i_u8[i] && !TMPX[1].m128i_u8[FWSBoxPermutation[i]]) {
			s_r2.sbx_a[s_r2.sbx_num] = i;
			s_r2.sbx_tag[s_r2.sbx_num] = true;
			TMPX2R[2].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s_r2.nr_sbx_num++;
			s_r2.sbx_num++;
		}
	}
	s_r1.nr_minw += s_r2.sbx_num * weight[1]; 
	s_r1.sbx_num--;
	s_r2.sbx_num--;
	return;
}

STATE GenBn_STATE_FW() {
	STATE s(Rnum - 1, Bn);
	if (!Best_W[Rnum - 2]) {
		s.rnum = Rnum - 2;
	}

	s.W -= Best_W[s.rnum - 1];
	s.w = 0;
	Trail_FW[s.rnum] = BestTrail[s.rnum];
#if(TYPE)
	TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum - 1], 48), _mm_srli_epi64(BestTrail[s.rnum - 1], 16));
	TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 48), _mm_srli_epi64(BestTrail[s.rnum], 16));
#else
	TMPX[s.rnum - 1] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum - 1], 16), _mm_srli_epi64(BestTrail[s.rnum - 1], 48));
	TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 16), _mm_srli_epi64(BestTrail[s.rnum], 48));
#endif
	TMPX2R[s.rnum] = _mm_setzero_si128();

	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_FW[s.rnum].m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]]) {
				TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1; 
			}
			s.sbx_num++;
		}
		else if (!Trail_FW[s.rnum].m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.nr_minw += FWWeightMinandMax[FWSBoxPermutation[i]][TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_FW[s.rnum].m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_FW[s.rnum].m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1]; 
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[FWSBoxPermutation[FWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;

	if (Best_W[Rnum - 2]) {
		int tmp_Bn = 0;
		for (int i = 0; i < SBox_NUM; i++) {
			if (Trail_FW[s.rnum].m128i_u8[i]) {
				tmp_Bn += FWWeightMinandMax[FWSBoxPermutation[i]][(TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]] ^ FWWeightOrderV[i][Trail_FW[s.rnum].m128i_u8[i]][0])][0];
			}
			else if (TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]) {
				tmp_Bn += FWWeightMinandMax[FWSBoxPermutation[i]][TMPX[s.rnum - 1].m128i_u8[FWSBoxPermutation[i]]][0];
			}
		}
		Bn += tmp_Bn;
	}
	else {
		__m128i tmp_out;
#if(TYPE)
		tmp_out = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 48), _mm_srli_epi64(BestTrail[s.rnum], 16));
#else
		tmp_out = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 16), _mm_srli_epi64(BestTrail[s.rnum], 48));
#endif
		int tmp_Bn = 0;
		for (int i = 0; i < SBox_NUM; i++) {
			if (tmp_out.m128i_u8[i]) {
				tmp_Bn += FWWeightMinandMax[i][tmp_out.m128i_u8[i]][0];
			}
		}
		Bn += tmp_Bn;
	}
	return s;
}

STATE GenBn_STATE_BW(int W) {
	STATE s(2, W);
	if (!Best_W[0]) {
		s.rnum = 3;
	}
	s.W -= Best_W[s.rnum - 2];
	s.w = 0;

	Trail_BW[s.rnum] = BestTrail[s.rnum - 1];
#if(TYPE)
	TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 16), _mm_srli_epi64(BestTrail[s.rnum], 48));
	TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum - 1], 16), _mm_srli_epi64(BestTrail[s.rnum - 1], 48));
#else						
	TMPX[s.rnum + 1] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum], 48), _mm_srli_epi64(BestTrail[s.rnum], 16));
	TMPX[s.rnum] = _mm_xor_si128(_mm_slli_epi64(BestTrail[s.rnum - 1], 48), _mm_srli_epi64(BestTrail[s.rnum - 1], 16));
#endif

	TMPX2R[s.rnum] = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_BW[s.rnum].m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]]) {
				TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			}
			s.sbx_num++;
		}
		else if (!Trail_BW[s.rnum].m128i_u8[i] && TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.nr_minw += FWWeightMinandMax[BWSBoxPermutation[i]][TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail_BW[s.rnum].m128i_u8[i] && !TMPX[s.rnum + 1].m128i_u8[BWSBoxPermutation[i]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = Trail_BW[s.rnum].m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[BWSBoxPermutation[BWSBoxPermutation[i]]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}





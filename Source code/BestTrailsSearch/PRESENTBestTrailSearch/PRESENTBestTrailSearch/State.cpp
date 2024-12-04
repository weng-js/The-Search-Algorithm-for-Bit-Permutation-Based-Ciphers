#include<emmintrin.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include "GlobleVariables.h"
#include "State.h"

// for trail
void initial_Trail() {
	memset(Trail, 0, RNUM * STATE_LEN);
	memset(TmpBestTrail, 0, RNUM * STATE_LEN);
	memset(TmpNaBestTrail, 0, RNUM * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Tmp_Best_w, 0, sizeof(Tmp_Best_w));
	memset(TmpNaBestw, 0, sizeof(TmpNaBestw));
}

void initial_AllTrail() {
	memset(Trail, 0, RNUM * STATE_LEN);
	memset(BestTrail, 0, RNUM * STATE_LEN);
	memset(TmpBestTrail, 0, RNUM * STATE_LEN);
	memset(TmpNaBestTrail, 0, RNUM * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Best_w, 0, sizeof(Best_w));
	memset(Tmp_Best_w, 0, sizeof(Tmp_Best_w));
	memset(TmpNaBestw, 0, sizeof(TmpNaBestw));
}

// for matsui
STATE GenStateToGenBnUP_FW(__m128i& sbox_out, int NaTag) {
	STATE s(Rnum - 1, 0);
	if (NaTag == 0) {
		s.W = BestB[Rnum - 1] - Best_w[Rnum - 2];
	}
	else if (NaTag == 1) {
		s.W = NaLB[Rnum - 1][0] - Best_w[Rnum - 2];
	}
	else {
		s.W = NaLB[Rnum - 1][1] - Best_w[Rnum - 2];
	}
	s.w = Best_w[Rnum - 2];
	__m128i tmp_sbx = _mm_setzero_si128();
	for (int i = 0; i < 16; i++) {
		if (sbox_out.m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = sbox_out.m128i_u8[i];
			s.sbx_g[s.sbx_num] = Sbox_loc[i];
			if (s.sbx_num && (s.sbx_g[s.sbx_num] != s.sbx_g[s.sbx_num - 1])) s.g_num++;
			tmp_sbx = _mm_xor_si128(tmp_sbx, FWSPTable[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0]);
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	int TmpBn = 0; int count = 0;
	for (int i = 0; i < 16; i++) {
		if (tmp_sbx.m128i_u8[i]) {
			TmpBn += FWWeightMinandMax[tmp_sbx.m128i_u8[i]][0];
			count++;
		}
	}
	if (NaTag == 0) Bn = BestB[Rnum - 1] + TmpBn; 
	else if (NaTag == 1) Bn = NaLB[Rnum - 1][0] + TmpBn;
	else {
		if (count >= 2) Bn = NaLB[Rnum - 1][1] + TmpBn;
		else  Bn = NaLB[Rnum - 1][1] + 2 * weight[WeightLen - 2];
	}
	return s;
}

STATE GenStateToGenBnUP_BW(__m128i& sbox_out, int NaTag) {
	STATE s(2, 0);
	if (NaTag == 0) {
		s.W = BestB[Rnum - 1] - Best_w[0];
	}
	else if (NaTag == 1) {
		s.W = NaLB[Rnum - 1][0] - Best_w[0];
	}
	else {
		s.W = NaLB[Rnum - 1][1] - Best_w[0];
	}
	s.w = Best_w[0];
	for (int i = 0; i < SBox_NUM; i++) {
		if (sbox_out.m128i_u8[INVSbox_loc[i][1]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = sbox_out.m128i_u8[INVSbox_loc[i][1]];
			s.sbx_g[s.sbx_num] = INVSbox_loc[i][0];
			if (s.sbx_num && (s.sbx_g[s.sbx_num] != s.sbx_g[s.sbx_num - 1])) s.g_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE FWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out) {
	s.w += w;
	t_w[s.rnum - 1] = s.w;
	Trail[s.rnum] = sbox_out;
	s.W += s.w;
	s.j = 0;
	s.w = s.nr_minw + nxt_minw;
	s.sbx_num = 0; s.g_num = 0;
	for (int i = 0; i < 16; i++) {
		if (sbox_out.m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = sbox_out.m128i_u8[i];
			s.sbx_g[s.sbx_num] = Sbox_loc[i];
			if (s.sbx_num && (s.sbx_g[s.sbx_num] != s.sbx_g[s.sbx_num - 1])) s.g_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	s.nr_minw = 0;
	s.rnum++;
	return s;
}

STATE BWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out) { //
	s.w += w;
	Tmp_Best_w[s.rnum - 1] = s.w;
	TmpBestTrail[s.rnum - 2] = sbox_out;
	s.W += s.w;
	s.j = 0;
	s.w = s.nr_minw + nxt_minw;
	s.sbx_num = 0; s.g_num = 0;
	for (int i = 0; i < SBox_NUM; i++) {
		if (sbox_out.m128i_u8[INVSbox_loc[i][1]]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = sbox_out.m128i_u8[INVSbox_loc[i][1]];
			s.sbx_g[s.sbx_num] = INVSbox_loc[i][0];
			if (s.sbx_num && (s.sbx_g[s.sbx_num] != s.sbx_g[s.sbx_num - 1])) s.g_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	s.nr_minw = 0;
	s.rnum--;
	return s;
}


STATE FWupdate_state_sbx(STATE s, int w, int nxt_min) { 
	s.w += w;
	s.j++;
	if (s.sbx_g[s.j] != s.sbx_g[s.j - 1]) {
		s.g_num--; 
		s.nr_minw += nxt_min;
	}
	return s;
}

STATE BWupdate_state_sbx(STATE s, int w, int nxt_min) {
	s.w += w;
	s.j++;
	if (s.sbx_g[s.j] != s.sbx_g[s.j - 1]) {
		s.g_num--;
		s.nr_minw += nxt_min;
	}
	return s;
}

bool FWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw) { 
	if (s.j == s.sbx_num || s.sbx_g[s.j] != s.sbx_g[s.j + 1]) {
		//Calculated weight
		group_minw = 0;
		for (int i = 0; i < 4; i++) {
			group_minw += FWWeightMinandMax[sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]][0];
		}

		if ((!FindBn && (s.W + tmp_w + s.w + s.nr_minw + group_minw + s.g_num * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) <= FWBn)
			|| (FindBn && (s.W + tmp_w + s.w + s.nr_minw + group_minw + s.g_num * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) < FWBn)) return false;
		else return true;
	}
	else {
		//Estimated minimum weight by counting only the number of individuals
		int count = 0;
		for (int i = 0; i < 4; i++) {
			if (sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]) {
				count++;
			}
		}
		if ((!FindBn && (s.W + tmp_w + s.w + s.nr_minw + (count + s.g_num) * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) <= FWBn)
			|| (FindBn && (s.W + tmp_w + s.w + s.nr_minw + (count + s.g_num) * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) < FWBn)) return false;
		else return true;
	}
}

bool BWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw) { 
	if (s.j == s.sbx_num || s.sbx_g[s.j] != s.sbx_g[s.j + 1]) {
		group_minw = 0;
		for (int i = 0; i < 4; i++) {
			group_minw += BWWeightMinandMax[sbx_out.m128i_u8[BWGroup_SBox[s.sbx_g[s.j]][i]]][0];
		}
		if ((!FindBn && !BWSearchOver && (s.W + s.w + tmp_w + s.nr_minw + group_minw + s.g_num * weight[1] + NaLB[s.rnum - 2][NaBWIndex]) > BWBn)
			|| ((FindBn || BWSearchOver) && (s.W + s.w + tmp_w + s.nr_minw + group_minw + s.g_num * weight[1] + NaLB[s.rnum - 2][NaBWIndex]) >= BWBn)) return true;
		else return false;
	}
	else {
		int count = 0;
		for (int i = 0; i < 4; i++) {
			if (sbx_out.m128i_u8[BWGroup_SBox[s.sbx_g[s.j]][i]]) {
				count++;
			}
		}
		if ((!FindBn && (s.W + tmp_w + s.w + s.nr_minw + (count + s.g_num) * weight[1] + NaLB[s.rnum - 2][NaIndex]) > BWBn)
			|| (FindBn && (s.W + tmp_w + s.w + s.nr_minw + (count + s.g_num) * weight[1] + NaLB[s.rnum - 2][NaIndex]) >= BWBn)) return true;
		else return false;
	}
}
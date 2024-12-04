#include "GlobleVariables.h"
#include "State.h"

// for trail
void initial_AllTrail() {
	memset(Trail, 0, (RNUM + 1) * STATE_LEN);
	memset(BestTrail, 0, (RNUM + 1) * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Best_w, 0, sizeof(Best_w));
	memset(TMPX, 0, sizeof(TMPX));
}

// for ASN
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

STATE UpdateStateRoundI_ASN(STATE s, int sbx_nr_w) {
	s.W += s.w;
	s.w = s.nr_sbx_num + sbx_nr_w;
	s.j = 0;	
	s.nr_sbx_num = 0;
	s.sbx_num = 0;
	s.rnum++;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	__m128i tmp_sbxout = _mm_shuffle_epi8(Trail[s.rnum], SBoxPermutationSSE);
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i]&& TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[SBoxPermutation[i]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[i]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i]&&!TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++; 
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}


STATE UpdateStateRoundN(STATE s, int w, int sbx_nr_w) { 
	s.w += w;
	t_w[s.rnum - 1] = s.w;
	s.W += s.w;
	s.w = s.nr_minw + sbx_nr_w;
	s.rnum++;	
	return s;
}

STATE UpdateStateRoundI_j(STATE s, int w, int nr_w) { 
	s.w += w;
	s.nr_minw += nr_w;
	if (!s.sbx_tag[s.j] && nr_w) s.nr_sbx_num++;
	s.j++;
	return s;
}

STATE UpdateStateRoundI(STATE s, int w, int sbx_nr_w) { 
	s.w += w;
	t_w[s.rnum - 1] = s.w;
	s.W += s.w;	
	s.w = s.nr_minw + sbx_nr_w;
	s.j = 0; s.sbx_num = 0; s.nr_minw = 0; s.nr_sbx_num = 0; 
	s.rnum++;	
	TMPX2R[s.rnum] = _mm_setzero_si128();
	__m128i tmp_sbxout = _mm_shuffle_epi8(Trail[s.rnum], SBoxPermutationSSE);
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i]&& TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = tmp_sbxout.m128i_u8[i];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[SBoxPermutation[i]]) {
				TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[i]) {
			s.nr_minw += FWWeightMinandMax[TMPX[s.rnum - 1].m128i_u8[i]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
		}
	}

	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i]&&!TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = tmp_sbxout.m128i_u8[i];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;	
	return s;
}

STATE UpdateStateRound1() { 
	STATE s(1, 0);
	TMPX2R[s.rnum] = _mm_setzero_si128();
	__m128i tmp_sbxout = _mm_shuffle_epi8(Trail[s.rnum], SBoxPermutationSSE);
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = tmp_sbxout.m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[SBoxPermutation[i]]) {
				TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[i]) {
			s.nr_minw += FWWeightMinandMax[TMPX[s.rnum - 1].m128i_u8[i]][0];
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = tmp_sbxout.m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
			s.sbx_tag[s.sbx_num] = true;
			s.nr_minw += weight[1]; 
			s.nr_sbx_num++;
			TMPX2R[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}


STATE GenStateRI_ASN_FW(STATE s) {
	s.W = 0; s.j = 0; s.sbx_num = 0; s.nr_sbx_num = 0;
	s.rnum++;
	TMPX2R_ASN[s.rnum] = _mm_setzero_si128();
	__m128i tmp_sbxout = _mm_shuffle_epi8(Trail[s.rnum], SBoxPermutationSSE);
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i] && TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = false;
			if (TMPX[s.rnum].m128i_u8[SBoxPermutation[i]]) {
				TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1; 
			}
			s.sbx_num++;
		}
		else if (TMPX[s.rnum - 1].m128i_u8[i]) {
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
		}
	}
	for (int i = 0; i < SBox_NUM; i++) {
		if (tmp_sbxout.m128i_u8[i] && !TMPX[s.rnum - 1].m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_tag[s.sbx_num] = true;
			s.nr_sbx_num++;
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[i]] = 1;
			s.sbx_num++;
		}
	}
	s.w = s.sbx_num;
	s.sbx_num--;
	return s;
}

STATE GenStateRI_ASN(STATE s) {
	s.W = 0;
	s.w = s.sbx_num + 1;
	TMPX2R_ASN[s.rnum] = TMPX2R[s.rnum];	
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
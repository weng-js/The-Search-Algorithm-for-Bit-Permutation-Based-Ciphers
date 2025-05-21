#include<emmintrin.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<algorithm>
#include<bitset>
#include "GlobleVariables.h"
#include "State.h"

// for trail
void initial_AllTrail() {
	memset(Trail, 0, (RNUM + 1) * STATE_LEN);
	memset(BestTrail, 0, (RNUM + 1) * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Best_w, 0, sizeof(Best_w));
}

// for matsui
STATE GenStateForRound1(__m128i& sbox_out) {
	STATE s(1, 0);
	for (int i = 0; i < 16; i++) {
		if (sbox_out.m128i_u8[i]) {
			s.sbx_a[s.sbx_num] = i;
			s.sbx_in[s.sbx_num] = sbox_out.m128i_u8[i];
			s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
			s.sbx_g[s.sbx_num] = Sbox_loc[i];
			if (s.sbx_num && (s.sbx_g[s.sbx_num] != s.sbx_g[s.sbx_num - 1])) s.g_num++;
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}


STATE FWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out) { //
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
	if (s.rnum == Rnum - 1 && s.sbx_num != TrailOutputASN) s.w = -1;
	s.sbx_num--;
	s.nr_minw = 0;
	s.rnum++;
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


bool FWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw) { 
	if (s.j == s.sbx_num || s.sbx_g[s.j] != s.sbx_g[s.j + 1]) {
		group_minw = 0;
		for (int i = 0; i < 4; i++) {
			if (sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]) {
				if (s.rnum == Rnum - 1) {
					if (TrailOutputASBO[FWGroup_SBox[s.sbx_g[s.j]][i]] && DDTorLAT[sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]][TrailOutputASBO[FWGroup_SBox[s.sbx_g[s.j]][i]]] != INFINITY)
						group_minw += DDTorLAT[sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]][TrailOutputASBO[FWGroup_SBox[s.sbx_g[s.j]][i]]];
					else return false;
				}
				else group_minw += FWWeightMinandMax[sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]][0];
			}			
		}
		if ((s.W + tmp_w + s.w + s.nr_minw + group_minw + s.g_num * weight[1] + BestB[Rnum - s.rnum - 1]) > Bn) {
			return false;
		}
		else return true;
	}
	else {
		int count = 0;
		for (int i = 0; i < 4; i++) {
			if (sbx_out.m128i_u8[FWGroup_SBox[s.sbx_g[s.j]][i]]) {
				count++;
			}
		}
		if ((s.W + tmp_w + s.w + s.nr_minw + (count + s.g_num) * weight[1] + BestB[Rnum - s.rnum - 1]) > Bn) {
			return false;
		}
		else {
			return true;
		}
	}
}

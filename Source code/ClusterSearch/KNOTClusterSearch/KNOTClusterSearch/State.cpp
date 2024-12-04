#include<emmintrin.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include "GlobleVariables.h"
#include "State.h"

// for trail
void initial_AllTrail() {
	memset(Trail, 0, sizeof(Trail));
	memset(BestTrail, 0, sizeof(BestTrail));
	memset(t_w, 0, sizeof(t_w));
	memset(Best_w, 0, sizeof(Best_w));
}

// for matsui
STATE GenStateForRound1() { //
	STATE s(1, 0);
	for (int i = 0; i < State_NUM; i++) {
		for (int j = 0; j < 16; j++) {
			if (!Trail[0][i].m128i_u8[j])  continue;
			if (Trail[0][i].m128i_u8[j] & 0xf) {
				s.sbx_a[s.sbx_num] = (i << 5) | (j << 1);
				s.sbx_in[s.sbx_num] = Trail[0][i].m128i_u8[j] & 0xf;
				s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
				s.sbx_num++;
			}
			if (Trail[0][i].m128i_u8[j] & 0xf0) {
				s.sbx_a[s.sbx_num] = (i << 5) | (j << 1) | 1;
				s.sbx_in[s.sbx_num] = Trail[0][i].m128i_u8[j] >> 4;
				s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
				s.sbx_num++;
			}
		}
	}
	s.sbx_num--;
	return s;
}

STATE FWupdate_state_row(STATE s, double w, __m128i sbx_out[]) { //
	s.w += w; t_w[s.rnum - 1] = s.w;
	memcpy(Trail[s.rnum], sbx_out, STATE_LEN);
	s.W += s.w;
	s.j = 0; s.w = 0; s.sbx_num = 0;
	for (int i = 0; i < State_NUM; i++) {
		for (int j = 0; j < 16; j++) {
			if (!sbx_out[i].m128i_u8[j]) continue;
			if (sbx_out[i].m128i_u8[j] & 0xf) {
				s.sbx_a[s.sbx_num] = (i << 5) | (j << 1);
				s.sbx_in[s.sbx_num] = sbx_out[i].m128i_u8[j] & 0xf;
				if (s.rnum == Rnum - 1) {
					if (DDTorLAT[s.sbx_in[s.sbx_num]][TrailOutputASBO[s.sbx_a[s.sbx_num]]] != INFINITY) {
						s.w += DDTorLAT[s.sbx_in[s.sbx_num]][TrailOutputASBO[s.sbx_a[s.sbx_num]]];
					}
					else {
						s.w = -1;
						return s;
					}
				}
				else
					s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
				s.sbx_num++;
			}
			if (sbx_out[i].m128i_u8[j] & 0xf0) {
				s.sbx_a[s.sbx_num] = (i << 5) | (j << 1) | 1;
				s.sbx_in[s.sbx_num] = sbx_out[i].m128i_u8[j] >> 4;
				if (s.rnum == Rnum - 1) {
					if (DDTorLAT[s.sbx_in[s.sbx_num]][TrailOutputASBO[s.sbx_a[s.sbx_num]]] != INFINITY) {
						s.w += DDTorLAT[s.sbx_in[s.sbx_num]][TrailOutputASBO[s.sbx_a[s.sbx_num]]];
					}
					else {
						s.w = -1;
						return s;
					}
				}
				else
					s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
				s.sbx_num++;
			}
		}
	}
	s.sbx_num--;	
	s.rnum++;	
	return s;
}

STATE update_state_sbx(STATE s, double w) { 
	s.w += w;
	s.j++;
	return s;
}


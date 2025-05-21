#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1];        
extern int t_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];    
extern int Best_w[RNUM];
void initial_AllTrail();

extern u8  TrailOutputASBO[SBox_NUM];    
extern int TrailOutputASN;               


//void initial_STATE(STATE& s);
STATE FWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out); //
STATE GenStateForRound1(__m128i& sbox_out);
STATE FWupdate_state_sbx(STATE s, int w, int nxt_min);
bool FWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw);

#endif // !_STATE_H_

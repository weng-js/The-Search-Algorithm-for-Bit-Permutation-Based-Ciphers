#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_

extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
extern int t_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i 
extern int Best_w[RNUM];
extern u8  TrailOutputASBO[SBox_NUM];     
extern int TrailOutputASN;                
extern int ASNUB_ER;

void initial_AllTrail();

STATE GenStateForRound1();
STATE FWupdate_state_row(STATE s, double w, __m128i sbx_out[]); //
STATE update_state_sbx(STATE s, double w);

#endif // !_STATE_H_

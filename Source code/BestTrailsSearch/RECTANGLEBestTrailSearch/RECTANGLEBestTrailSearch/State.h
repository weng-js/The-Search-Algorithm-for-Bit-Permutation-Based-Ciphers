#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


void GenRound1Pattern();
extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
extern int t_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //记录最优的结果，这个用来最后输出
extern int Best_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果，这个用来最后输出
extern int Tmp_Best_w[RNUM]; //临时标记
extern ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果，这个用来最后输出
extern int TmpNaBestw[RNUM]; //临时标记
void initial_Trail();
void initial_AllTrail();
extern int NaIndex;             //当前正在搜索的子集
extern int NaBWIndex;           //对应的逆向标记

STATE GenStateToGenBnUP_FW(__m128i sbox_out[], int NaTag);
STATE GenStateToGenBnUP_BW(__m128i sbox_out[], int NaTag);
STATE FWupdate_state_row(STATE s, double w, __m128i sbx_out[]); //
STATE BWupdate_state_row(STATE s, double w, __m128i sbx_out[]); //
STATE update_state_sbx(STATE s, double w);

#endif // !_STATE_H_

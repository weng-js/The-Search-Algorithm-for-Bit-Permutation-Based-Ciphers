#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM];        
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM];    
extern ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM]; 
extern int t_w[RNUM];
extern int Best_w[RNUM];
extern int Tmp_Best_w[RNUM];
extern int TmpNaBestw[RNUM]; 

extern int NaIndex;             //Index of the subset being searched
extern int NaBWIndex;           //Index of the subset being searched
extern bool FindBn;
extern bool BWSearchOver;

void initial_Trail();
void initial_AllTrail();
STATE GenStateToGenBnUP_FW(__m128i& sbox_out, int NaTag);
STATE GenStateToGenBnUP_BW(__m128i& sbox_out, int NaTag);
STATE FWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out); 
STATE BWupdate_state_row(STATE s, int w, int nxt_minw, __m128i& sbox_out); 
STATE FWupdate_state_sbx(STATE s, int w, int nxt_min);
STATE BWupdate_state_sbx(STATE s, int w, int nxt_min);
bool FWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw);
bool BWjudge_state_ri(STATE s, int tmp_w, __m128i& sbx_out, int& group_minw);
#endif // !_STATE_H_

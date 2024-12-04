#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1];       
extern int t_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];   
extern int Best_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) TMPX[RNUM + 1]; 
extern ALIGNED_TYPE_(__m128i, 16) TMPX2R[RNUM + 1];
extern int Extern2RMask[RNUM + 1];

extern ALIGNED_TYPE_(__m128i, 16) TMPX2R_ASN[RNUM + 1];


void initial_AllTrail();

//For ASN search
STATE UpdateStateRoundN_ASN(STATE s, int sbx_nr_w);
STATE UpdateStateRoundI_j_ASN(STATE s, int nr_w);
STATE UpdateStateRoundI_ASN(STATE s, int sbx_nr_w);

//For Trail search
STATE UpdateStateRoundN(STATE s, int w, int sbx_nr_w);
STATE UpdateStateRoundI_j(STATE s, int w, int nr_w);
STATE UpdateStateRoundI(STATE s, int w, int sbx_nr_w);
STATE UpdateStateRound1();

STATE GenStateRI_ASN_FW(STATE s);
STATE GenStateRI_ASN(STATE s);
STATE GenStateRI_j_ASN(STATE s, int nr_num);
#endif // !_STATE_H_

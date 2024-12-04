#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


extern ALIGNED_TYPE_(__m128i, 16) Trail_BW[RNUM + 1];
extern ALIGNED_TYPE_(__m128i, 16) Trail_FW[RNUM + 1];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];  
extern ALIGNED_TYPE_(__m128i, 16) TMPX[RNUM + 1]; 
extern ALIGNED_TYPE_(__m128i, 16) TMPX2R[RNUM + 1]; 
extern ALIGNED_TYPE_(__m128i, 16) TMPX2R_ASN[RNUM + 1]; 
extern int T_W_BW[RNUM];
extern int T_W_FW[RNUM];
extern int Best_W[RNUM];
extern int Extern2RMask[RNUM + 1];
extern int ASP_INDEX;

void initial_AllTrail();

STATE UpdateStateRoundN_ASN(STATE s, int sbx_nr_w);
STATE UpdateStateRoundI_j_ASN(STATE s, int sbx_nr_w);
STATE FWUpdateStateRoundI_ASN(STATE s, int sbx_nr_w);
STATE BWUpdateStateRoundI_ASN(STATE s, int sbx_nr_w);

STATE FWUpdateStateRoundN(STATE s, int w, int sbx_nr_w);
STATE BWUpdateStateRoundN(STATE s, int w, int sbx_nr_w);
STATE FWUpdateStateRoundI(STATE s, int w, int sbx_nr_w);
STATE BWUpdateStateRoundI(STATE s, int w, int sbx_nr_w);
STATE UpdateStateRoundI_j(STATE s, int w, int sbx_nr_w);
STATE UpdateStateRoundNP_j(STATE s, int w);
STATE UpdateStateRoundNP_BW(STATE s, int w);
STATE UpdateStateRoundNP_FW(int w, int Round);

STATE GenNRFWState_ASN(int r);
STATE GenNRBWState_ASN(int r);

STATE GenStateNPFWForNa0();
STATE GenStateNRBWForNa0(int r);
STATE GenStateNRForNa0(int r);

STATE GenStateRI_ASN(STATE s);
STATE GenStateRI_ASN_FW(STATE s);
STATE GenStateRI_ASN_BW(STATE s);
STATE GenStateRI_j_ASN(STATE s, int nr_num);
STATE UpdateStateR2Input(STATE s, int w, int nr_w);
STATE UpdateStateFWR2andBWInput(STATE s, int w);
STATE GenR2OrR3State_ASN(int asn1, int asn2, u8 sbx1_info[], u8 sbx2_info[]);
void GenR1R2StateForNa1(STATE& s_r1, STATE& s_r2, int asn1, int asn2, u8 sbx1_info[], u8 sbx2_info[]);

STATE GenBn_STATE_FW();
STATE GenBn_STATE_BW(int W);

#endif // !_STATE_H_

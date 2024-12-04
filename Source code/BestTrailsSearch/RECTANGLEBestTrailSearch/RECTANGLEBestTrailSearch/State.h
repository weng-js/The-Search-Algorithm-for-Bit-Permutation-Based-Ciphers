#include"GlobleVariables.h"
#ifndef _STATE_H_
#define _STATE_H_


void GenRound1Pattern();
extern ALIGNED_TYPE_(__m128i, 16) Trail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
extern int t_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //��¼���ŵĽ�����������������
extern int Best_w[RNUM];
extern ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ�����������������
extern int Tmp_Best_w[RNUM]; //��ʱ���
extern ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ�����������������
extern int TmpNaBestw[RNUM]; //��ʱ���
void initial_Trail();
void initial_AllTrail();
extern int NaIndex;             //��ǰ�����������Ӽ�
extern int NaBWIndex;           //��Ӧ��������

STATE GenStateToGenBnUP_FW(__m128i sbox_out[], int NaTag);
STATE GenStateToGenBnUP_BW(__m128i sbox_out[], int NaTag);
STATE FWupdate_state_row(STATE s, double w, __m128i sbx_out[]); //
STATE BWupdate_state_row(STATE s, double w, __m128i sbx_out[]); //
STATE update_state_sbx(STATE s, double w);

#endif // !_STATE_H_

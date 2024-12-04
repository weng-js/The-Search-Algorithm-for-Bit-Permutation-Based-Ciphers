 #include<iostream>
#include<emmintrin.h>
#include<ctime>
#include<string>
#include<fstream>
#include<iomanip>
#include<nmmintrin.h>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
//#pragma comment(linker,"/STACK:1024000000,1024000000") 
#pragma warning(disable:4996)

using namespace std;

ALIGNED_TYPE_(__m128i, 16) Trail[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
int t_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM][State_NUM];    //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //��¼���ŵĽ�����������������
int Best_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ�����������������
int Tmp_Best_w[RNUM]; //��ʱ���

ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ�����������������
int TmpNaBestw[RNUM]; //��ʱ���

ALIGNED_TYPE_(__m128i, 16) GenBnBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ��
int GenBnBestw[RNUM]; //��ʱ���->�洢��չ�õ������ż�

ALIGNED_TYPE_(__m128i, 16) GenBnNa1BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ��
int GenBnNa1Bestw[RNUM]; //��ʱ���->�洢��չ�õ������ż�

ALIGNED_TYPE_(__m128i, 16) GenBnNa2BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //������ʱ��¼���ż�¼���ŵĽ��
int GenBnNa2Bestw[RNUM]; //��ʱ���->�洢��չ�õ������ż�

map<int*, pair<__m128i**, int*>> NaBestTrailMap;

int NaIndex;             //��ǰ�����������Ӽ�
int NaBWIndex;           //��Ӧ��������

int FBWRound;      //����/�������������
bool FBRoundOverTag;

bool FWSearchOver; //�������������һ�֣���ʾ��������������
bool BWSearchOver; //�������������һ�֣����ڱ���Ƿ���������
bool SubFindBn;    //���ڸ����Ӽ������Ͻ�

bool UpdateFW;  //�ж��Ƿ���Ҫ���������
bool UpdateBW;  //�ж��Ƿ���Ҫ���������

int UpdateFWRoundNa1;
int UpdateFWRoundNa2;
int UpdateBWRoundNa1;
int UpdateBWRoundNa2;

//������չ�õ���������
int BnInNA;	   //����Bn�������Ӽ�
int GenBnInNA; //��չ������NA
bool GenBnTag; //Ŀǰ�Ƿ�����չ��
bool GenBnDir; //������չ����������չ�õ�������
int GenBnNa1NPRnum;
int GenBnNa2NPRnum;

bool FindBnNa1;
bool FindBnNa2;
bool FindBnNa3;
int Na1PreNxtBn;
int Na2PreNxtBn;

int Rnum, NPRnum;
int BNPRnum;               //��¼���ŵ���ʼ������

int Bn;   //����������Ͻ�
int BWBn, FWBn; //���������������Ͻ磬��֤�ض�������صĲ��ּ��ǵ�ǰ������ŵ�
int ODirWMin;

//�������������и���NA2
int NA2_SBX1_VALUE;

int BestB[RNUM + 1] = { 0 };
bool FindBn;//�ҵ�����

__m128i count_asn = _mm_setzero_si128();
//__m128i MASK1 = _mm_set1_epi8(0xf);
//__m128i MASK2 = _mm_set1_epi8(0xf0);
clock_t start, End, tmp_time;

FILE* fp;
Node* T;

void FileOutputTrail() {
	char tmpFILENAME[50] = { 0 };
	FILE* tmpfp;
#if(TYPE==0)
	strcat_s(tmpFILENAME, "result/RECTANGLE_Diff_Trail.txt");
#elif(TYPE==1)
	strcat_s(tmpFILENAME, "result/RECTANGLE_Linear_Trail.txt");
#endif

	// �����Ӧ������/�������������Ӧ���ɶ˵�����һ��ֵ
	// BNPRnum    ��ʼ������ 
	
	//��ʼ��
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	ALIGNED_TYPE_(__m128i, 16) PO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	memset(SO, 0, RNUM * STATE_LEN);
	memset(PO, 0, RNUM * STATE_LEN);
	memcpy(SO, BestTrail, (BNPRnum - 1) * STATE_LEN);
	memcpy(&PO[BNPRnum], &BestTrail[BNPRnum], (Rnum - BNPRnum) * STATE_LEN);

	//����1~BNPRnum��������
	for (int r = 0; r < BNPRnum - 1; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if ((SO[r][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //�������Ա任
					PO[r + 1][k] = _mm_xor_si128(PO[r + 1][k], PTable[i][SO[r][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	//����BNPRnum~Rnum-1��������
	for (int r = BNPRnum; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if ((PO[r][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //�������Ա任
					SO[r - 1][k] = _mm_xor_si128(SO[r - 1][k], INVPTable[i][PO[r][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	//�������ɶ�
	for (int i = 0; i < 0x10; i++) {
		//��һ�ֵ�������
		if (SO[0][0].m128i_u8[i]) {
			PO[0][0].m128i_u8[i] = BWWeightOrderIndex[SO[0][0].m128i_u8[i]][0];
		}
		//Rnum�ֵ�������
		if (PO[Rnum - 1][0].m128i_u8[i]) {
			SO[Rnum - 1][0].m128i_u8[i] = FWWeightOrderIndex[PO[Rnum - 1][0].m128i_u8[i]][0];
		}
	}
	//������ż�������
	tmpfp = fopen(tmpFILENAME, "a+");
	fprintf(tmpfp, "\nRNUM_%d:  Bn:%d NP:%d\n", Rnum, Bn, BNPRnum);

	for (int r = 0; r < Rnum; r++) {
		fprintf(tmpfp, "PO[%02d]: 0x", r+1);
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				fprintf(tmpfp, "%02x ", PO[r][s].m128i_u8[k]); //��λ����������䣬������
			}
			//fprintf(tmpfp, "  ");
		}
		fprintf(tmpfp, "\nSO[%02d]: 0x", r+1);
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				fprintf(tmpfp, "%02x ", SO[r][s].m128i_u8[k]); //��λ����������䣬������
			}
			//fprintf(tmpfp, "  ");
		}
		fprintf(tmpfp, "  w: %d\n\n", Best_w[r]);
	}

	fprintf(tmpfp, "\n\n");
	fclose(tmpfp);
}

inline void UpdateFWLBNa1(int i) {
	if (FBRoundOverTag) { //�ѵ���
		Na1FWMinW[FBWRound][i][1] = 1; // ��Ǹñ�����������
		__m128i** NaBestTrail = new __m128i * [FBWRound];
		*NaBestTrail = new __m128i[FBWRound * State_NUM];
		for (int row = 0; row < FBWRound; row++) NaBestTrail[row] = *NaBestTrail + row * State_NUM;
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0][0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na1FWMinW[FBWRound][i][0], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn) Na1FWMinW[FBWRound][i][0] = FWBn;
	else Na1FWMinW[FBWRound][i][0] = FWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //��Ӧ����ֵ����Ҫ�ı�
		for (int k = FBWRound; k < r; k++) {
			Na1FWMinW[r][i][0] = (Na1FWMinW[r][i][0] > (Na1FWMinW[k][i][0] + NaLB[r - k][0]) ? Na1FWMinW[r][i][0] : (Na1FWMinW[k][i][0] + NaLB[r - k][0]));
	}
}
	UpdateFWRoundNa1 = (UpdateFWRoundNa1 < FBWRound) ? UpdateFWRoundNa1 : FBWRound;
}

inline void UpdateFWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) { //�ѵ���
		Na2FWMinW[FBWRound][i][sbox][1] = 1; // ��Ǹñ�����������
		__m128i** NaBestTrail = new __m128i * [FBWRound];
		*NaBestTrail = new __m128i[FBWRound * State_NUM];
		for (int row = 0; row < FBWRound; row++) NaBestTrail[row] = *NaBestTrail + row * State_NUM;
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0][0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na2FWMinW[FBWRound][i][sbox][0], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn) Na2FWMinW[FBWRound][i][sbox][0] = FWBn;
	else Na2FWMinW[FBWRound][i][sbox][0] = FWBn + 1;

	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //��Ӧ����ֵ����Ҫ���и���
		for (int k = FBWRound; k < r; k++) {
			Na2FWMinW[r][i][sbox][0] = (Na2FWMinW[r][i][sbox][0] > (Na2FWMinW[k][i][sbox][0] + NaLB[r - k][1]) ? Na2FWMinW[r][i][sbox][0] : (Na2FWMinW[k][i][sbox][0] + NaLB[r - k][1]));
		}
	}

	UpdateFWRoundNa2 = (UpdateFWRoundNa2 < FBWRound) ? UpdateFWRoundNa2 : FBWRound;
}

inline void UpdateBWLBNa1(int i) {
	if (FBRoundOverTag) { //�ѵ���
		Na1BWMinW[FBWRound][i][1] = 1; // ��Ǹñ�����������

		__m128i** NaBestTrail = new __m128i * [FBWRound];
		*NaBestTrail = new __m128i[FBWRound * State_NUM];
		for (int row = 0; row < FBWRound; row++) NaBestTrail[row] = *NaBestTrail + row * State_NUM;
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0][0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na1BWMinW[FBWRound][i][0], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn || BWSearchOver) Na1BWMinW[FBWRound][i][0] = BWBn;
	else Na1BWMinW[FBWRound][i][0] = BWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //��Ӧ����ֵ����Ҫ�ı�
		for (int k = FBWRound; k < r; k++) {
			Na1BWMinW[r][i][0] = (Na1BWMinW[r][i][0] > (Na1BWMinW[k][i][0] + NaLB[r - k][1]) ? Na1BWMinW[r][i][0] : (Na1BWMinW[k][i][0] + NaLB[r - k][1]));
		}
	}
	UpdateBWRoundNa1 = (UpdateBWRoundNa1 < FBWRound) ? UpdateBWRoundNa1 : FBWRound;
}

inline void UpdateBWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) { //�ѵ���
		Na2BWMinW[FBWRound][i][sbox][1] = 1; // ��Ǹñ�����������
	}

	if (FindBn || BWSearchOver) Na2BWMinW[FBWRound][i][sbox][0] = BWBn;
	else Na2BWMinW[FBWRound][i][sbox][0] = BWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //��Ӧ����ֵ����Ҫ�ı�
		for (int k = FBWRound; k < r; k++) {
			Na2BWMinW[r][i][sbox][0] = (Na2BWMinW[r][i][sbox][0] > (Na2BWMinW[k][i][sbox][0] + NaLB[r - k][2]) ?
				Na2BWMinW[r][i][sbox][0] : (Na2BWMinW[k][i][sbox][0] + NaLB[r - k][2]));
	}
}
	UpdateBWRoundNa2 = (UpdateBWRoundNa2 < FBWRound) ? UpdateBWRoundNa2 : FBWRound;
}

void BWRound_n(STATE s) {
	s.W    += s.w;
	Tmp_Best_w[0] = s.w;
	BWSearchOver = true;
	BWBn = s.W;

	if (UpdateBW) {
		FBRoundOverTag = true;
		memcpy(TmpNaBestTrail, TmpBestTrail, FBWRound * STATE_LEN);
		memcpy(TmpNaBestw    , Tmp_Best_w  , FBWRound * sizeof(int));
	}

	if (GenBnTag) {
		//��չ
		Bn = s.W;
		FindBn = true;
		BNPRnum = NPRnum;
		memcpy(BestTrail[Rnum - 1], TmpBestTrail[0], STATE_LEN);
		Best_w[0] = Tmp_Best_w[1];
		Best_w[Rnum - 1] = Tmp_Best_w[0];
		GenBnDir = false;
		GenBnInNA = s.sbx_num;
	}
	else {
		if (NPRnum == Rnum) {
			//ֻ��������-> ����Ȼ����
			Bn = s.W + ODirWMin;
			FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
			memcpy(BestTrail, TmpBestTrail, Rnum * STATE_LEN);
			memcpy(Best_w, Tmp_Best_w, Rnum * sizeof(int));

		}
		else {  
			memcpy(Trail, TmpBestTrail, Rnum * STATE_LEN);
			memcpy(t_w, Tmp_Best_w, Rnum * sizeof(int));
		}
	}
	return;
}

void BWRound_i(STATE s, __m128i sbx_out[]) {
	int asn; int i; int record_sbx1_value;
	for (i = 0; i < SBox_SIZE; i++) {
		if ((!FindBn && !BWSearchOver && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[s.rnum - 1][NaBWIndex] > BWBn))
			|| ((FindBn || BWSearchOver) && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[s.rnum - 1][NaBWIndex] >= BWBn))) break;
		sbx_out[0] = _mm_xor_si128(sbx_out[0], BWSPTableXor[s.sbx_a[s.j]][s.sbx_in[s.j]][i][0]);
		asn = SBox_NUM - _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(sbx_out[0], count_asn)));

		if ((!FindBn && !BWSearchOver && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + NaLB[s.rnum - 2][NaBWIndex]) > BWBn)
			|| ((FindBn || BWSearchOver) && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + NaLB[s.rnum - 2][NaBWIndex]) >= BWBn)) continue;
		if (s.j == s.sbx_num) {
			if (asn < NaBWIndex + 1) continue; //��һ��asnС���½磬����
			STATE s_nr = BWupdate_state_row(s, BWWeightOrderW[s.sbx_in[s.j]][i], sbx_out);
			if (s.sbx_num == 1 && !GenBnTag) record_sbx1_value = NA2_SBX1_VALUE;
						
			if ((!FindBn && !BWSearchOver && (s_nr.W + s_nr.w + NaLB[s.rnum - 2][NaBWIndex] <= BWBn))
				|| ((FindBn || BWSearchOver) && (s_nr.W + s_nr.w + NaLB[s.rnum - 2][NaBWIndex] < BWBn))) {
				if (s.rnum - 1 == 1) {
					BWRound_n(s_nr);
				}
				else {
					ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM]; memset(tmp_out, 0, STATE_LEN);
					BWRound_i(s_nr, tmp_out);
				}
			}

			if (s.sbx_num == 1 && !GenBnTag) {
				if (s.sbx_a[1] - s.sbx_a[0] <= SBox_NUM / 2) {
					if (FindBn)  Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] > BWBn - s_nr.W) ?
						 Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] : BWBn - s_nr.W;
					else  Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] > BWBn - s_nr.W + 1) ?
						 Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] : BWBn - s_nr.W + 1;
				}
				else {
					if (FindBn) Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] > BWBn - s_nr.W) ?
						 Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] : BWBn - s_nr.W;
					else Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] > BWBn - s_nr.W + 1) ?
						 Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] : BWBn - s_nr.W + 1;
				}
			}
		}
		else {
			if (s.j == 0 && s.sbx_num == 1 && !GenBnTag) NA2_SBX1_VALUE = BWWeightOrderIndex[s.sbx_in[s.j]][i];
			BWRound_i(update_state_sbx(s, BWWeightOrderW[s.sbx_in[s.j]][i]), sbx_out);
		}
	}
	for (int k = 0; k < State_NUM; k++) sbx_out[k] = _mm_xor_si128(sbx_out[k], BWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i - 1][k]);
	return;
}

void FWRound_n(STATE s) { // NPRnum==1 �Ż����ú���
	s.W    += s.w;
	t_w[s.rnum - 1] = s.w;
	FWBn		  = s.W;
	Bn			  = s.W + ODirWMin;
	FindBn = true; SubFindBn = true; BNPRnum = NPRnum;
	if (UpdateFW) {
		FBRoundOverTag = true;
		memcpy(TmpNaBestTrail, Trail[NPRnum], FBWRound * STATE_LEN);
		memcpy(TmpNaBestw    , &t_w[NPRnum], FBWRound * sizeof(int));
	}

	if (GenBnTag) {
		memcpy(BestTrail[Rnum - 1], Trail[Rnum - 1], STATE_LEN);
		Best_w[Rnum - 2] = t_w[Rnum - 2];
		Best_w[Rnum - 1] = t_w[Rnum - 1];
		GenBnDir = true;
		GenBnInNA = s.sbx_num;
	}
	else {
		BnInNA = NaIndex;
		memcpy(BestTrail, Trail, Rnum * STATE_LEN);
		memcpy(Best_w, t_w, Rnum * sizeof(int));
	}
	return;
}

void FWRound_i(STATE s, __m128i sbx_out[]) {
	int asn; int i; int record_sbx1_value;
	for ( i = 0; i < SBox_SIZE; i++) {
		if ((!FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[Rnum - s.rnum][NaIndex] > FWBn))
			|| (FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[Rnum - s.rnum][NaIndex] >= FWBn))) break;
		sbx_out[0] = _mm_xor_si128(sbx_out[0], FWSPTableXor[s.sbx_a[s.j]][s.sbx_in[s.j]][i][0]);
		asn = SBox_NUM - _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(sbx_out[0], count_asn)));

		if ((!FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) > FWBn)
			|| (FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + NaLB[Rnum - s.rnum - 1][NaIndex]) >= FWBn)) continue;
		if (s.j == s.sbx_num) {
			if (asn < NaIndex + 1) continue;
			STATE s_nr = FWupdate_state_row(s, FWWeightOrderW[s.sbx_in[s.j]][i], sbx_out);

			if (s.sbx_num == 0 && !GenBnTag) { //����֪���½��ж��Ƿ�����
				if ((!FindBn && (s_nr.W + Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][0] > FWBn))
					|| (FindBn && (s_nr.W + Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][0] >= FWBn))) continue;
			}
			else if (s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) {
				record_sbx1_value = NA2_SBX1_VALUE;
				if ((s.sbx_a[1] - s.sbx_a[0] <= SBox_NUM / 2)
					&& ((!FindBn && (s_nr.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] > FWBn))
						|| (FindBn && (s_nr.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] >= FWBn)))) continue;
				else if ((s.sbx_a[1] - s.sbx_a[0] > SBox_NUM / 2)
					&& ((!FindBn && (s_nr.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] > FWBn))
						|| (FindBn && (s_nr.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] >= FWBn)))) continue;
			}

			if ((!FindBn && (s_nr.W + s_nr.w + NaLB[Rnum - s.rnum - 1][NaIndex] <= FWBn))
				|| (FindBn && (s_nr.W + s_nr.w + NaLB[Rnum - s.rnum - 1][NaIndex] < FWBn))) {
				if (s.rnum + 1 == Rnum) {
					FWRound_n(s_nr);
				}
				else {
					ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM]; memset(tmp_out, 0, STATE_LEN);
					FWRound_i(s_nr, tmp_out);
				}
			}

			if (s.sbx_num == 0 && !GenBnTag) {
				//�������������Ը����½�
				if (FindBn) Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][0] = FWBn - s_nr.W;
				else Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][0] = FWBn - s_nr.W + 1;
			}
			else if (s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) {
				if (s.sbx_a[1] - s.sbx_a[0] <= SBox_NUM / 2) {
					if (FindBn)  Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] = FWBn - s_nr.W;
					else  Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[1] - s.sbx_a[0] - 1][0] = FWBn - s_nr.W + 1;
				}
				else {
					if (FindBn) Na2FWMinW[Rnum - s.rnum][Na2InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] = FWBn - s_nr.W;
					else Na2FWMinW[Rnum - s.rnum][Na2InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][s.sbx_a[0] - s.sbx_a[1] + SBox_NUM - 1][0] = FWBn - s_nr.W + 1;
				}				
			}
		}
		else {
			if (s.j == 0 && s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) NA2_SBX1_VALUE = FWWeightOrderIndex[s.sbx_in[s.j]][i];
			FWRound_i(update_state_sbx(s, FWWeightOrderW[s.sbx_in[s.j]][i]), sbx_out);
		}
	}
	for (int k = 0; k < State_NUM; k++) sbx_out[k] = _mm_xor_si128(sbx_out[k], FWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i - 1][k]);
	return;
}

void FWRound_1(STATE s, __m128i sbx_out[], Tree p) { //�ӵ�һ�ֿ�ʼ����������NA3
	int asn; int i;
	for (i = 1; i < SBox_SIZE; i++) { //����Ķ��ǻ�ԾSBox�����Ա����1��ʼ: �п�����������
		if ((!FindBn && (s.w + Round1MinW[i] + NaLB[Rnum - 1][NaIndex] > FWBn)) || (FindBn && (s.w + Round1MinW[i] + NaLB[Rnum - 1][NaIndex] >= FWBn))) break;
		sbx_out[0] = _mm_xor_si128(sbx_out[0], Round1MinSPTableXor[s.sbx_a[s.j]][i][0]);
		asn = SBox_NUM - _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(sbx_out[0], count_asn)));

		if ((!FindBn && (s.w + Round1MinW[i] + asn * weight[1] + NaLB[Rnum - 2][NaIndex]) > FWBn)
			|| (FindBn && (s.w + Round1MinW[i] + asn * weight[1] + NaLB[Rnum - 2][NaIndex]) >= FWBn)) continue;
		if (s.j == s.sbx_num) {
			if (asn < NaIndex + 1) continue;
			STATE s_nr = FWupdate_state_row(s, Round1MinW[i], sbx_out);
			if ((!FindBn && (s_nr.W + s_nr.w + NaLB[Rnum - s.rnum - 1][NaIndex] <= FWBn))
				|| (FindBn && (s_nr.W + s_nr.w + NaLB[Rnum - s.rnum - 1][NaIndex] < FWBn))) {
				if (s.rnum + 1 == Rnum) {
					FWRound_n(s_nr);
				}
				else {
					ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM]; memset(tmp_out, 0, STATE_LEN);
					FWRound_i(s_nr, tmp_out);
				}
			}
		}
		else {
			Node* q = p->lc;
			while (q != NULL) {
				s.sbx_a[s.j + 1] = q->index;
				FWRound_1(update_state_sbx(s, Round1MinW[i]), sbx_out, q);
				q = q->rc;
			}
		}
	}
	for (int k = 0; k < State_NUM; k++) sbx_out[k] = _mm_xor_si128(sbx_out[k], Round1MinSPTable[s.sbx_a[s.j]][i - 1][k]);
	return;
}

void Round_NA1() { 
	//NA1�������жϣ�������
	UpdateFWRoundNa1 = Rnum - 1; UpdateBWRoundNa1 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][0] > Bn)) || (FindBn && (NaLB[Rnum][0] >= Bn))) return;

	NaIndex = 0; NaBWIndex = 1;
	initial_Trail();
	SubFindBn = false;
	//����֮ǰ���ж��Ӽ��������½�
	NPRnum = 1;      //������: ֻ�������
	UpdateFW = true; //ֻ�������������������������ף���˶���Ҫ����һ�����£�ͳһtrue
	STATE s(2, 0);   //�ӵڶ��ֿ�ʼ����
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	//clock_t na_s, na_e;
	//na_s = clock();
	if (Rnum == 2) { //ֱ�Ӹ��ݶ�Ӧ������������С����
		for (int i = 0; i < NA1_NUM; i++) {
			if ((!FindBn && (Na1FWMinW[1][i][0] + Na1RoundNPInput[i][3]) > Bn) || (FindBn && (Na1FWMinW[1][i][0] + Na1RoundNPInput[i][3]) >= Bn)) continue;
			FindBn = true; SubFindBn = true; BNPRnum = NPRnum;	BnInNA = NaIndex;
			Bn = Na1FWMinW[1][i][0] + Na1RoundNPInput[i][3];
			Best_w[0] = Na1RoundNPInput[i][3];
			Best_w[1] = Na1RoundNPFWInfo[i][1];
			memcpy(BestTrail[1], PTable[0][Na1RoundNPInput[i][0]], STATE_LEN);
		}
	}
	else {
		for (int i = 0; i < NA1_NUM; i++) {  //ֻ������������LBArr�����һ�ּ�����С����
			if ((!FindBn && (Na1FWMinW[Rnum - 1][i][0] + Na1RoundNPInput[i][3] > Bn)) || (FindBn && (Na1FWMinW[Rnum - 1][i][0] + Na1RoundNPInput[i][3] >= Bn))) continue;
			FBRoundOverTag = false; FBWRound = Rnum - 1; ODirWMin = Na1RoundNPInput[i][3]; FWBn = Bn - ODirWMin;
			//��¼��һ���������Ϣ
			s.W = 0; s.w = Na1RoundNPFWInfo[i][1];
			memcpy(s.sbx_a, Na1RoundNPFWARRInfo[i][0], ARR_LEN);
			memcpy(s.sbx_in, Na1RoundNPFWARRInfo[i][1], ARR_LEN);
			s.sbx_num = Na1RoundNPFWInfo[i][0] - 1;
			t_w[0] = Na1RoundNPInput[i][3];
			memcpy(Trail[1], PTable[0][Na1RoundNPInput[i][0]], STATE_LEN);

			FWRound_i(s, sbx_out);

			UpdateFWLBNa1(i);
		}	
	}
	//na_e = clock();
	//printf("only fw Time: %fs, %fmin\n", ((double)(na_e - na_s)) / CLOCKS_PER_SEC, (((double)(na_e - na_s)) / CLOCKS_PER_SEC) / 60);

	//for (NPRnum = Rnum; NPRnum >= 2; NPRnum--) {
	for (NPRnum = 2; NPRnum <= Rnum; NPRnum++) {
		//�жϷָ��Ƿ����������������
		if ((!FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int i = 0; i < NA1_NUM; i++) { //�������
			//��ǰ����Ƿ���Խ�������������
			if ((!FindBn && (Na1BWMinW[NPRnum - 1][i][0] + Na1FWOutLB[Rnum - NPRnum][i][0] > Bn))
				|| (FindBn && (Na1BWMinW[NPRnum - 1][i][0] + Na1FWOutLB[Rnum - NPRnum][i][0] >= Bn))) continue;
			BWSearchOver = false;
			if (NPRnum == 2) {
				//ֱ�ӿ��Լ���õ���������
				BWSearchOver = true; BWBn = Na1BWMinW[NPRnum - 1][i][0];  //������������ //����NP�ֵ���С���������ڸ��������񣩣�ͳһ��������
				t_w[0] = Na1RoundNPBWInfo[i][1];
				memcpy(Trail[0], INVPTable[0][Na1RoundNPInput[i][0]], STATE_LEN);
			}
			else {
				//NPRnum > 2													//�ж��Ƿ���Ҫ���£������Ӧ�����Ӧ���������������ã�������Ҫ
				if (Na1BWMinW[NPRnum - 1][i][1]) {
					//֮ǰ�������ף��ʿ���ֱ�ӵõ���������
					BWSearchOver = true; BWBn = Na1BWMinW[NPRnum - 1][i][0];
					auto itor = NaBestTrailMap.find(&Na1BWMinW[NPRnum - 1][i][0]);
					if (itor != NaBestTrailMap.end()) {
						memcpy(Trail, &itor->second.first[0][0], (NPRnum - 1) * STATE_LEN);
						memcpy(t_w, &itor->second.second[0], (NPRnum - 1) * sizeof(int));
					}
				}
				else {
					UpdateBW = true; FBRoundOverTag = false; FBWRound = NPRnum - 1;
					ODirWMin = Na1FWOutLB[Rnum - NPRnum][i][0]; BWBn = Bn - ODirWMin;
					s.W = 0; 
					s.w = Na1RoundNPBWInfo[i][1];								//��һ�ֵ���С����
					memcpy(s.sbx_a, Na1RoundNPBWARRInfo[i][0], ARR_LEN);
					memcpy(s.sbx_in, Na1RoundNPBWARRInfo[i][1], ARR_LEN);
					s.sbx_num = Na1RoundNPBWInfo[i][0] - 1;
					s.rnum = NPRnum - 1; //ʵ���ϴ���һ�ֿ�ʼ
					memcpy(TmpBestTrail[NPRnum - 2], INVPTable[0][Na1RoundNPInput[i][0]], STATE_LEN);
					//��������
					BWRound_i(s, sbx_out);

					UpdateBWLBNa1(i);
				}

			}

			//�ж����������꣬���������ڷ�Χ�ڣ���������������Ҫ�ж������֣������1/2�֣�����ֱ�Ӹ����������������͸���������Ϣ
			if (!BWSearchOver) continue;

			if (NPRnum == Rnum) continue;
			else if (NPRnum == Rnum - 1) {
				//����ֱ�Ӹ��ݱ��õ���С��
				for (int Out = 0; Out < NA1_NUM; Out++) {
					if (Na1OutWeightOrder[i][Out] == INFINITY) break;
					if ((!FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1] > Bn))
						|| (FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1] >= Bn))) continue;
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1]; //��������
					memcpy(Best_w, t_w, NPRnum * sizeof(int));
					Best_w[NPRnum - 1] = Na1OutWeightOrder[i][Out];
					Best_w[NPRnum] = Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1];
					memcpy(BestTrail, Trail, NPRnum * STATE_LEN);
					memcpy(BestTrail[NPRnum], PTable[0][Na1RoundNPInput[Na1InOutLink[i][Out]][0]], STATE_LEN); //������
				}
			}
			else if (Na1FWOutLB[Rnum - NPRnum][i][1]) {
				//����ֱ�ӵõ�����
				FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
				Bn = BWBn + Na1FWOutLB[Rnum - NPRnum][i][0]; //��������
				auto itor = NaBestTrailMap.find(&Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Na1FWOutLB[Rnum - NPRnum][i][2]]][0]);
				if (itor != NaBestTrailMap.end()) {
					memcpy(Trail[NPRnum], &itor->second.first[0][0], (Rnum - NPRnum) * STATE_LEN);
					memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
				}
				t_w[NPRnum - 1] = Na1OutWeightOrder[i][Na1FWOutLB[Rnum - NPRnum][i][2]];
				memcpy(BestTrail, Trail, Rnum * STATE_LEN);
				memcpy(Best_w, t_w, Rnum * sizeof(int));
			}
			else {
				for (int Out = 0; Out < NA1_NUM; Out++) {
					if (Na1OutWeightOrder[i][Out] == INFINITY) break;
					if ((!FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][0] > Bn))
						|| (FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][0] >= Bn))) continue;
					if (Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][1]) {
						FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
						Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][0]; //��������
						auto itor = NaBestTrailMap.find(&Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][0]);
						if (itor != NaBestTrailMap.end()) {
							memcpy(Trail[NPRnum], &itor->second.first[0][0], (Rnum - NPRnum) * STATE_LEN);
							memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
						}
						t_w[NPRnum - 1] = Na1OutWeightOrder[i][Out];
						memcpy(BestTrail, Trail, Rnum * STATE_LEN);
						memcpy(Best_w, t_w, Rnum * sizeof(int));
					}
					else {
						UpdateFW = true; FBRoundOverTag = false; FBWRound = Rnum - NPRnum;
						ODirWMin = BWBn + Na1OutWeightOrder[i][Out]; FWBn = Bn - ODirWMin;
						t_w[NPRnum - 1] = Na1OutWeightOrder[i][Out];
						memcpy(Trail[NPRnum], PTable[0][Na1RoundNPInput[Na1InOutLink[i][Out]][0]], STATE_LEN);
						s.w = Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1]; s.W = 0;
						memcpy(s.sbx_a, Na1RoundNPFWARRInfo[Na1InOutLink[i][Out]][0], ARR_LEN);
						memcpy(s.sbx_in, Na1RoundNPFWARRInfo[Na1InOutLink[i][Out]][1], ARR_LEN);
						s.sbx_num = Na1RoundNPFWInfo[Na1InOutLink[i][Out]][0] - 1;
						s.rnum = NPRnum + 1;

						FWRound_i(s, sbx_out);

						UpdateFWLBNa1(Na1InOutLink[i][Out]);
					}
				}
			}
		}
	}

	if (!FindBnNa1) FindBnNa1 = SubFindBn;

	if (SubFindBn) {
		NaLB[Rnum][0] = Bn;
		if (Rnum < Na12UBRnum) {
			memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
			memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
			GenBnNa1NPRnum = BNPRnum;
			Na1PreNxtBn = Bn;
		}
	}
	else if (FindBn) NaLB[Rnum][0] = (NaLB[Rnum][0] > Bn ? NaLB[Rnum][0] : Bn); //!SubFindBn&&FindBn
	else NaLB[Rnum][0] = (NaLB[Rnum][0] > (Bn + 1) ? NaLB[Rnum][0] : (Bn + 1));
}

void Round_NA2() { 
	//NA2�������жϣ�������
	UpdateFWRoundNa2 = Rnum - 1; UpdateBWRoundNa2 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][1] > Bn)) || (FindBn && (NaLB[Rnum][1] >= Bn))) return;
	SubFindBn = false;
	NaIndex = 1; NaBWIndex = 2;
	initial_Trail();
	NPRnum = 1;
	UpdateFW = true; //��������������LBArr
	STATE s(2, 0);
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	//clock_t na_s, na_e;
	//na_s = clock();
	//Na2û�н�������������������Ҫ��ǻ�ԾSBox�ĸ���
	if (Rnum == 2) { //ֱ�Ӹ��ݶ�Ӧ������������С����
		for (int s = 0; s < (SBox_NUM / 2); s++) {
			for (int i = 0; i < NA2_NUM; i++) {
				if ((!FindBn && (Na2FWMinW[1][i][s][0] + Na2RoundNPInput[i][3]) > Bn) || (FindBn && (Na2FWMinW[1][i][s][0] + Na2RoundNPInput[i][3]) >= Bn)) continue;
				FindBn = true;	SubFindBn = true; BNPRnum = NPRnum;	BnInNA = NaIndex;
				Bn = Na2FWMinW[1][i][s][0] + Na2RoundNPInput[i][3];
				Best_w[0] = Na2RoundNPInput[i][3];
				Best_w[1] = Na2RoundNPFWInfo[i][s][1];
				memcpy(BestTrail[1], Na2FWOutput[i][s], STATE_LEN);
			}
		}
	}
	else {
		for (int sbox = 0; sbox < (SBox_NUM / 2); sbox++) {
			for (int i = 0; i < NA2_NUM; i++) {
				//���½��֦Ӧ�ø���
				if ((!FindBn && (Na2FWMinW[Rnum - 1][i][sbox][0] + Na2RoundNPInput[i][3] > Bn)) || (FindBn && (Na2FWMinW[Rnum - 1][i][sbox][0] + Na2RoundNPInput[i][3] >= Bn))) continue;
				FBRoundOverTag = false; FBWRound = Rnum - 1;
				ODirWMin = Na2RoundNPInput[i][3]; FWBn = Bn - ODirWMin;
				s.W = 0; s.w = Na2RoundNPFWInfo[i][sbox][1];
				memcpy(s.sbx_a, Na2RoundNPFWARRInfo[i][sbox][0], ARR_LEN);
				memcpy(s.sbx_in, Na2RoundNPFWARRInfo[i][sbox][1], ARR_LEN);
				s.sbx_num = Na2RoundNPFWInfo[i][sbox][0] - 1;

				t_w[0] = Na2RoundNPInput[i][3];
				memcpy(Trail[1], Na2FWOutput[i][sbox], STATE_LEN);
				FWRound_i(s, sbx_out);

				UpdateFWLBNa2(i, sbox);
			}
		}
	}
	//na_e = clock();
	//printf("only fw Time: %fs, %fmin\n", ((double)(na_e - na_s)) / CLOCKS_PER_SEC, (((double)(na_e - na_s)) / CLOCKS_PER_SEC) / 60);

	for (NPRnum = 2; NPRnum <= Rnum; NPRnum++) {
		//�жϷָ��Ƿ����������������
		if ((!FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int sbox = 0; sbox < (SBox_NUM / 2); sbox++) {
			for (int i = 0; i < NA2_NUM; i++) { //�������						
				if ((!FindBn && (Na2BWMinW[NPRnum - 1][i][sbox][0] + Na2FWOutLB[Rnum - NPRnum][i][sbox][0] > Bn))
					|| (FindBn && (Na2BWMinW[NPRnum - 1][i][sbox][0] + Na2FWOutLB[Rnum - NPRnum][i][sbox][0] >= Bn))) continue;
				BWSearchOver = false;
				if (NPRnum == 2) {
					//����ֱ�Ӹ��ݱ��õ���С��
					BWSearchOver = true;
					BWBn = Na2BWMinW[NPRnum - 1][i][sbox][0];  //������������ //����NP�ֵ���С������ͳһ��������
					t_w[0] = Na2RoundNPBWInfo[i][sbox][1];
					memcpy(Trail[0], Na2BWOutput[i][sbox], STATE_LEN);
				}
				else {
					//�ж��Ƿ���Ҫ������£������Ӧ�����Ӧ���������������ã�������Ҫ
					if (Na2BWMinW[NPRnum - 1][i][sbox][1]) {
						BWSearchOver = true;
						BWBn = Na2BWMinW[NPRnum - 1][i][sbox][0];  //������������ //����NP�ֵ���С������ͳһ��������
						auto itor = NaBestTrailMap.find(&Na2BWMinW[NPRnum - 1][i][sbox][0]);
						if (itor != NaBestTrailMap.end()) {
							memcpy(Trail, &itor->second.first[0][0], (NPRnum - 1) * STATE_LEN);
							memcpy(t_w, &itor->second.second[0], (NPRnum - 1) * sizeof(int));
						}
					}
					else {
						UpdateBW = true; FBRoundOverTag = false; FBWRound = NPRnum - 1;
						ODirWMin = Na2FWOutLB[Rnum - NPRnum][i][sbox][0]; BWBn = Bn - ODirWMin;
						s.W = 0;
						s.w = Na2RoundNPBWInfo[i][sbox][1];                            //��һ�ֵ���С����
						memcpy(s.sbx_a, Na2RoundNPBWARRInfo[i][sbox][0], ARR_LEN);
						memcpy(s.sbx_in, Na2RoundNPBWARRInfo[i][sbox][1], ARR_LEN);
						s.sbx_num = Na2RoundNPBWInfo[i][sbox][0] - 1;
						s.rnum = NPRnum - 1; //ʵ���ϴ���һ�ֿ�ʼ
						memcpy(TmpBestTrail[NPRnum - 2], Na2BWOutput[i][sbox], STATE_LEN);
						BWRound_i(s, sbx_out);

						UpdateBWLBNa2(i, sbox);
					}

				}

				//�ж����������꣬���������ڷ�Χ�ڣ�����������
				if (!BWSearchOver) continue;
				if (NPRnum == Rnum) continue;
				else if (NPRnum == Rnum - 1) { //ֱ�ӿ��Լ���õ���������
					for (int Out = 0; Out < NA2_NUM; Out++) {
						if (Na2OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1] > Bn))
							|| (FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1] >= Bn))) continue;
						FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
						Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1]; //��������
						memcpy(Best_w, t_w, Rnum * sizeof(int));
						Best_w[NPRnum - 1] = Na2RoundNPFWInfo[Na1InOutLink[i][Out]][sbox][1];
						Best_w[NPRnum] = Na2OutWeightOrder[i][Out];
						memcpy(BestTrail, Trail, Rnum * STATE_LEN);
						memcpy(BestTrail[NPRnum], Na2FWOutput[Na2InOutLink[i][Out]][sbox], STATE_LEN);
					}
				}
				else if (Na2FWOutLB[Rnum - NPRnum][i][sbox][1]) {
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na2FWOutLB[Rnum - NPRnum][i][sbox][0]; //��������
					auto itor = NaBestTrailMap.find(&Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Na2FWOutLB[Rnum - NPRnum][i][sbox][2]]][sbox][0]);
					if (itor != NaBestTrailMap.end()) {
						memcpy(Trail[NPRnum], &itor->second.first[0][0], (Rnum - NPRnum) * STATE_LEN);
						memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
					}
					t_w[NPRnum - 1] = Na2OutWeightOrder[i][Na2FWOutLB[Rnum - NPRnum][i][sbox][2]];
					memcpy(BestTrail, Trail, Rnum * STATE_LEN);
					memcpy(Best_w, t_w, Rnum * sizeof(int));
				}
				else {
					//NPRnum > 2
					for (int Out = 0; Out < NA2_NUM; Out++) {
						if (Na2OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][0] > Bn))
							|| (FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][0] >= Bn))) continue;
						if (Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][1]) {
							FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
							Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][0]; //��������
							auto itor = NaBestTrailMap.find(&Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][0]);
							if (itor != NaBestTrailMap.end()) {
								memcpy(Trail[NPRnum], &itor->second.first[0][0], (Rnum - NPRnum) * STATE_LEN);
								memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
							}
							t_w[NPRnum - 1] = Na2OutWeightOrder[i][Out];
							memcpy(BestTrail, Trail, Rnum * STATE_LEN);
							memcpy(Best_w, t_w, Rnum * sizeof(int));
						}
						else {
							UpdateFW = true; FBRoundOverTag = false; FBWRound = Rnum - NPRnum;
							t_w[NPRnum - 1] = Na2OutWeightOrder[i][Out];
							memcpy(Trail[NPRnum], Na2FWOutput[Na2InOutLink[i][Out]][sbox], STATE_LEN);
							ODirWMin = BWBn + Na2OutWeightOrder[i][Out]; FWBn = Bn - ODirWMin;
							s.W = 0;
							s.w = Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1];
							memcpy(s.sbx_a, Na2RoundNPFWARRInfo[Na2InOutLink[i][Out]][sbox][0], ARR_LEN);
							memcpy(s.sbx_in, Na2RoundNPFWARRInfo[Na2InOutLink[i][Out]][sbox][1], ARR_LEN);
							s.sbx_num = Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][0] - 1;
							s.rnum = NPRnum + 1;

							FWRound_i(s, sbx_out);

							UpdateFWLBNa2(Na2InOutLink[i][Out], sbox);
						}
					}
				}
			}
		}
	}

	if (!FindBnNa2)FindBnNa2 = SubFindBn;

	if (SubFindBn) {
		NaLB[Rnum][1] = Bn;
		if (Rnum < Na12UBRnum) {
			memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
			memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
			GenBnNa2NPRnum = BNPRnum;
			Na2PreNxtBn = Bn;
		}
	}
	else if (FindBn) NaLB[Rnum][1] = (NaLB[Rnum][1] > Bn ? NaLB[Rnum][1] : Bn); //!SubFindBn&&FindBn
	else NaLB[Rnum][1] = (NaLB[Rnum][1] > Bn + 1 ? NaLB[Rnum][1] : Bn + 1);

}

void Round_NA3() { //�����������������
	//NA3�������ж�������
	if ((!FindBn && (NaLB[Rnum][2] > Bn)) || (FindBn && (NaLB[Rnum][2] >= Bn))) return;
	NaIndex = 2; UpdateFW = false; SubFindBn = false;
	ODirWMin = 0; FWBn = Bn;
	STATE s(1, 0);
	initial_Trail();

	ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
	memset(tmp_out, 0, STATE_LEN);

	NPRnum = 1; //�ӵ�һ�ֿ�ʼ����

	// ��ʹ����search pattern��������Ҫ��¼��ԾSBox��Ϊ��������AS��index�������
	s.sbx_a[0] = 0; s.sbx_num = 2;
	while ((!FindBn && ((s.sbx_num + 1) * weight[1] + NaLB[Rnum - 1][2] <= Bn)) || (FindBn && ((s.sbx_num + 1) * weight[1] + NaLB[Rnum - 1][2] < Bn))) {
		s.w = (s.sbx_num + 1) * weight[1];
		FWRound_1(s, tmp_out, T); //���һ���ǵ�ǰ�Ļ�ԾSBox
		s.sbx_num++;
	}

	FindBnNa3 = SubFindBn;
	if (SubFindBn)   NaLB[Rnum][2] = Bn;
	else if (FindBn) NaLB[Rnum][2] = (NaLB[Rnum][2] > Bn ? NaLB[Rnum][2] : Bn); //!SubFindBn&&FindBn
	else NaLB[Rnum][2] = (NaLB[Rnum][2] > Bn + 1 ? NaLB[Rnum][2] : Bn + 1);
	return;
}

void InitLB() { 
	//Rnun>=3
	//��ʼ�� NaLB : �����������и��£���ʼ��Ϊ0����˿���ֱ�ӱȽϣ�ʡȥ��һ��
	for (int i = 0; i < 3; i++) {
		NaLB[Rnum][i] = NaLB[1][i] + NaLB[Rnum - 1][i];
	}
	for (int r = 2; r < Rnum / 2; r++) {
		for (int i = 0; i < 3; i++) {
			NaLB[Rnum][i] = (NaLB[Rnum][i] > (NaLB[r][i] + NaLB[Rnum - r][i]) ? NaLB[Rnum][i] : (NaLB[r][i] + NaLB[Rnum - r][i]));
		}
	}
	//��������������
	//Na1:
	bool tag = true;
	for (int i = 0; i < NA1_NUM; i++) {
		//��ʼ����Ϊ0����˿���ֱ�ӿ�ʼ�Ƚ�
		for (int r = 1; r < Rnum-1; r++) {
			Na1BWMinW[Rnum - 1][i][0] = (Na1BWMinW[Rnum - 1][i][0] > (Na1BWMinW[r][i][0] + NaLB[Rnum - 1 - r][1]) ?
				Na1BWMinW[Rnum - 1][i][0] : (Na1BWMinW[r][i][0] + NaLB[Rnum - 1 - r][1]));
			Na1FWMinW[Rnum - 1][i][0] = (Na1FWMinW[Rnum - 1][i][0] > (Na1FWMinW[r][i][0] + NaLB[Rnum - 1 - r][0]) ?
				Na1FWMinW[Rnum - 1][i][0] : (Na1FWMinW[r][i][0] + NaLB[Rnum - 1 - r][0]));
		}

		if (i == 0) {
			Na1FWLB[Rnum - 1] = Na1FWMinW[Rnum - 1][i][0];
			Na1BWLB[Rnum - 1] = Na1BWMinW[Rnum - 1][i][0];
		}
		else {
			if (Na1FWMinW[Rnum - 1][i][0] < Na1FWLB[Rnum - 1]) Na1FWLB[Rnum - 1] = Na1FWMinW[Rnum - 1][i][0];
			if (Na1BWMinW[Rnum - 1][i][0] < Na1BWLB[Rnum - 1]) Na1BWLB[Rnum - 1] = Na1BWMinW[Rnum - 1][i][0];
		}
	}

	//��¼��������ݵ���С��������
	for (int i = 0; i < NA1_NUM; i++) {
		Na1FWOutLB[Rnum - 1][i][0] = Na1FWMinW[Rnum - 1][Na1InOutLink[i][0]][0] + Na1OutWeightOrder[i][0];
		if (Na1FWMinW[Rnum - 1][Na1InOutLink[i][0]][1]) Na1FWOutLB[Rnum - 1][i][1] = 1;
		Na1FWOutLB[Rnum - 1][i][2] = 0;
		for (int j = 1; j < NA1_NUM; j++) {
			if (Na1OutWeightOrder[i][j] == INFINITY) break;
			if (Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j] < Na1FWOutLB[Rnum - 1][i][0]) {
				Na1FWOutLB[Rnum - 1][i][0] = Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j];
				if(Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][1]) Na1FWOutLB[Rnum - 1][i][1] = 1;
				else Na1FWOutLB[Rnum - 1][i][1] = 0;
				Na1FWOutLB[Rnum - 1][i][2] = j;
			}
			else if (Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j] == Na1FWOutLB[Rnum - 1][i][0]
				&& (!Na1FWOutLB[Rnum - 1][i][1]) && Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][1]) {
				Na1FWOutLB[Rnum - 1][i][1] = 1;
				Na1FWOutLB[Rnum - 1][i][2] = j;
			}			
		}
	}

	//Na2:
	tag = true;
	for (int i = 0; i < NA2_NUM; i++) {
		for (int s = 0; s < (SBox_NUM / 2); s++) {
			//��ʼ����Ϊ0����˿���ֱ�ӿ�ʼ�Ƚ�
			for (int r = 1; r < Rnum - 1; r++) {
				Na2BWMinW[Rnum - 1][i][s][0] = (Na2BWMinW[Rnum - 1][i][s][0] > (Na2BWMinW[r][i][s][0] + NaLB[Rnum - 1 - r][2]) ?
					Na2BWMinW[Rnum - 1][i][s][0] : (Na2BWMinW[r][i][s][0] + NaLB[Rnum - 1 - r][2]));
				Na2FWMinW[Rnum - 1][i][s][0] = (Na2FWMinW[Rnum - 1][i][s][0] > (Na2FWMinW[r][i][s][0] + NaLB[Rnum - 1 - r][1]) ?
					Na2FWMinW[Rnum - 1][i][s][0] : (Na2FWMinW[r][i][s][0] + NaLB[Rnum - 1 - r][1]));
			}

			if (tag) {
				Na2FWLB[Rnum - 1] = Na2FWMinW[Rnum - 1][i][s][0];
				Na2BWLB[Rnum - 1] = Na2BWMinW[Rnum - 1][i][s][0];
				tag = false;
			}
			else {
				if (Na2FWMinW[Rnum - 1][i][s][0] < Na2FWLB[Rnum - 1]) Na2FWLB[Rnum - 1] = Na2FWMinW[Rnum - 1][i][s][0];
				if (Na2BWMinW[Rnum - 1][i][s][0] < Na2BWLB[Rnum - 1]) Na2BWLB[Rnum - 1] = Na2BWMinW[Rnum - 1][i][s][0];
			}
		}
	}

	for (int i = 0; i < NA2_NUM; i++) { //��������ģ������پ�������������
		for (int s = 0; s < (SBox_NUM / 2); s++) {
			tag = true;
			for (int j = 0; j < NA2_NUM; j++) {
				if (Na2OutWeightOrder[i][j] == INFINITY) break;
				if (tag) {
					Na2FWOutLB[Rnum - 1][i][s][0] = Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j] ;
					if (Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][1]) Na2FWOutLB[Rnum - 1][i][s][1] = 1;
					Na2FWOutLB[Rnum - 1][i][s][2] = j;
					tag = false;
				}
				else if ((Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[Rnum - 1][i][s][0]) {
					Na2FWOutLB[Rnum - 1][i][s][0] = Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j];
					if (Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][1]) Na2FWOutLB[Rnum - 1][i][s][1] = 1;
					else Na2FWOutLB[Rnum - 1][i][s][1] = 0;
					Na2FWOutLB[Rnum - 1][i][s][2] = j;
				}
				else if (((Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]) == Na2FWOutLB[Rnum - 1][i][s][0])
					&& (!Na2FWOutLB[Rnum - 1][i][s][1]) && (Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s][1])) {
					Na2FWOutLB[Rnum - 1][i][s][1] = 1;
					Na2FWOutLB[Rnum - 1][i][s][2] = j;
				}
			}
		}
	}

}

void UpdateLB() { //������֮�󣬸����½��ã����µ���������Ǹ�
	bool tag1, tag2;
	//Na1:
	for (int r = UpdateFWRoundNa1; r <= Rnum - 1; r++) {
		for (int i = 0; i < NA1_NUM; i++) {
			if (i == 0) Na1FWLB[r] = Na1FWMinW[r][i][0];
			else if (Na1FWMinW[r][i][0] < Na1FWLB[r]) Na1FWLB[r] = Na1FWMinW[r][i][0];

			Na1FWOutLB[r][i][0] = Na1FWMinW[r][Na1InOutLink[i][0]][0] + Na1OutWeightOrder[i][0];
			if (Na1FWMinW[r][Na1InOutLink[i][0]][1]) Na1FWOutLB[r][i][1] = 1;
			else Na1FWOutLB[r][i][1] = 0;
			Na1FWOutLB[r][i][2] = 0;
			for (int j = 1; j < NA1_NUM; j++) {
				if (Na1OutWeightOrder[i][j] == INFINITY) break;
				if (Na1FWMinW[r][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j] < Na1FWOutLB[r][i][0]) {
					Na1FWOutLB[r][i][0] = Na1FWMinW[r][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j];
					if (Na1FWMinW[r][Na1InOutLink[i][j]][1]) Na1FWOutLB[r][i][1] = 1;
					else Na1FWOutLB[r][i][1] = 0;
					Na1FWOutLB[r][i][2] = j;
				}
				else if (Na1FWMinW[r][Na1InOutLink[i][j]][0] + Na1OutWeightOrder[i][j] == Na1FWOutLB[r][i][0]
					&& (!Na1FWOutLB[r][i][1]) && Na1FWMinW[r][Na1InOutLink[i][j]][1]) {
					Na1FWOutLB[r][i][1] = 1;
					Na1FWOutLB[r][i][2] = j;
				}
			}
		}
	}

	for (int r = UpdateBWRoundNa1; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int i = 0; i < NA1_NUM; i++) {
			if (tag1) {
				Na1BWLB[r] = Na1BWMinW[r][i][0];
				tag1 = false;
			}
			else if (Na1BWMinW[r][i][0] < Na1BWLB[r]) Na1BWLB[r] = Na1BWMinW[r][i][0];
		}
	}

	//Na2:
	for (int r = UpdateFWRoundNa2; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int i = 0; i < NA2_NUM; i++) {
			for (int s = 0; s < (SBox_NUM / 2); s++) {
				if (tag1) {
					Na2FWLB[r] = Na2FWMinW[r][i][s][0];
					tag1 = false;
				}
				else if (Na2FWMinW[r][i][s][0] < Na2FWLB[r]) Na2FWLB[r] = Na2FWMinW[r][i][s][0];

				tag2 = true;
				for (int j = 0; j < NA2_NUM; j++) {
					if (Na2OutWeightOrder[i][j] == INFINITY) break;
					if (tag2) {
						Na2FWOutLB[r][i][s][0] = Na2FWMinW[r][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j];
						if (Na2FWMinW[r][Na2InOutLink[i][j]][s][1]) Na2FWOutLB[r][i][s][1] = 1;
						else Na2FWOutLB[r][i][s][1] = 0;
						Na2FWOutLB[r][i][s][2] = j;
						tag2 = false;
					}
					else if ((Na2FWMinW[r][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[r][i][s][0]) {
						Na2FWOutLB[r][i][s][0] = Na2FWMinW[r][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j];
						if (Na2FWMinW[r][Na2InOutLink[i][j]][s][1]) Na2FWOutLB[r][i][s][1] = 1;
						else Na2FWOutLB[r][i][s][1] = 0;
						Na2FWOutLB[r][i][s][2] = j;
					}
					else if (((Na2FWMinW[r][Na2InOutLink[i][j]][s][0] + Na2OutWeightOrder[i][j]) == Na2FWOutLB[r][i][s][0])
						&& (!Na2FWOutLB[r][i][s][1]) && (Na2FWMinW[r][Na2InOutLink[i][j]][s][1])) {
						Na2FWOutLB[r][i][s][1] = 1;
						Na2FWOutLB[r][i][s][2] = j;
					}
				}
			}
		}
	}
	for (int r = UpdateBWRoundNa2; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int i = 0; i < NA2_NUM; i++) {
			for (int s = 0; s < (SBox_NUM / 2); s++) {
				if (tag1) {
					Na2BWLB[r] = Na2BWMinW[r][i][s][0];
					tag1 = false;
				}
				else if (Na2BWMinW[r][i][s][0] < Na2BWLB[r]) Na2BWLB[r] = Na2BWMinW[r][i][s][0];
			}
		}
	}
}

void GenBnUP(int NaTag) {
	//��չr-1�������������õ�Bn���Ͻ�
	//����ǰ��չ
	NPRnum = BNPRnum;
	if (NaTag == 0) {
		//û������
		NaIndex = 0; NaBWIndex = 0; 
	}
	else if (NaTag == 1) {
		//�Ӽ�1
		NaIndex = 0; NaBWIndex = 1; 
	}
	else {
		//�Ӽ�2
		NaIndex = 1; NaBWIndex = 2;
	}
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	ALIGNED_TYPE_(__m128i, 16) sbox_in1[State_NUM]; memset(sbox_in1, 0, STATE_LEN);
	ALIGNED_TYPE_(__m128i, 16) sbox_in2[State_NUM]; memset(sbox_in2, 0, STATE_LEN);
	if (NPRnum == Rnum - 1) {
		//��Ҫ����������
		for (int i = 0; i < SBox_NUM; i++) {
			if ((BestTrail[Rnum - 3][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //�������Ա任
					sbox_in1[k] = _mm_xor_si128(sbox_in1[k], PTable[i][BestTrail[Rnum - 3][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	else memcpy(sbox_in1, BestTrail[Rnum - 2], STATE_LEN);
	if (NPRnum == 1) {
		memset(sbox_in2, 0, STATE_LEN);
		for (int i = 0; i < SBox_NUM; i++) {
			if ((BestTrail[1][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //�������Ա任
					sbox_in2[k] = _mm_xor_si128(sbox_in2[k], INVPTable[i][BestTrail[1][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	else memcpy(sbox_in2, BestTrail[0], STATE_LEN);

	STATE s = GenStateToGenBnUP_FW(sbox_in1, NaTag);
	ODirWMin = 0; FWBn = Bn;
	int RecordBestW = Best_w[Rnum - 2];
	FWRound_i(s, sbx_out);	
	int RecordBn = Bn;
	s = GenStateToGenBnUP_BW(sbox_in2, NaTag);
	BWBn = Bn;
	BWRound_i(s, sbx_out);
	if (Bn != RecordBn) Best_w[Rnum - 2] = RecordBestW;
}

void matsui() {
	GenTables();
	GenRound1Pattern(); //����pattern��������ʱ����
	char FILENAME[50] = { 0 };
	clock_t s1, e1;

#if(TYPE==0)
	strcat_s(FILENAME, "result/RECTANGLE_Diff.txt");
#elif(TYPE==1)
	strcat_s(FILENAME, "result/RECTANGLE_Linear.txt");
#endif
	////��֤һ������û����
	//fp = fopen(FILENAME, "a+");
	//fclose(fp);

	fp = fopen(FILENAME, "a+");
	fprintf(fp, "Pre-Search Round: Na1Na2:%d  Na3:%d\n\n", Na12UBRnum, Na3UBRnum);
	fclose(fp);

	BestB[1] = weight[1];
	int NextBnInNA; GenBnTag = false;
	double RecordTotalTime = 0;
	s1 = clock();
	for (int i = 2; i <= RNUM; i++) {
		Rnum = i;

		cout << "Round NUM: " << dec << i << endl;		
		if (i == 2) Bn = BestB[i - 1] + weight[1];
		fp = fopen(FILENAME, "a+");
		fprintf(fp, "RUN_%d :\nBeginBn:%d\n", i, Bn);
		fclose(fp);

		if (i > 2) {
			FindBn = true;
			InitLB();
			cout << "Bn: " << Bn << endl;
			start = clock();
			//���ݹ۲�ȷ��������˳��
			if (i <= Na12UBRnum) {
				FindBnNa1 = false; FindBnNa2 = false; FindBnNa3 = false;
			}
			BnInNA = NextBnInNA;
			Round_NA1();
			Round_NA2();
			Round_NA3();
			UpdateLB();
			End = clock();
		}
		else {
			FindBn = false;
			start = clock();
			while (!FindBn) {
				cout << "Bn: " << Bn << endl;
				initial_AllTrail();
				//���ݹ۲�ȷ��������˳��
				FindBnNa1 = false; FindBnNa2 = false; FindBnNa3 = false;
				Round_NA1();
				Round_NA2();
				Round_NA3();
				if (!FindBn) Bn += weight[2];
			}
			End = clock();
		}

		printf("Final Bn:%d\ntime: %fs, %fmin\n\n", Bn, ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);

		FileOutputTrail();

		fp = fopen(FILENAME, "a+");
		fprintf(fp, "BestBn:%d \nSearch Time: %f s, %f min \n", Bn, ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
		fclose(fp);

		RecordTotalTime += ((double)End - (double)start);

		BestB[i] = Bn;
		int GenBnNPRnum;
		if (i < RNUM) {
			if (i <= Na12UBRnum) {
				if (!FindBnNa1 && !FindBnNa2 && !FindBnNa3) {
					//����û֪��������չ�ľ������ŵ�
					if (NextBnInNA == 0) FindBnNa1 = true;
					else if (NextBnInNA == 1) FindBnNa2 = true;
					else  FindBnNa3 = true;
				}
			}
			FindBn = false; GenBnTag = true;
			Rnum++;
			GenBnUP(0); 
			//cout << "TotalNxtBn:" << Bn << endl;
			//�洢���ż�
			if (GenBnDir) {
				//����õ���ֱ�Ӵ�				
				memcpy(GenBnBestTrail, BestTrail, Rnum * STATE_LEN);
				memcpy(GenBnBestw, Best_w, Rnum * sizeof(int));
				GenBnNPRnum = BNPRnum;
			}
			else {
				//����õ���Ҫ��λ
				memcpy(&GenBnBestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
				memcpy(&GenBnBestw[1], Best_w, (Rnum - 1) * sizeof(int));
				memcpy(&GenBnBestTrail[0], &BestTrail[Rnum - 1], STATE_LEN);
				GenBnBestw[0] = Best_w[Rnum - 1];
				GenBnNPRnum = BNPRnum + 1;
			}
			NextBnInNA = (BnInNA < GenBnInNA) ? BnInNA : GenBnInNA;
			Rnum--; GenBnTag = false;
		}

		if (i <= Na12UBRnum) {
			int TmpBn = Bn;
			//�����������Ͻ�
			start = clock();
			if (!FindBnNa1) {
				if (i == 2) {
					Bn = NaLB[i][0];
					FindBn = false;
					//cout << endl << "Na1Rnum:" << i << " Bn:" << Bn;
					while (!FindBnNa1) {
						Round_NA1();
						if (i > 2) UpdateLB();
						if (!FindBnNa1) Bn += weight[2];
					}
					//cout << " to " << Bn << endl;
				}
				else {
					Bn = Na1PreNxtBn;
					memcpy(BestTrail, GenBnNa1BestTrail, Rnum * STATE_LEN);
					memcpy(Best_w, GenBnNa1Bestw, Rnum * sizeof(int));
					BNPRnum = GenBnNa1NPRnum;
					FindBn = true;
					//cout << endl << "Na1Rnum:" << i << " Bn:" << Bn;
					Round_NA1();
					UpdateLB();
					//cout << " to " << Bn << endl;
				}				
				if (i < Na12UBRnum) {
					FindBn = false; GenBnTag = true;
					Rnum++;
					GenBnUP(1);
					if (GenBnDir) {
						//����õ���ֱ�Ӵ�				
						memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa1NPRnum = BNPRnum;
					}
					else {
						//����õ���Ҫ��λ
						memcpy(&GenBnNa1BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
						memcpy(&GenBnNa1Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
						memcpy(GenBnNa1BestTrail[0], BestTrail[Rnum - 1], STATE_LEN);
						GenBnNa1Bestw[0] = Best_w[Rnum - 1];
						GenBnNa1NPRnum = BNPRnum + 1;
					}
					Rnum--;
					Na1PreNxtBn = Bn; 
					//cout << "Na1NxtBn:" << Na1PreNxtBn << endl;
					GenBnTag = false;
				}
			}
			else if (i < Na12UBRnum) {
				memcpy(BestTrail, GenBnNa1BestTrail, Rnum* STATE_LEN);
				memcpy(Best_w, GenBnNa1Bestw, Rnum * sizeof(int));
				BNPRnum = GenBnNa1NPRnum;
				Bn = Na1PreNxtBn;
				FindBn = false; GenBnTag = true;
				Rnum++;
				GenBnUP(1);
				if (GenBnDir) {
					//����õ���ֱ�Ӵ�				
					memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa1NPRnum = BNPRnum;
				}
				else {
					//����õ���Ҫ��λ
					memcpy(&GenBnNa1BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
					memcpy(&GenBnNa1Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
					memcpy(&GenBnNa1BestTrail[0], &BestTrail[Rnum - 1], STATE_LEN);
					GenBnNa1Bestw[0] = Best_w[Rnum - 1];
					GenBnNa1NPRnum = BNPRnum + 1;
				}
				Rnum--;
				Na1PreNxtBn = Bn; 
				//cout << "Na1NxtBn:" << Na1PreNxtBn << endl;
				GenBnTag = false;
			}
			if (!FindBnNa2) {
				if (i == 2) {
					Bn = NaLB[i][1];
					FindBn = false;
					//cout << "Na2Rnum:" << i << " Bn:" << Bn;
					while (!FindBnNa2) {
						Round_NA2();
						if (i > 2) UpdateLB();
						if (!FindBnNa1) Bn += weight[2];
					}
					//cout << " to " << Bn << endl;
				}
				else {
					Bn = Na2PreNxtBn;
					FindBn = true;
					//cout << "Na2Rnum:" << i << " Bn:" << Bn;
					memcpy(BestTrail, GenBnNa2BestTrail, Rnum* STATE_LEN);
					memcpy(Best_w, GenBnNa2Bestw, Rnum * sizeof(int));
					BNPRnum = GenBnNa2NPRnum;
					Round_NA2();
					UpdateLB();
					//cout << " to " << Bn << endl;
				}
				
				if (i < Na12UBRnum) {
					FindBn = false; GenBnTag = true;
					Rnum++;
					GenBnUP(2);
					if (GenBnDir) {
						//����õ���ֱ�Ӵ�				
						memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa2NPRnum = BNPRnum;
					}
					else {
						//����õ���Ҫ��λ
						memcpy(&GenBnNa2BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
						memcpy(&GenBnNa2Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
						memcpy(&GenBnNa2BestTrail[0], &BestTrail[Rnum - 1], STATE_LEN);
						GenBnNa2Bestw[0] = Best_w[Rnum - 1];
						GenBnNa2NPRnum = BNPRnum + 1;
					}
					Rnum--;
					Na2PreNxtBn = Bn; 
					//cout << "Na2NxtBn:" << Na2PreNxtBn << endl;
					GenBnTag = false;
				}
			}
			else if (i < Na12UBRnum) {
				memcpy(BestTrail, GenBnNa2BestTrail, Rnum* STATE_LEN);
				memcpy(Best_w, GenBnNa2Bestw, Rnum * sizeof(int));
				BNPRnum = GenBnNa2NPRnum;
				Bn = Na2PreNxtBn;
				FindBn = false; GenBnTag = true;
				Rnum++;
				GenBnUP(2);
				if (GenBnDir) {
					//����õ���ֱ�Ӵ�				
					memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa2NPRnum = BNPRnum;
				}
				else {
					//����õ���Ҫ��λ
					memcpy(&GenBnNa2BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
					memcpy(&GenBnNa2Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
					memcpy(&GenBnNa2BestTrail[0], &BestTrail[Rnum - 1], STATE_LEN);
					GenBnNa2Bestw[0] = Best_w[Rnum - 1];
					GenBnNa2NPRnum = BNPRnum + 1;
				}
				Rnum--;
				Na2PreNxtBn = Bn; 
				//cout << "Na2NxtBn:" << Na2PreNxtBn << endl;
				GenBnTag = false; 
			}


			if (i <= Na3UBRnum) {
				Bn = NaLB[i][2]; FindBn = false;
				while (!FindBnNa3) {
					Bn += weight[2];
					Round_NA3();
				}
				//cout << "Na3Rnum:" << i << " :" << Bn << endl;
			}
			End = clock();
			printf("PreSearch time: %fs, %fmin\n", ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
			fp = fopen(FILENAME, "a+");
			fprintf(fp, "PreSearch time: %fs, %fmin\n", ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
			fclose(fp);
			RecordTotalTime += (double)End - (double)start;
			Bn = TmpBn;
		}

		initial_AllTrail();

		for (int kk = 1; kk >= 0; kk--) NaLB[Rnum][kk] = (NaLB[Rnum][kk] < NaLB[Rnum][kk + 1]) ? NaLB[Rnum][kk] : NaLB[Rnum][kk + 1];

		if (i < RNUM) {
			memcpy(BestTrail, GenBnBestTrail, (Rnum + 1) * STATE_LEN);
			memcpy(Best_w, GenBnBestw, (Rnum + 1) * sizeof(int));
			BNPRnum = GenBnNPRnum;
		}

		printf("Total time: %fs, %fmin\n\n", RecordTotalTime / CLOCKS_PER_SEC, (RecordTotalTime / CLOCKS_PER_SEC) / 60);
		fp = fopen(FILENAME, "a+");
		fprintf(fp, "\nTotal Time: %f s, %f min \n\n", (RecordTotalTime / CLOCKS_PER_SEC), (RecordTotalTime / CLOCKS_PER_SEC) / 60);
		fclose(fp);

	}
	e1 = clock();

	printf("B: ");
	fp = fopen(FILENAME, "a+");
	fprintf(fp, "BestB:\n");
	fclose(fp);
	for (int i = 1; i <= RNUM; i++) {
		printf("%d ", BestB[i]);
		fp = fopen(FILENAME, "a+");
		fprintf(fp, "%d, ", BestB[i]);
		fclose(fp);
	}

	return;
}

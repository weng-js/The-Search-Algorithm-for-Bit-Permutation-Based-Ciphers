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
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM][State_NUM];    //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //记录最优的结果，这个用来最后输出
int Best_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果，这个用来最后输出
int Tmp_Best_w[RNUM]; //临时标记

ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果，这个用来最后输出
int TmpNaBestw[RNUM]; //临时标记

ALIGNED_TYPE_(__m128i, 16) GenBnBestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果
int GenBnBestw[RNUM]; //临时标记->存储扩展得到的最优迹

ALIGNED_TYPE_(__m128i, 16) GenBnNa1BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果
int GenBnNa1Bestw[RNUM]; //临时标记->存储扩展得到的最优迹

ALIGNED_TYPE_(__m128i, 16) GenBnNa2BestTrail[RNUM][State_NUM]; //256bit 2*__m128i  384 3*__m128i 512 4*__m128i //用来临时记录最优记录最优的结果
int GenBnNa2Bestw[RNUM]; //临时标记->存储扩展得到的最优迹

map<int*, pair<__m128i**, int*>> NaBestTrailMap;

int NaIndex;             //当前正在搜索的子集
int NaBWIndex;           //对应的逆向标记

int FBWRound;      //正向/逆向深入的轮数
bool FBRoundOverTag;

bool FWSearchOver; //正向搜索到最后一轮，表示可以逆向搜索了
bool BWSearchOver; //逆向搜索到最后一轮，用于标记是否搜索到底
bool SubFindBn;    //用于更新子集重量上界

bool UpdateFW;  //判定是否需要更新正向表
bool UpdateBW;  //判定是否需要更新逆向表

int UpdateFWRoundNa1;
int UpdateFWRoundNa2;
int UpdateBWRoundNa1;
int UpdateBWRoundNa2;

//用于扩展得到条件最优
int BnInNA;	   //最优Bn所属的子集
int GenBnInNA; //扩展轮数的NA
bool GenBnTag; //目前是否在扩展？
bool GenBnDir; //正向扩展还是逆向扩展得到的最优
int GenBnNa1NPRnum;
int GenBnNa2NPRnum;

bool FindBnNa1;
bool FindBnNa2;
bool FindBnNa3;
int Na1PreNxtBn;
int Na2PreNxtBn;

int Rnum, NPRnum;
int BNPRnum;               //记录最优的起始搜索点

int Bn;   //整体的重量上界
int BWBn, FWBn; //逆向搜索的重量上界，保证特定输出返回的部分迹是当前输出最优的
int ODirWMin;

//用于搜索过程中更新NA2
int NA2_SBX1_VALUE;

int BestB[RNUM + 1] = { 0 };
bool FindBn;//找到最优

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
	strcat_s(tmpFILENAME, "RECTANGLE_Diff_Trail.txt");
#elif(TYPE==1)
	strcat_s(tmpFILENAME, "RECTANGLE_Linear_Trail.txt");
#endif

	// 计算对应的输入/输出，并给出对应自由端的其中一个值
	// BNPRnum    起始搜索点 
	
	//初始化
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	ALIGNED_TYPE_(__m128i, 16) PO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	memset(SO, 0, RNUM * STATE_LEN);
	memset(PO, 0, RNUM * STATE_LEN);
	memcpy(SO, BestTrail, (BNPRnum - 1) * STATE_LEN);
	memcpy(&PO[BNPRnum], &BestTrail[BNPRnum], (Rnum - BNPRnum) * STATE_LEN);

	//计算1~BNPRnum的输入差分
	for (int r = 0; r < BNPRnum - 1; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if ((SO[r][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //正向线性变换
					PO[r + 1][k] = _mm_xor_si128(PO[r + 1][k], PTable[i][SO[r][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	//计算BNPRnum~Rnum-1的输出差分
	for (int r = BNPRnum; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if ((PO[r][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //正向线性变换
					SO[r - 1][k] = _mm_xor_si128(SO[r - 1][k], INVPTable[i][PO[r][0].m128i_u8[i]][k]);
				}
			}
		}
	}
	//计算自由端
	for (int i = 0; i < 0x10; i++) {
		//第一轮的输入差分
		if (SO[0][0].m128i_u8[i]) {
			PO[0][0].m128i_u8[i] = BWWeightOrderIndex[SO[0][0].m128i_u8[i]][0];
		}
		//Rnum轮的输出差分
		if (PO[Rnum - 1][0].m128i_u8[i]) {
			SO[Rnum - 1][0].m128i_u8[i] = FWWeightOrderIndex[PO[Rnum - 1][0].m128i_u8[i]][0];
		}
	}
	//输出最优迹和重量
	tmpfp = fopen(tmpFILENAME, "a+");
	fprintf(tmpfp, "\nRNUM_%d:  Bn:%d NP:%d\n", Rnum, Bn, BNPRnum);

	for (int r = 0; r < Rnum; r++) {
		fprintf(tmpfp, "PO[%02d]: 0x", r+1);
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				fprintf(tmpfp, "%02x ", PO[r][s].m128i_u8[k]); //两位，不足用填充，输入差分
			}
			//fprintf(tmpfp, "  ");
		}
		fprintf(tmpfp, "\nSO[%02d]: 0x", r+1);
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				fprintf(tmpfp, "%02x ", SO[r][s].m128i_u8[k]); //两位，不足用填充，输出差分
			}
			//fprintf(tmpfp, "  ");
		}
		fprintf(tmpfp, "  w: %d\n\n", Best_w[r]);
	}

	fprintf(tmpfp, "\n\n");
	fclose(tmpfp);
}

inline void UpdateFWLBNa1(int i) {
	if (FBRoundOverTag) { //搜到底
		Na1FWMinW[FBWRound][i][1] = 1; // 标记该变量搜索到底
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
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //对应估计值都需要改变
		for (int k = FBWRound; k < r; k++) {
			Na1FWMinW[r][i][0] = (Na1FWMinW[r][i][0] > (Na1FWMinW[k][i][0] + NaLB[r - k][0]) ? Na1FWMinW[r][i][0] : (Na1FWMinW[k][i][0] + NaLB[r - k][0]));
	}
}
	UpdateFWRoundNa1 = (UpdateFWRoundNa1 < FBWRound) ? UpdateFWRoundNa1 : FBWRound;
}

inline void UpdateFWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) { //搜到底
		Na2FWMinW[FBWRound][i][sbox][1] = 1; // 标记该变量搜索到底
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

	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //对应估计值都需要进行更新
		for (int k = FBWRound; k < r; k++) {
			Na2FWMinW[r][i][sbox][0] = (Na2FWMinW[r][i][sbox][0] > (Na2FWMinW[k][i][sbox][0] + NaLB[r - k][1]) ? Na2FWMinW[r][i][sbox][0] : (Na2FWMinW[k][i][sbox][0] + NaLB[r - k][1]));
		}
	}

	UpdateFWRoundNa2 = (UpdateFWRoundNa2 < FBWRound) ? UpdateFWRoundNa2 : FBWRound;
}

inline void UpdateBWLBNa1(int i) {
	if (FBRoundOverTag) { //搜到底
		Na1BWMinW[FBWRound][i][1] = 1; // 标记该变量搜索到底

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
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //对应估计值都需要改变
		for (int k = FBWRound; k < r; k++) {
			Na1BWMinW[r][i][0] = (Na1BWMinW[r][i][0] > (Na1BWMinW[k][i][0] + NaLB[r - k][1]) ? Na1BWMinW[r][i][0] : (Na1BWMinW[k][i][0] + NaLB[r - k][1]));
		}
	}
	UpdateBWRoundNa1 = (UpdateBWRoundNa1 < FBWRound) ? UpdateBWRoundNa1 : FBWRound;
}

inline void UpdateBWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) { //搜到底
		Na2BWMinW[FBWRound][i][sbox][1] = 1; // 标记该变量搜索到底
	}

	if (FindBn || BWSearchOver) Na2BWMinW[FBWRound][i][sbox][0] = BWBn;
	else Na2BWMinW[FBWRound][i][sbox][0] = BWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { //对应估计值都需要改变
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
		//扩展
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
			//只逆向搜索-> 最后必然最优
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
			if (asn < NaBWIndex + 1) continue; //下一轮asn小于下界，返回
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

void FWRound_n(STATE s) { // NPRnum==1 才会进入该函数
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

			if (s.sbx_num == 0 && !GenBnTag) { //用已知的下界判断是否搜索
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
				//进入搜索，可以更新下界
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

void FWRound_1(STATE s, __m128i sbx_out[], Tree p) { //从第一轮开始搜索，用于NA3
	int asn; int i;
	for (i = 1; i < SBox_SIZE; i++) { //输入的都是活跃SBox，所以必须从1开始: 有可能搜索到底
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
	//NA1先整体判断，再搜索
	UpdateFWRoundNa1 = Rnum - 1; UpdateBWRoundNa1 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][0] > Bn)) || (FindBn && (NaLB[Rnum][0] >= Bn))) return;

	NaIndex = 0; NaBWIndex = 1;
	initial_Trail();
	SubFindBn = false;
	//搜素之前得判定子集的重量下界
	NPRnum = 1;      //都遍历: 只遍历输出
	UpdateFW = true; //只有正向搜索，都不会搜索到底，因此都需要进行一个更新，统一true
	STATE s(2, 0);   //从第二轮开始搜索
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	//clock_t na_s, na_e;
	//na_s = clock();
	if (Rnum == 2) { //直接根据对应输入和输出找最小即可
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
		for (int i = 0; i < NA1_NUM; i++) {  //只在这块更新正向LBArr，最后一轮即是最小重量
			if ((!FindBn && (Na1FWMinW[Rnum - 1][i][0] + Na1RoundNPInput[i][3] > Bn)) || (FindBn && (Na1FWMinW[Rnum - 1][i][0] + Na1RoundNPInput[i][3] >= Bn))) continue;
			FBRoundOverTag = false; FBWRound = Rnum - 1; ODirWMin = Na1RoundNPInput[i][3]; FWBn = Bn - ODirWMin;
			//记录下一轮输入的信息
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
		//判断分割是否允许接着往下搜索
		if ((!FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int i = 0; i < NA1_NUM; i++) { //遍历输出
			//当前输出是否可以接着往下搜索？
			if ((!FindBn && (Na1BWMinW[NPRnum - 1][i][0] + Na1FWOutLB[Rnum - NPRnum][i][0] > Bn))
				|| (FindBn && (Na1BWMinW[NPRnum - 1][i][0] + Na1FWOutLB[Rnum - NPRnum][i][0] >= Bn))) continue;
			BWSearchOver = false;
			if (NPRnum == 2) {
				//直接可以计算得到逆向最优
				BWSearchOver = true; BWBn = Na1BWMinW[NPRnum - 1][i][0];  //用于正向搜索 //包括NP轮的最小重量（用于更新正向表格），统一后续计算
				t_w[0] = Na1RoundNPBWInfo[i][1];
				memcpy(Trail[0], INVPTable[0][Na1RoundNPInput[i][0]], STATE_LEN);
			}
			else {
				//NPRnum > 2													//判断是否需要更新，如果对应输出对应轮数搜索到底则不用，否则，需要
				if (Na1BWMinW[NPRnum - 1][i][1]) {
					//之前搜索到底，故可以直接得到最优重量
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
					s.w = Na1RoundNPBWInfo[i][1];								//上一轮的最小重量
					memcpy(s.sbx_a, Na1RoundNPBWARRInfo[i][0], ARR_LEN);
					memcpy(s.sbx_in, Na1RoundNPBWARRInfo[i][1], ARR_LEN);
					s.sbx_num = Na1RoundNPBWInfo[i][0] - 1;
					s.rnum = NPRnum - 1; //实际上从上一轮开始
					memcpy(TmpBestTrail[NPRnum - 2], INVPTable[0][Na1RoundNPInput[i][0]], STATE_LEN);
					//正向搜索
					BWRound_i(s, sbx_out);

					UpdateBWLBNa1(i);
				}

			}

			//判断逆向搜索完，且重量仍在范围内，则正向搜索，还要判断正向几轮，如果是1/2轮，可以直接给出，而不用搜索和更新任意消息
			if (!BWSearchOver) continue;

			if (NPRnum == Rnum) continue;
			else if (NPRnum == Rnum - 1) {
				//可以直接根据表格得到最小的
				for (int Out = 0; Out < NA1_NUM; Out++) {
					if (Na1OutWeightOrder[i][Out] == INFINITY) break;
					if ((!FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1] > Bn))
						|| (FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1] >= Bn))) continue;
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1]; //整体重量
					memcpy(Best_w, t_w, NPRnum * sizeof(int));
					Best_w[NPRnum - 1] = Na1OutWeightOrder[i][Out];
					Best_w[NPRnum] = Na1RoundNPFWInfo[Na1InOutLink[i][Out]][1];
					memcpy(BestTrail, Trail, NPRnum * STATE_LEN);
					memcpy(BestTrail[NPRnum], PTable[0][Na1RoundNPInput[Na1InOutLink[i][Out]][0]], STATE_LEN); //输出差分
				}
			}
			else if (Na1FWOutLB[Rnum - NPRnum][i][1]) {
				//可以直接得到最优
				FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
				Bn = BWBn + Na1FWOutLB[Rnum - NPRnum][i][0]; //整体重量
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
						Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][0]; //整体重量
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
	//NA2先整体判断，再搜索
	UpdateFWRoundNa2 = Rnum - 1; UpdateBWRoundNa2 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][1] > Bn)) || (FindBn && (NaLB[Rnum][1] >= Bn))) return;
	SubFindBn = false;
	NaIndex = 1; NaBWIndex = 2;
	initial_Trail();
	NPRnum = 1;
	UpdateFW = true; //正向搜索，更新LBArr
	STATE s(2, 0);
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	//clock_t na_s, na_e;
	//na_s = clock();
	//Na2没有进入具体的搜索函数，需要标记活跃SBox的个数
	if (Rnum == 2) { //直接根据对应输入和输出找最小即可
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
				//用下界剪枝应该更快
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
		//判断分割是否允许接着往下搜索
		if ((!FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int sbox = 0; sbox < (SBox_NUM / 2); sbox++) {
			for (int i = 0; i < NA2_NUM; i++) { //遍历输出						
				if ((!FindBn && (Na2BWMinW[NPRnum - 1][i][sbox][0] + Na2FWOutLB[Rnum - NPRnum][i][sbox][0] > Bn))
					|| (FindBn && (Na2BWMinW[NPRnum - 1][i][sbox][0] + Na2FWOutLB[Rnum - NPRnum][i][sbox][0] >= Bn))) continue;
				BWSearchOver = false;
				if (NPRnum == 2) {
					//可以直接根据表格得到最小的
					BWSearchOver = true;
					BWBn = Na2BWMinW[NPRnum - 1][i][sbox][0];  //用于正向搜索 //包括NP轮的最小重量，统一后续计算
					t_w[0] = Na2RoundNPBWInfo[i][sbox][1];
					memcpy(Trail[0], Na2BWOutput[i][sbox], STATE_LEN);
				}
				else {
					//判断是否需要正向更新，如果对应输出对应轮数搜索到底则不用，否则，需要
					if (Na2BWMinW[NPRnum - 1][i][sbox][1]) {
						BWSearchOver = true;
						BWBn = Na2BWMinW[NPRnum - 1][i][sbox][0];  //用于正向搜索 //包括NP轮的最小重量，统一后续计算
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
						s.w = Na2RoundNPBWInfo[i][sbox][1];                            //下一轮的最小重量
						memcpy(s.sbx_a, Na2RoundNPBWARRInfo[i][sbox][0], ARR_LEN);
						memcpy(s.sbx_in, Na2RoundNPBWARRInfo[i][sbox][1], ARR_LEN);
						s.sbx_num = Na2RoundNPBWInfo[i][sbox][0] - 1;
						s.rnum = NPRnum - 1; //实际上从下一轮开始
						memcpy(TmpBestTrail[NPRnum - 2], Na2BWOutput[i][sbox], STATE_LEN);
						BWRound_i(s, sbx_out);

						UpdateBWLBNa2(i, sbox);
					}

				}

				//判断逆向搜索完，且重量仍在范围内，则正向搜索
				if (!BWSearchOver) continue;
				if (NPRnum == Rnum) continue;
				else if (NPRnum == Rnum - 1) { //直接可以计算得到逆向最优
					for (int Out = 0; Out < NA2_NUM; Out++) {
						if (Na2OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1] > Bn))
							|| (FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1] >= Bn))) continue;
						FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
						Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWInfo[Na2InOutLink[i][Out]][sbox][1]; //整体重量
						memcpy(Best_w, t_w, Rnum * sizeof(int));
						Best_w[NPRnum - 1] = Na2RoundNPFWInfo[Na1InOutLink[i][Out]][sbox][1];
						Best_w[NPRnum] = Na2OutWeightOrder[i][Out];
						memcpy(BestTrail, Trail, Rnum * STATE_LEN);
						memcpy(BestTrail[NPRnum], Na2FWOutput[Na2InOutLink[i][Out]][sbox], STATE_LEN);
					}
				}
				else if (Na2FWOutLB[Rnum - NPRnum][i][sbox][1]) {
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na2FWOutLB[Rnum - NPRnum][i][sbox][0]; //整体重量
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
							Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox][0]; //整体重量
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

void Round_NA3() { //反正不会从这搜索到
	//NA3先整体判断再搜索
	if ((!FindBn && (NaLB[Rnum][2] > Bn)) || (FindBn && (NaLB[Rnum][2] >= Bn))) return;
	NaIndex = 2; UpdateFW = false; SubFindBn = false;
	ODirWMin = 0; FWBn = Bn;
	STATE s(1, 0);
	initial_Trail();

	ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
	memset(tmp_out, 0, STATE_LEN);

	NPRnum = 1; //从第一轮开始搜索

	// 即使有了search pattern，还是需要记录活跃SBox，为了最后根据AS和index计算输出
	s.sbx_a[0] = 0; s.sbx_num = 2;
	while ((!FindBn && ((s.sbx_num + 1) * weight[1] + NaLB[Rnum - 1][2] <= Bn)) || (FindBn && ((s.sbx_num + 1) * weight[1] + NaLB[Rnum - 1][2] < Bn))) {
		s.w = (s.sbx_num + 1) * weight[1];
		FWRound_1(s, tmp_out, T); //最后一个是当前的活跃SBox
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
	//初始化 NaLB : 在搜索函数中更新：初始化为0，因此可以直接比较，省去第一行
	for (int i = 0; i < 3; i++) {
		NaLB[Rnum][i] = NaLB[1][i] + NaLB[Rnum - 1][i];
	}
	for (int r = 2; r < Rnum / 2; r++) {
		for (int i = 0; i < 3; i++) {
			NaLB[Rnum][i] = (NaLB[Rnum][i] > (NaLB[r][i] + NaLB[Rnum - r][i]) ? NaLB[Rnum][i] : (NaLB[r][i] + NaLB[Rnum - r][i]));
		}
	}
	//正向和逆向均更新
	//Na1:
	bool tag = true;
	for (int i = 0; i < NA1_NUM; i++) {
		//初始化均为0，因此可以直接开始比较
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

	//记录与输出兼容的最小逆向重量
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
			//初始化均为0，因此可以直接开始比较
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

	for (int i = 0; i < NA2_NUM; i++) { //关心总体的，后续再具体问题具体分析
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

void UpdateLB() { //搜索完之后，更新下界用，更新的是总体的那个
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
	//扩展r-1的最优特征，得到Bn的上界
	//先向前扩展
	NPRnum = BNPRnum;
	if (NaTag == 0) {
		//没有限制
		NaIndex = 0; NaBWIndex = 0; 
	}
	else if (NaTag == 1) {
		//子集1
		NaIndex = 0; NaBWIndex = 1; 
	}
	else {
		//子集2
		NaIndex = 1; NaBWIndex = 2;
	}
	ALIGNED_TYPE_(__m128i, 16) sbx_out[State_NUM]; memset(sbx_out, 0, STATE_LEN);
	ALIGNED_TYPE_(__m128i, 16) sbox_in1[State_NUM]; memset(sbox_in1, 0, STATE_LEN);
	ALIGNED_TYPE_(__m128i, 16) sbox_in2[State_NUM]; memset(sbox_in2, 0, STATE_LEN);
	if (NPRnum == Rnum - 1) {
		//需要先生成输入
		for (int i = 0; i < SBox_NUM; i++) {
			if ((BestTrail[Rnum - 3][0].m128i_u8[i])) {
				for (int k = 0; k < State_NUM; k++) { //正向线性变换
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
				for (int k = 0; k < State_NUM; k++) { //正向线性变换
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
	GenRound1Pattern(); //生成pattern不记在总时间里
	char FILENAME[50] = { 0 };
	clock_t s1, e1;

#if(TYPE==0)
	strcat_s(FILENAME, "RECTANGLE_Diff.txt");
#elif(TYPE==1)
	strcat_s(FILENAME, "RECTANGLE_Linear.txt");
#endif
	////验证一下链接没问题
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
			//根据观察确定的搜索顺序
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
				//根据观察确定的搜索顺序
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
					//若都没知道，则扩展的就是最优的
					if (NextBnInNA == 0) FindBnNa1 = true;
					else if (NextBnInNA == 1) FindBnNa2 = true;
					else  FindBnNa3 = true;
				}
			}
			FindBn = false; GenBnTag = true;
			Rnum++;
			GenBnUP(0); 
			//cout << "TotalNxtBn:" << Bn << endl;
			//存储最优迹
			if (GenBnDir) {
				//正向得到，直接存				
				memcpy(GenBnBestTrail, BestTrail, Rnum * STATE_LEN);
				memcpy(GenBnBestw, Best_w, Rnum * sizeof(int));
				GenBnNPRnum = BNPRnum;
			}
			else {
				//逆向得到需要移位
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
			//需求解最紧的上界
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
						//正向得到，直接存				
						memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa1NPRnum = BNPRnum;
					}
					else {
						//逆向得到需要移位
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
					//正向得到，直接存				
					memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa1NPRnum = BNPRnum;
				}
				else {
					//逆向得到需要移位
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
						//正向得到，直接存				
						memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa2NPRnum = BNPRnum;
					}
					else {
						//逆向得到需要移位
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
					//正向得到，直接存				
					memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa2NPRnum = BNPRnum;
				}
				else {
					//逆向得到需要移位
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

 #include<iostream>
#include<emmintrin.h>
#include<ctime>
#include<string>
#include<fstream>
#include<sstream>
#include<iomanip>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
#pragma comment(linker,"/STACK:1024000000,1024000000") 
using namespace std;

ALIGNED_TYPE_(__m128i, 16) Trail[RNUM];        
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM];   
ALIGNED_TYPE_(__m128i, 16) TmpBestTrail[RNUM];
ALIGNED_TYPE_(__m128i, 16) TmpNaBestTrail[RNUM];
ALIGNED_TYPE_(__m128i, 16) GenBnBestTrail[RNUM];
ALIGNED_TYPE_(__m128i, 16) GenBnNa1BestTrail[RNUM];
ALIGNED_TYPE_(__m128i, 16) GenBnNa2BestTrail[RNUM];

int t_w[RNUM];
int Best_w[RNUM];
int Tmp_Best_w[RNUM];
int TmpNaBestw[RNUM]; 
int GenBnBestw[RNUM]; 
int GenBnNa1Bestw[RNUM];
int GenBnNa2Bestw[RNUM]; 

map<bool*, pair < __m128i*, int* >> NaBestTrailMap;

int NaIndex;             //Index of the subset being searched
int NaBWIndex;          

int FBWRound;			//Maximum number of rounds searched in forward/reverse direction
bool FBRoundOverTag;

bool FWSearchOver; // Forward search whether to search to the last round
bool BWSearchOver; //Backward search whether to search to the first round
bool SubFindBn;   

bool UpdateFW;  //Determine if the  forward table needs to be updated
bool UpdateBW;  //Determine if the backward table needs to be updated

int UpdateFWRoundNa1;
int UpdateFWRoundNa2;
int UpdateBWRoundNa1;
int UpdateBWRoundNa2;

// Used to extend to obtain the conditional best trail and its weight
int BnInNA;	   
int GenBnInNA; 
bool GenBnTag; 
bool GenBnDir; 
int GenBnNa1NPRnum;
int GenBnNa2NPRnum;

//For pre-search
bool FindBnNa1;
bool FindBnNa2;
bool FindBnNa3;
int Na1PreNxtBn;
int Na2PreNxtBn;

int Rnum, NPRnum;
int BNPRnum;     

int Bn;			
int BWBn, FWBn; 
int ODirWMin;
int NA2_SBX1_VALUE;

int BestB[RNUM + 1] = { 0 };
bool FindBn;

void logToFile(const string& fileNameR, const string& message) {
	ofstream file(fileNameR, ios::app);
	if (!file.is_open()) {
		cerr << "Unable to open file: " << fileNameR << endl;
		return;
}
	file << message;
	file.close();
}

void FileOutputTrail() {
#if(TYPE==0)
	string fileName = "PRESENT_Diff_Trail.txt";
#elif(TYPE==1)
	string fileName = "PRESENT_Linear_Trail.txt";
#endif

	ALIGNED_TYPE_(__m128i, 16) SO[RNUM];       
	ALIGNED_TYPE_(__m128i, 16) PO[RNUM];        
	memset(SO, 0, RNUM * STATE_LEN);
	memset(PO, 0, RNUM * STATE_LEN);
	memcpy(SO, BestTrail, (BNPRnum - 1) * STATE_LEN);
	memcpy(&PO[BNPRnum], &BestTrail[BNPRnum], (Rnum - BNPRnum) * STATE_LEN);

	for (int r = 0; r < BNPRnum - 1; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (SO[r].m128i_u8[i]) {
				PO[r + 1] = _mm_xor_si128(PO[r + 1], PTable[i][SO[r].m128i_u8[i]]);
			}
		}
	}

	for (int r = BNPRnum; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (PO[r].m128i_u8[i]) {
				SO[r - 1] = _mm_xor_si128(SO[r - 1], INVPTable[i][PO[r].m128i_u8[i]]);
			}
		}
	}

	for (int i = 0; i < 0x10; i++) {
		if (SO[0].m128i_u8[i]) {
			PO[0].m128i_u8[i] = BWWeightOrderIndex[SO[0].m128i_u8[i]][0];
		}
		if (PO[Rnum - 1].m128i_u8[i]) {
			SO[Rnum - 1].m128i_u8[i] = FWWeightOrderIndex[PO[Rnum - 1].m128i_u8[i]][0];
		}
	}


	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Bn << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(PO[r].m128i_u8[k]);
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(SO[r].m128i_u8[k]);
		}
		message << "  w: " << dec << Best_w[r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());
}

inline void UpdateFWLBNa1(int i, int sbox) {
	if (FBRoundOverTag) { 
		Na1FWMinWOver[FBWRound][i][sbox] = true; 
		__m128i* NaBestTrail = new __m128i[FBWRound];
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0]    , TmpNaBestw    , FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na1FWMinWOver[FBWRound][i][sbox], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn) Na1FWMinW[FBWRound][i][sbox] = FWBn;
	else Na1FWMinW[FBWRound][i][sbox] = FWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { 
		for (int k = FBWRound; k < r; k++) {
			Na1FWMinW[r][i][sbox] = (Na1FWMinW[r][i][sbox] > (Na1FWMinW[k][i][sbox] + NaLB[r - k][0]) ?
				Na1FWMinW[r][i][sbox] : (Na1FWMinW[k][i][sbox] + NaLB[r - k][0]));
		}
	}
	UpdateFWRoundNa1 = (UpdateFWRoundNa1 < FBWRound) ? UpdateFWRoundNa1 : FBWRound;
}

inline void UpdateFWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) { 
		Na2FWMinWOver[FBWRound][i][sbox] = true;
		__m128i* NaBestTrail = new __m128i[FBWRound];
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na2FWMinWOver[FBWRound][i][sbox], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn) Na2FWMinW[FBWRound][i][sbox] = FWBn;
	else Na2FWMinW[FBWRound][i][sbox] = FWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { 
		for (int k = FBWRound; k < r; k++) {
			Na2FWMinW[r][i][sbox] = (Na2FWMinW[r][i][sbox] > (Na2FWMinW[k][i][sbox] + NaLB[r - k][1]) ? Na2FWMinW[r][i][sbox] : (Na2FWMinW[k][i][sbox] + NaLB[r - k][1]));
		}
	}
	UpdateFWRoundNa2 = (UpdateFWRoundNa2 < FBWRound) ? UpdateFWRoundNa2 : FBWRound;
}

inline void UpdateBWLBNa1(int i, int sbox) {
	if (FBRoundOverTag) { 
		Na1BWMinWOver[FBWRound][i][sbox] = true; 
		__m128i* NaBestTrail = new __m128i[FBWRound];
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na1BWMinWOver[FBWRound][i][sbox], make_pair(NaBestTrail, NaBestw)));
	}

	if (FindBn || BWSearchOver) Na1BWMinW[FBWRound][i][sbox] = BWBn;
	else  Na1BWMinW[FBWRound][i][sbox] = BWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { 
		for (int k = FBWRound; k < r; k++) {
			Na1BWMinW[r][i][sbox] = (Na1BWMinW[r][i][sbox] > (Na1BWMinW[k][i][sbox] + NaLB[r - k][1]) ? Na1BWMinW[r][i][sbox] : (Na1BWMinW[k][i][sbox] + NaLB[r - k][1]));
		}
	}

	UpdateBWRoundNa1 = (UpdateBWRoundNa1 < FBWRound) ? UpdateBWRoundNa1 : FBWRound;
}

inline void UpdateBWLBNa2(int i, int sbox) {
	if (FBRoundOverTag) {
		Na2BWMinWOver[FBWRound][i][sbox] = true; 
		__m128i* NaBestTrail = new __m128i[FBWRound];
		int* NaBestw = new int[FBWRound];
		memcpy(&NaBestTrail[0], TmpNaBestTrail, FBWRound * STATE_LEN);
		memcpy(&NaBestw[0], TmpNaBestw, FBWRound * sizeof(int));
		NaBestTrailMap.insert(make_pair(&Na2BWMinWOver[FBWRound][i][sbox], make_pair(NaBestTrail, NaBestw)));
	}
	
	if (FindBn || BWSearchOver) Na2BWMinW[FBWRound][i][sbox] = BWBn;
	else Na2BWMinW[FBWRound][i][sbox] = BWBn + 1;
	for (int r = FBWRound + 1; r <= Rnum - 1; r++) { 
		for (int k = FBWRound; k < r; k++) {
			Na2BWMinW[r][i][sbox] = (Na2BWMinW[r][i][sbox] > (Na2BWMinW[k][i][sbox] + NaLB[r - k][2]) ?
				Na2BWMinW[r][i][sbox] : (Na2BWMinW[k][i][sbox] + NaLB[r - k][2]));
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
		Bn = s.W;
		FindBn = true;
		BNPRnum = NPRnum;
		BestTrail[Rnum - 1] = TmpBestTrail[0];
		Best_w[0] = Tmp_Best_w[1];
		Best_w[Rnum - 1] = Tmp_Best_w[0];
		GenBnDir = false;
		GenBnInNA = s.sbx_num;
	}
	else {
		if (NPRnum == Rnum) {
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

void BWRound_i(STATE s, __m128i sbx_out) {
	int group_minw = 0; int record_sbx1_value;
	for (int i = 0; i < SBox_SIZE; i++) {
		if ((!FindBn && !BWSearchOver && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[s.rnum - 1][NaBWIndex] > BWBn))
			|| ((FindBn || BWSearchOver) && (s.W + s.w + BWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[s.rnum - 1][NaBWIndex] >= BWBn))) break;
		sbx_out = _mm_xor_si128(sbx_out, BWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i]);
		if (BWjudge_state_ri(s, BWWeightOrderW[s.sbx_in[s.j]][i], sbx_out, group_minw)) continue;
		if (s.j == s.sbx_num) {
			STATE nxt_s = BWupdate_state_row(s, BWWeightOrderW[s.sbx_in[s.j]][i], group_minw, sbx_out);
			if (nxt_s.sbx_num < NaBWIndex) continue;
			if (s.sbx_num == 1 && !GenBnTag) record_sbx1_value = NA2_SBX1_VALUE;
			if (s.rnum - 1 == 1) {
				BWRound_n(nxt_s);
			}
			else if ((!FindBn && !BWSearchOver && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[s.rnum - 3][NaBWIndex] <= BWBn))
				|| ((FindBn || BWSearchOver) && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[s.rnum - 3][NaBWIndex] < BWBn))) {
				__m128i tmp_out = _mm_setzero_si128();
				BWRound_i(nxt_s, tmp_out);
			}
			if (s.sbx_num == 1 && !GenBnTag) {
				if (INVSbox_loc[s.sbx_a[0]][1] < INVSbox_loc[s.sbx_a[1]][1]) {
					if (FindBn)  Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] > BWBn - nxt_s.W) ?
						Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] : BWBn - nxt_s.W;
					else  Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] > BWBn - nxt_s.W + 1) ?
						Na2BWMinW[s.rnum - 1][Na2InputIndex[record_sbx1_value][BWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[0]][1]][INVSbox_loc[s.sbx_a[1]][1]]] : BWBn - nxt_s.W + 1;
				}
				else {
					if (FindBn)  Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] > BWBn - nxt_s.W) ?
						Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] : BWBn - nxt_s.W;
					else  Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] =
						(Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] > BWBn - nxt_s.W + 1) ?
						Na2BWMinW[s.rnum - 1][Na2InputIndex[BWWeightOrderIndex[s.sbx_in[s.j]][i]][record_sbx1_value]][Na2SBoxInputIndex[INVSbox_loc[s.sbx_a[1]][1]][INVSbox_loc[s.sbx_a[0]][1]]] : BWBn - nxt_s.W + 1;
				}
			}

		}
		else {
			if (s.j == 0 && s.sbx_num == 1 && !GenBnTag) NA2_SBX1_VALUE = BWWeightOrderIndex[s.sbx_in[s.j]][i];
			BWRound_i(BWupdate_state_sbx(s, BWWeightOrderW[s.sbx_in[s.j]][i], group_minw), sbx_out);
		}
	}
	return;
}

void FWRound_n(STATE s) { 
	s.W            += s.w;
	t_w[s.rnum - 1] = s.w ;
	FWBn		  = s.W;
	Bn			  = s.W + ODirWMin;
	FindBn		  = true; SubFindBn = true; BNPRnum = NPRnum;
	if (UpdateFW) {
		FBRoundOverTag = true;
		memcpy(TmpNaBestTrail, &Trail[NPRnum], FBWRound * STATE_LEN);
		memcpy(TmpNaBestw    , &t_w[NPRnum], FBWRound * sizeof(int));
	}

	if (GenBnTag) {
		BestTrail[Rnum - 1] = Trail[Rnum - 1];
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

void FWRound_i(STATE s, __m128i sbx_out) {
	int group_minw = 0; int record_sbx1_value;
	for (int i = 0; i < SBox_SIZE; i++) {
		if ((!FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[Rnum - s.rnum][NaIndex] > FWBn))
			|| (FindBn && (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + NaLB[Rnum - s.rnum][NaIndex] >= FWBn))) break;
		sbx_out = _mm_xor_si128(sbx_out, FWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i]);
		if (FWjudge_state_ri(s, FWWeightOrderW[s.sbx_in[s.j]][i], sbx_out, group_minw)) continue;		
		if (s.j == s.sbx_num) {
			STATE nxt_s = FWupdate_state_row(s, FWWeightOrderW[s.sbx_in[s.j]][i], group_minw, sbx_out);
			if (nxt_s.sbx_num < NaIndex) continue; 
			if (s.sbx_num == 0 && !GenBnTag) { 
				if ((!FindBn && (nxt_s.W + Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[0]] > FWBn))
					|| (FindBn && (nxt_s.W + Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[0]] >= FWBn))) continue;
			}
			else if (s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) {
				record_sbx1_value = NA2_SBX1_VALUE;
				if ((!FindBn && (nxt_s.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[s.sbx_a[0]][s.sbx_a[1]]] > FWBn))
					|| (FindBn && (nxt_s.W + Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[s.sbx_a[0]][s.sbx_a[1]]] >= FWBn))) continue;
			}

			if (s.rnum + 1 == Rnum) {
				FWRound_n(nxt_s);
			}
			else if ((!FindBn && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[Rnum - s.rnum - 2][NaIndex] <= FWBn))
				|| (FindBn && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[Rnum - s.rnum - 2][NaIndex] < FWBn))) {
				__m128i tmp_out = _mm_setzero_si128();
				FWRound_i(nxt_s, tmp_out);
			}

			if (s.sbx_num == 0 && !GenBnTag) {
				if (FindBn)  Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[0]] = FWBn - nxt_s.W;
				else  Na1FWMinW[Rnum - s.rnum][Na1InputIndex[FWWeightOrderIndex[s.sbx_in[s.j]][i]]][s.sbx_a[0]] = FWBn - nxt_s.W + 1;
			}
			else if (s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) {
				if (FindBn)  Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[s.sbx_a[0]][s.sbx_a[1]]] = FWBn - nxt_s.W;
				else  Na2FWMinW[Rnum - s.rnum][Na2InputIndex[record_sbx1_value][FWWeightOrderIndex[s.sbx_in[s.j]][i]]][Na2SBoxInputIndex[s.sbx_a[0]][s.sbx_a[1]]] = FWBn - nxt_s.W + 1;
			}
		}
		else {
			if (s.j == 0 && s.sbx_num == 1 && NaIndex == 1 && !GenBnTag) NA2_SBX1_VALUE = FWWeightOrderIndex[s.sbx_in[s.j]][i];
			FWRound_i(FWupdate_state_sbx(s, FWWeightOrderW[s.sbx_in[s.j]][i], group_minw), sbx_out);
		}
	}
	return;
}

void FWRound_1(STATE s, __m128i sbx_out) {
	int group_minw = 0;
	for (int i = 1; i < SBox_SIZE; i++) { 
		if ((!FindBn && (s.w + Round1MinW[i] + NaLB[Rnum - 1][NaIndex] > FWBn)) || (FindBn && (s.w + Round1MinW[i] + NaLB[Rnum - 1][NaIndex] >= FWBn))) break;
		sbx_out = _mm_xor_si128(sbx_out, Round1MinSPTable[s.sbx_a[s.j]][i]);
		if (FWjudge_state_ri(s, Round1MinW[i], sbx_out, group_minw)) continue;

		if (s.j == s.sbx_num) {
			STATE nxt_s = FWupdate_state_row(s, Round1MinW[i], group_minw, sbx_out);
			if (nxt_s.sbx_num < NaIndex) continue;
			if (s.rnum + 1 == Rnum) {
				FWRound_n(nxt_s);
			}
			else if ((!FindBn && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[Rnum - s.rnum - 2][NaIndex] <= FWBn))
				|| (FindBn && (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + NaLB[Rnum - s.rnum - 2][NaIndex] < FWBn))) {
				__m128i tmp_out = _mm_setzero_si128();
				FWRound_i(nxt_s, tmp_out);
			}
		}
		else {
			FWRound_1(FWupdate_state_sbx(s, Round1MinW[i], group_minw), sbx_out);
		}
	}
	return;
}

void Round_NA1() { 
	UpdateFWRoundNa1 = Rnum - 1; UpdateBWRoundNa1 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][0] > Bn)) || (FindBn && (NaLB[Rnum][0] >= Bn))) return;
	NaIndex = 0; NaBWIndex = 1;
	initial_Trail();

	__m128i nxt_out = _mm_setzero_si128();
	SubFindBn = false;
	NPRnum = 1;      
	UpdateFW = true; 
	if (Rnum == 2) {
		for (int sbox = 0; sbox < SBox_NUM; sbox++) {
			for (int i = 0; i < NA1_NUM; i++) {
				if ((!FindBn && (Na1FWMinW[1][i][sbox] + Na1RoundNPInfo[i][1]) > Bn) || (FindBn && (Na1FWMinW[1][i][sbox] + Na1RoundNPInfo[i][1]) >= Bn)) continue;
				FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
				Bn = Na1FWMinW[1][i][sbox] + Na1RoundNPInfo[i][1];
				Best_w[0] = Na1RoundNPInfo[i][1];
				Best_w[1] = Na1FWMinW[1][i][sbox];
				BestTrail[1] = PTable[sbox][Na1RoundNPInput[i]];
			}
		}
	}
	else {
		for (int sbox = 0; sbox < SBox_NUM; sbox++) {
			for (int i = 0; i < NA1_NUM; i++) {  
				if ((!FindBn && Na1FWMinW[Rnum - 1][i][sbox] + Na1RoundNPInfo[i][1] > Bn) || (FindBn && Na1FWMinW[Rnum - 1][i][sbox] + Na1RoundNPInfo[i][1] >= Bn)) continue;
				FBRoundOverTag = false; FBWRound = Rnum - 1;  
				ODirWMin = Na1RoundNPInfo[i][1]; FWBn = Bn - ODirWMin;
				STATE s(2, 0);  
				s.w = Na1RoundNPFWMinW[i][sbox];
				memcpy(s.sbx_a, Na1RoundNPFWARRInfo[i][sbox][0], ARR_LEN / 2);
				memcpy(s.sbx_in, Na1RoundNPFWARRInfo[i][sbox][1], ARR_LEN / 2);
				memcpy(s.sbx_g, Na1RoundNPFWARRInfo[i][sbox][2], ARR_LEN / 2);
				s.sbx_num = Na1RoundNPFWASNandG[i][sbox][0] - 1;
				s.g_num = Na1RoundNPFWASNandG[i][sbox][1] - 1;
				t_w[0] = Na1RoundNPInfo[i][1];
				Trail[1] = PTable[sbox][Na1RoundNPInput[i]];

				FWRound_i(s, nxt_out);

				UpdateFWLBNa1(i, sbox);
			}
		}

	}

#if(TYPE)
	for (NPRnum = Rnum; NPRnum >= 2; NPRnum--) {
#else
	for (NPRnum = 2; NPRnum <= Rnum; NPRnum++) {
#endif
		if ((!FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na1BWLB[NPRnum - 1] + weight[1] + Na1FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int sbox = 0; sbox < SBox_NUM; sbox++) {
			for (int i = 0; i < NA1_NUM; i++) { 
				if ((!FindBn && (Na1BWMinW[NPRnum - 1][i][sbox] + Na1FWOutLB[Rnum - NPRnum][i][sbox] > Bn))
					|| (FindBn && (Na1BWMinW[NPRnum - 1][i][sbox] + Na1FWOutLB[Rnum - NPRnum][i][sbox] >= Bn))) continue;
				
				BWSearchOver = false;
				if (NPRnum == 2) {
					BWSearchOver = true; BWBn = Na1BWMinW[NPRnum - 1][i][sbox]; 					
					t_w[0] = Na1RoundNPBWMinW[i][sbox];
					Trail[0] = INVPTable[sbox][Na1RoundNPInput[i]];
				}
				else {
					if (Na1BWMinWOver[NPRnum - 1][i][sbox]) {
						BWSearchOver = true; BWBn = Na1BWMinW[NPRnum - 1][i][sbox];
						auto itor = NaBestTrailMap.find(&Na1BWMinWOver[NPRnum - 1][i][sbox]);
						if (itor != NaBestTrailMap.end()) {
							memcpy(Trail, &itor->second.first[0], (NPRnum - 1) * STATE_LEN);
							memcpy(t_w, &itor->second.second[0], (NPRnum - 1) * sizeof(int));
						}
					}
					else {
						UpdateBW = true; FBRoundOverTag = false; FBWRound = NPRnum - 1;
						ODirWMin = Na1FWOutLB[Rnum - NPRnum][i][sbox]; BWBn = Bn - ODirWMin;
						STATE s(NPRnum - 1, 0);
						s.w = Na1RoundNPBWMinW[i][sbox];								
						memcpy(s.sbx_a, Na1RoundNPBWARRInfo[i][sbox][0], ARR_LEN / 2);
						memcpy(s.sbx_in, Na1RoundNPBWARRInfo[i][sbox][1], ARR_LEN / 2);
						memcpy(s.sbx_g, Na1RoundNPBWARRInfo[i][sbox][2], ARR_LEN / 2);
						s.sbx_num = Na1RoundNPBWASNandG[i][sbox][0] - 1;
						s.g_num = Na1RoundNPBWASNandG[i][sbox][1] - 1;

						TmpBestTrail[NPRnum - 2] = INVPTable[sbox][Na1RoundNPInput[i]];

						BWRound_i(s, nxt_out);
						UpdateBWLBNa1(i, sbox);
					}

				}

				if (!BWSearchOver) continue;
				if (NPRnum == Rnum) continue;
				else if (NPRnum == Rnum - 1) {
					for (int Out = 0; Out < NA1_NUM; Out++) {
						if (Na1OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWMinW[Na1InOutLink[i][Out]][sbox] > Bn))
							|| (FindBn && (BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWMinW[Na1InOutLink[i][Out]][sbox] >= Bn))) continue;
						FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
						Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1RoundNPFWMinW[Na1InOutLink[i][Out]][sbox]; 
						memcpy(Best_w, t_w, NPRnum * sizeof(int));
						Best_w[NPRnum - 1] = Na1OutWeightOrder[i][Out];
						Best_w[NPRnum] = Na1RoundNPFWMinW[Na1InOutLink[i][Out]][sbox];
						memcpy(BestTrail, Trail, NPRnum* STATE_LEN);
						BestTrail[NPRnum] = PTable[sbox][Na1RoundNPInput[Na1InOutLink[i][Out]]];
					}
				}
				else if (Na1FWOutLBInfo[Rnum - NPRnum][i][sbox][0]) {
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na1FWOutLB[Rnum - NPRnum][i][sbox]; 
					auto itor = NaBestTrailMap.find(&Na1FWMinWOver[Rnum - NPRnum][Na1InOutLink[i][Na1FWOutLBInfo[Rnum - NPRnum][i][sbox][1]]][sbox]);
					if (itor != NaBestTrailMap.end()) {
						memcpy(&Trail[NPRnum], &itor->second.first[0], (Rnum - NPRnum) * STATE_LEN);
						memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
					}
					t_w[NPRnum - 1] = Na1OutWeightOrder[i][Na1FWOutLBInfo[Rnum - NPRnum][i][sbox][1]];
					memcpy(BestTrail, Trail, Rnum * STATE_LEN);
					memcpy(Best_w, t_w, Rnum * sizeof(int));
				}
				else {
					for (int Out = 0; Out < NA1_NUM; Out++) {
						if (Na1OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][sbox] > Bn)
							|| (FindBn && BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][sbox] >= Bn)) continue;
						if (Na1FWMinWOver[Rnum - NPRnum][Na1InOutLink[i][Out]][sbox]) {
							FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
							Bn = BWBn + Na1OutWeightOrder[i][Out] + Na1FWMinW[Rnum - NPRnum][Na1InOutLink[i][Out]][sbox];
							auto itor = NaBestTrailMap.find(&Na1FWMinWOver[Rnum - NPRnum][Na1InOutLink[i][Out]][sbox]);
							if (itor != NaBestTrailMap.end()) {
								memcpy(&Trail[NPRnum], &itor->second.first[0], (Rnum - NPRnum) * STATE_LEN);
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
							Trail[NPRnum] = PTable[sbox][Na1RoundNPInput[Na1InOutLink[i][Out]]];
							STATE s(NPRnum + 1, 0);
							s.w = Na1RoundNPFWMinW[Na1InOutLink[i][Out]][sbox];
							memcpy(s.sbx_a, Na1RoundNPFWARRInfo[Na1InOutLink[i][Out]][sbox][0], ARR_LEN / 2);
							memcpy(s.sbx_in, Na1RoundNPFWARRInfo[Na1InOutLink[i][Out]][sbox][1], ARR_LEN / 2);
							memcpy(s.sbx_g, Na1RoundNPFWARRInfo[Na1InOutLink[i][Out]][sbox][2], ARR_LEN / 2);
							s.sbx_num = Na1RoundNPFWASNandG[Na1InOutLink[i][Out]][sbox][0] - 1;
							s.g_num = Na1RoundNPFWASNandG[Na1InOutLink[i][Out]][sbox][1] - 1;

							FWRound_i(s, nxt_out);

							UpdateFWLBNa1(Na1InOutLink[i][Out], sbox);
						}
					}
				}
			}
		}
	}

	if (!FindBnNa1) FindBnNa1 = SubFindBn;	

	if (SubFindBn) {
		NaLB[Rnum][0] = Bn;
		if (Rnum < Na12UBRnum) {
			memcpy(GenBnNa1BestTrail, BestTrail, Rnum* STATE_LEN);
			memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
			GenBnNa1NPRnum = BNPRnum;
			Na1PreNxtBn = Bn;
		}
	}
	else if (FindBn) NaLB[Rnum][0] = (NaLB[Rnum][0] > Bn ? NaLB[Rnum][0] : Bn); 
	else NaLB[Rnum][0] = (NaLB[Rnum][0] > (Bn + 1) ? NaLB[Rnum][0] : (Bn + 1));
}

void Round_NA2() { 
	UpdateFWRoundNa2 = Rnum - 1; UpdateBWRoundNa2 = Rnum - 1;
	if ((!FindBn && (NaLB[Rnum][1] > Bn)) || (FindBn && (NaLB[Rnum][1] >= Bn))) return;

	SubFindBn = false;
	NaIndex = 1; NaBWIndex = 2;

	initial_Trail();

	__m128i nxt_out = _mm_setzero_si128();
	NPRnum = 1;
	UpdateFW = true; 

	if (Rnum == 2) {
		for (int sbox = 0; sbox < NA2_SBoxNUM; sbox++) {
			for (int i = 0; i < NA2_NUM; i++) {
				if ((!FindBn && (Na2FWMinW[1][i][sbox] + Na2RoundNPInfo[i][1]) > Bn) || (FindBn && (Na2FWMinW[1][i][sbox] + Na2RoundNPInfo[i][1]) >= Bn)) continue;
				FindBn = true;	SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
				Bn = Na2FWMinW[1][i][sbox] + Na2RoundNPInfo[i][1];
				Best_w[0] = Na2RoundNPInfo[i][1];
				Best_w[1] = Na2RoundNPFWMinW[i][sbox];
				BestTrail[1] = _mm_xor_si128(PTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[i][1]], PTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[i][2]]);
			}
		}
	}
	else {
		for (int sbox = 0; sbox < NA2_SBoxNUM; sbox++) {
			for (int i = 0; i < NA2_NUM; i++) {
				if ((!FindBn && Na2FWMinW[Rnum - 1][i][sbox] + Na2RoundNPInfo[i][1] > Bn) || (FindBn && Na2FWMinW[Rnum - 1][i][sbox] + Na2RoundNPInfo[i][1] >= Bn)) continue;
				FBRoundOverTag = false; FBWRound = Rnum - 1;
				ODirWMin = Na2RoundNPInfo[i][1]; FWBn = Bn - ODirWMin;
				STATE s(2, 0);
				s.w = Na2RoundNPFWMinW[i][sbox];
				memcpy(s.sbx_a, Na2RoundNPFWARRInfo[i][sbox][0], ARR_LEN / 2);
				memcpy(s.sbx_in, Na2RoundNPFWARRInfo[i][sbox][1], ARR_LEN / 2);
				memcpy(s.sbx_g, Na2RoundNPFWARRInfo[i][sbox][2], ARR_LEN / 2);
				s.sbx_num = Na2RoundNPFWASNandG[i][sbox][0] - 1;
				s.g_num = Na2RoundNPFWASNandG[i][sbox][1] - 1;

				t_w[0] = Na2RoundNPInfo[i][1];
				Trail[1] = _mm_xor_si128(PTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[i][1]], PTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[i][2]]);

				FWRound_i(s, nxt_out);

				UpdateFWLBNa2(i, sbox);
			}
		}
	}

#if(TYPE)
	for (NPRnum = Rnum; NPRnum >= 2; NPRnum--) {
#else
	for (NPRnum = 2; NPRnum <= Rnum; NPRnum++) {
#endif
		if ((!FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] > Bn)) || (FindBn && (Na2BWLB[NPRnum - 1] + 2 * weight[1] + Na2FWLB[Rnum - NPRnum] >= Bn))) continue;
		for (int sbox = 0; sbox < NA2_SBoxNUM; sbox++) {
			for (int i = 0; i < NA2_NUM; i++) { 	
				if ((!FindBn && (Na2BWMinW[NPRnum - 1][i][sbox] + Na2FWOutLB[Rnum - NPRnum][i][sbox] > Bn))
					|| (FindBn && (Na2BWMinW[NPRnum - 1][i][sbox] + Na2FWOutLB[Rnum - NPRnum][i][sbox] >= Bn))) continue;
				BWSearchOver = false;
				if (NPRnum == 2) {
					BWSearchOver = true; BWBn = Na2BWMinW[NPRnum - 1][i][sbox];
					t_w[0] = Na2RoundNPBWMinW[i][sbox];
					Trail[0] = _mm_xor_si128(INVPTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[i][1]], INVPTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[i][2]]);
				}
				else {
					if (Na2BWMinWOver[NPRnum - 1][i][sbox]) {
						BWSearchOver = true; BWBn = Na2BWMinW[NPRnum - 1][i][sbox];
						auto itor = NaBestTrailMap.find(&Na2BWMinWOver[NPRnum - 1][i][sbox]);
						if (itor != NaBestTrailMap.end()) {
							memcpy(Trail, &itor->second.first[0], (NPRnum - 1) * STATE_LEN);
							memcpy(t_w, &itor->second.second[0], (NPRnum - 1) * sizeof(int));
						}
					}
					else {
						UpdateBW = true; FBRoundOverTag = false; FBWRound = NPRnum - 1;
						ODirWMin = Na2FWOutLB[Rnum - NPRnum][i][sbox]; BWBn = Bn - ODirWMin;
						STATE s(NPRnum - 1, 0);
						s.w = Na2RoundNPBWMinW[i][sbox];                           
						memcpy(s.sbx_a, Na2RoundNPBWARRInfo[i][sbox][0], ARR_LEN / 2);
						memcpy(s.sbx_in, Na2RoundNPBWARRInfo[i][sbox][1], ARR_LEN / 2);
						memcpy(s.sbx_g, Na2RoundNPBWARRInfo[i][sbox][2], ARR_LEN / 2);
						s.sbx_num = Na2RoundNPBWASNandG[i][sbox][0] - 1;
						s.g_num = Na2RoundNPBWASNandG[i][sbox][1] - 1;
						TmpBestTrail[NPRnum - 2] = _mm_xor_si128(INVPTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[i][1]], INVPTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[i][2]]);

						BWRound_i(s, nxt_out);

						UpdateBWLBNa2(i, sbox);

					}

				}

				if (!BWSearchOver) continue;
				if (NPRnum == Rnum) continue;
				else if (NPRnum == Rnum - 1) { 
					for (int Out = 0; Out < NA2_NUM; Out++) {
						if (Na2OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWMinW[Na2InOutLink[i][Out]][sbox] > Bn))
							|| (FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWMinW[Na2InOutLink[i][Out]][sbox] >= Bn))) continue;
						FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
						Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2RoundNPFWMinW[Na2InOutLink[i][Out]][sbox];
						memcpy(Best_w, t_w, Rnum * sizeof(int));
						Best_w[NPRnum - 1] = Na2RoundNPFWMinW[Na1InOutLink[i][Out]][sbox];
						Best_w[NPRnum] = Na2OutWeightOrder[i][Out];
						memcpy(BestTrail, Trail, Rnum * STATE_LEN);
						BestTrail[NPRnum] = _mm_xor_si128(PTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[Na2InOutLink[i][Out]][1]], PTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[Na2InOutLink[i][Out]][2]]);
					}
				}
				else if (Na2FWOutLBInfo[Rnum - NPRnum][i][sbox][0]) {
					FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
					Bn = BWBn + Na2FWOutLB[Rnum - NPRnum][i][sbox]; 
					auto itor = NaBestTrailMap.find(&Na2FWMinWOver[Rnum - NPRnum][Na2InOutLink[i][Na2FWOutLBInfo[Rnum - NPRnum][i][sbox][1]]][sbox]);
					if (itor != NaBestTrailMap.end()) {
						memcpy(&Trail[NPRnum], &itor->second.first[0], (Rnum - NPRnum) * STATE_LEN);
						memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
					}
					t_w[NPRnum - 1] = Na2OutWeightOrder[i][Na2FWOutLBInfo[Rnum - NPRnum][i][sbox][1]];
					memcpy(BestTrail, Trail, Rnum * STATE_LEN);
					memcpy(Best_w, t_w, Rnum * sizeof(int));
				}
				else {
					for (int Out = 0; Out < NA2_NUM; Out++) {
						if (Na2OutWeightOrder[i][Out] == INFINITY) break;
						if ((!FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox] > Bn))
							|| (FindBn && (BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox] >= Bn))) continue;
						if (Na2FWMinWOver[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox]) {
							FindBn = true; SubFindBn = true; BNPRnum = NPRnum; BnInNA = NaIndex;
							Bn = BWBn + Na2OutWeightOrder[i][Out] + Na2FWMinW[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox];
							auto itor = NaBestTrailMap.find(&Na2FWMinWOver[Rnum - NPRnum][Na2InOutLink[i][Out]][sbox]);
							if (itor != NaBestTrailMap.end()) {
								memcpy(&Trail[NPRnum], &itor->second.first[0], (Rnum - NPRnum) * STATE_LEN);
								memcpy(&t_w[NPRnum], &itor->second.second[0], (Rnum - NPRnum) * sizeof(int));
							}
							t_w[NPRnum - 1] = Na2OutWeightOrder[i][Out];
							memcpy(BestTrail, Trail, Rnum * STATE_LEN);
							memcpy(Best_w, t_w, Rnum * sizeof(int));
						}
						else {
							UpdateFW = true; FBRoundOverTag = false; FBWRound = Rnum - NPRnum;
							ODirWMin = BWBn + Na2OutWeightOrder[i][Out]; FWBn = Bn - ODirWMin;
							t_w[NPRnum - 1] = Na2OutWeightOrder[i][Out];
							Trail[NPRnum] = _mm_xor_si128(PTable[Na2SBoxIndex[sbox][0]][Na2RoundNPInput[Na2InOutLink[i][Out]][1]], PTable[Na2SBoxIndex[sbox][1]][Na2RoundNPInput[Na2InOutLink[i][Out]][2]]);
							STATE s(NPRnum + 1, 0);
							s.w = Na2RoundNPFWMinW[Na2InOutLink[i][Out]][sbox];
							memcpy(s.sbx_a, Na2RoundNPFWARRInfo[Na2InOutLink[i][Out]][sbox][0], ARR_LEN / 2);
							memcpy(s.sbx_in, Na2RoundNPFWARRInfo[Na2InOutLink[i][Out]][sbox][1], ARR_LEN / 2);
							memcpy(s.sbx_g, Na2RoundNPFWARRInfo[Na2InOutLink[i][Out]][sbox][2], ARR_LEN / 2);
							s.sbx_num = Na2RoundNPFWASNandG[Na2InOutLink[i][Out]][sbox][0] - 1;
							s.g_num = Na2RoundNPFWASNandG[Na2InOutLink[i][Out]][sbox][1] - 1;

							FWRound_i(s, nxt_out);

							UpdateFWLBNa2(Na2InOutLink[i][Out], sbox);
						}
					}
				}
			}
		}
		
	}

	if(!FindBnNa2)FindBnNa2 = SubFindBn;

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

void R1ASNPattern(int index, STATE s, int g_num) {
	int tmp_g_num; int lb;
	if (index) lb = s.sbx_a[index - 1] + 1; else lb = 0;
	for (; lb < SBox_NUM - (s.sbx_num - index); lb++) {
		s.sbx_a[index] = lb; s.sbx_g[index] = Sbox_loc[lb];
		if (index && s.sbx_g[index] != s.sbx_g[index - 1]) tmp_g_num = g_num + 1;
		else tmp_g_num = g_num;
		if (index == s.sbx_num) {
			s.g_num = tmp_g_num;
			if ((!FindBn && (s.w + (s.g_num + 1) * weight[1] + NaLB[Rnum - 2][2] <= Bn)) || (FindBn && (s.w + (s.g_num + 1) * weight[1] + NaLB[Rnum - 2][2] <= Bn))) {
				__m128i tmp_out = _mm_setzero_si128();
				FWRound_1(s, tmp_out); 
			}
		}
		else {
			R1ASNPattern(index + 1, s, tmp_g_num);
		}
	}
}

void Round_NA3() { 
	if ((!FindBn && (NaLB[Rnum][2] > Bn)) || (FindBn && (NaLB[Rnum][2] >= Bn))) return;
	NaIndex = 2; SubFindBn = false; UpdateFW = false;
	ODirWMin = 0; FWBn = Bn;

	STATE s(1, 0);
	initial_Trail();

	NPRnum = 1;
	for (int asn = 2; asn < SBox_NUM; asn++) {
		s.w = (asn + 1) * weight[1];
		if ((!FindBn && (s.w + NaLB[Rnum - 1][2] <= Bn)) || (FindBn && (s.w + NaLB[Rnum - 1][2] <= Bn))) {
			s.sbx_num = asn;
			R1ASNPattern(0, s, 0);
		}
		else break;
	}

	if (!FindBnNa3) FindBnNa3 = SubFindBn;

	if (SubFindBn)   NaLB[Rnum][2] = Bn;
	else if (FindBn) NaLB[Rnum][2] = (NaLB[Rnum][2] > Bn ? NaLB[Rnum][2] : Bn); //!SubFindBn&&FindBn
	else NaLB[Rnum][2] = (NaLB[Rnum][2] > Bn + 1 ? NaLB[Rnum][2] : Bn + 1);
	return;
}


void InitLB() { 
	for (int i = 0; i < 3; i++) {
		NaLB[Rnum][i] = NaLB[1][i] + NaLB[Rnum - 1][i];
	}
	for (int r = 2; r < Rnum / 2; r++) {
		for (int i = 0; i < 3; i++) {
			NaLB[Rnum][i] = (NaLB[Rnum][i] > (NaLB[r][i] + NaLB[Rnum - r][i]) ? NaLB[Rnum][i] : (NaLB[r][i] + NaLB[Rnum - r][i]));
		}
	}
	//Na1:
	bool tag = true; int tmp_NALB = 0xffff;
	for (int s = 0; s < SBox_NUM; s++) {
		for (int i = 0; i < NA1_NUM; i++) {
			for (int r = 1; r < Rnum - 1; r++) {
				Na1BWMinW[Rnum - 1][i][s] = (Na1BWMinW[Rnum - 1][i][s] > (Na1BWMinW[r][i][s] + NaLB[Rnum - 1 - r][1]) ?
					Na1BWMinW[Rnum - 1][i][s] : (Na1BWMinW[r][i][s] + NaLB[Rnum - 1 - r][1]));
				Na1FWMinW[Rnum - 1][i][s] = (Na1FWMinW[Rnum - 1][i][s] > (Na1FWMinW[r][i][s] + NaLB[Rnum - 1 - r][0]) ?
					Na1FWMinW[Rnum - 1][i][s] : (Na1FWMinW[r][i][s] + NaLB[Rnum - 1 - r][0]));
			}

			if (tag) {
				Na1FWLB[Rnum - 1] = Na1FWMinW[Rnum - 1][i][s];
				Na1BWLB[Rnum - 1] = Na1BWMinW[Rnum - 1][i][s];
				tag = false;
			}
			else{
				if (Na1FWMinW[Rnum - 1][i][s] < Na1FWLB[Rnum - 1]) Na1FWLB[Rnum - 1] = Na1FWMinW[Rnum - 1][i][s];
				if (Na1BWMinW[Rnum - 1][i][s] < Na1BWLB[Rnum - 1]) Na1BWLB[Rnum - 1] = Na1BWMinW[Rnum - 1][i][s];
					
			}

			tmp_NALB = tmp_NALB < Na1FWMinW[Rnum - 1][i][s] + Na1RoundNPInfo[i][1] ? tmp_NALB : Na1FWMinW[Rnum - 1][i][s] + Na1RoundNPInfo[i][1];
		}
	}

	for (int s = 0; s < SBox_NUM; s++) {
		for (int i = 0; i < NA1_NUM; i++) {
			Na1FWOutLB[Rnum - 1][i][s] = Na1FWMinW[Rnum - 1][Na1InOutLink[i][0]][s] + Na1OutWeightOrder[i][0];
			if (Na1FWMinWOver[Rnum - 1][Na1InOutLink[i][0]][s]) Na1FWOutLBInfo[Rnum - 1][i][s][0] = 1;
			Na1FWOutLBInfo[Rnum - 1][i][s][1] = 0;
			for (int j = 1; j < NA1_NUM; j++) {
				if (Na1OutWeightOrder[i][j] == INFINITY) break;
				if (Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j] < Na1FWOutLB[Rnum - 1][i][s]) {
					Na1FWOutLB[Rnum - 1][i][s] = Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j];
					if (Na1FWMinWOver[Rnum - 1][Na1InOutLink[i][j]][s]) Na1FWOutLBInfo[Rnum - 1][i][s][0] = 1;
					else Na1FWOutLBInfo[Rnum - 1][i][s][0] = 0;
					Na1FWOutLBInfo[Rnum - 1][i][s][1] = j;
				}
				else if (Na1FWMinW[Rnum - 1][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j] == Na1FWOutLB[Rnum - 1][i][s]
					&& (!Na1FWOutLBInfo[Rnum - 1][i][s][0]) && Na1FWMinWOver[Rnum - 1][Na1InOutLink[i][j]][s]) {
					Na1FWOutLBInfo[Rnum - 1][i][s][0] = 1;
					Na1FWOutLBInfo[Rnum - 1][i][s][1] = j;
				}
			}
		}
	}	
	for (int r = 2; r <= Rnum - 1; r++) {
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				tmp_NALB = tmp_NALB < Na1BWMinW[r][i][s] + Na1FWOutLB[Rnum - 1 - r][i][s] ? tmp_NALB : Na1BWMinW[r][i][s] + Na1FWOutLB[Rnum - 1 - r][i][s];
			}
		}
	}
	if (tmp_NALB > NaLB[Rnum][0]) NaLB[Rnum][0] = tmp_NALB;

	//Na2:
	tag = true; tmp_NALB = 0xffff;
	for (int i = 0; i < NA2_NUM; i++) {
		for (int s = 0; s < NA2_SBoxNUM; s++) {
			for (int r = 1; r < Rnum - 1; r++) {
				Na2BWMinW[Rnum - 1][i][s] = (Na2BWMinW[Rnum - 1][i][s] > (Na2BWMinW[r][i][s] + NaLB[Rnum - 1 - r][2]) ?
					Na2BWMinW[Rnum - 1][i][s] : (Na2BWMinW[r][i][s] + NaLB[Rnum - 1 - r][2]));
				Na2FWMinW[Rnum - 1][i][s] = (Na2FWMinW[Rnum - 1][i][s] > (Na2FWMinW[r][i][s] + NaLB[Rnum - 1 - r][1]) ?
					Na2FWMinW[Rnum - 1][i][s] : (Na2FWMinW[r][i][s] + NaLB[Rnum - 1 - r][1]));
			}

			if (tag) {
				Na2FWLB[Rnum - 1] = Na2FWMinW[Rnum - 1][i][s];
				Na2BWLB[Rnum - 1] = Na2BWMinW[Rnum - 1][i][s];
				tag = false;
			}
			else {
				if (Na2FWMinW[Rnum - 1][i][s] < Na2FWLB[Rnum - 1]) Na2FWLB[Rnum - 1] = Na2FWMinW[Rnum - 1][i][s];
				if (Na2BWMinW[Rnum - 1][i][s] < Na2BWLB[Rnum - 1]) Na2BWLB[Rnum - 1] = Na2BWMinW[Rnum - 1][i][s];
			}

			tmp_NALB = tmp_NALB < Na2FWMinW[Rnum - 1][i][s] + Na2RoundNPInfo[i][1] ? tmp_NALB : Na2FWMinW[Rnum - 1][i][s] + Na2RoundNPInfo[i][1];
		}
	}

	for (int i = 0; i < NA2_NUM; i++) {
		for (int s = 0; s < NA2_SBoxNUM; s++) {
			tag = true;
			for (int j = 0; j < NA2_NUM; j++) {
				if (Na2OutWeightOrder[i][j] == INFINITY) break;
				if (tag) {
					Na2FWOutLB[Rnum - 1][i][s] = Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j];
					if (Na2FWMinWOver[Rnum - 1][Na2InOutLink[i][j]][s]) Na2FWOutLBInfo[Rnum - 1][i][s][0] = 1;
					Na2FWOutLBInfo[Rnum - 1][i][s][1] = j;
					tag = false;
				}
				else if (((Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[Rnum - 1][i][s])) {
					Na2FWOutLB[Rnum - 1][i][s] = Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j];
					if (Na2FWMinWOver[Rnum - 1][Na2InOutLink[i][j]][s]) Na2FWOutLBInfo[Rnum - 1][i][s][0] = 1;
					else Na2FWOutLBInfo[Rnum - 1][i][s][0] = 0;
					Na2FWOutLBInfo[Rnum - 1][i][s][1] = j;
				}
				else if (((Na2FWMinW[Rnum - 1][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]) == Na2FWOutLB[Rnum - 1][i][s])
					&& (!Na2FWOutLBInfo[Rnum - 1][i][s][0]) && (Na2FWMinWOver[Rnum - 1][Na2InOutLink[i][j]][s])) {
					Na2FWOutLBInfo[Rnum - 1][i][s][0] = 1;
					Na2FWOutLBInfo[Rnum - 1][i][s][1] = j;
				}					
			}
		}
	}

	for (int r = 2; r <= Rnum - 1; r++) {
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				tmp_NALB = tmp_NALB < Na2BWMinW[r][i][s] + Na2FWOutLB[Rnum - 1 - r][i][s] ? tmp_NALB : Na2BWMinW[r][i][s] + Na2FWOutLB[Rnum - 1 - r][i][s];
			}
		}
	}
	if (tmp_NALB > NaLB[Rnum][1]) NaLB[Rnum][1] = tmp_NALB;

	for (int i = 1; i >= 0; i--) NaLB[Rnum][i] = (NaLB[Rnum][i] < NaLB[Rnum][i + 1]) ? NaLB[Rnum][i] : NaLB[Rnum][i + 1];
}

void UpdateLB() { 
	bool tag1, tag2; int tmp_NALB , tmp_r; 
	//Na1:	
	for (int r = UpdateFWRoundNa1; r <= Rnum - 1; r++) {
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				if (i == 0 && s == 0) Na1FWLB[r] = Na1FWMinW[r][i][s];
				else if (Na1FWMinW[r][i][s] < Na1FWLB[r]) Na1FWLB[r] = Na1FWMinW[r][i][s];

				Na1FWOutLB[r][i][s] = Na1FWMinW[r][Na1InOutLink[i][0]][s] + Na1OutWeightOrder[i][0];
				if (Na1FWMinWOver[r][Na1InOutLink[i][0]][s]) Na1FWOutLBInfo[r][i][s][0] = 1;
				else Na1FWOutLBInfo[r][i][s][0] = 0;
				Na1FWOutLBInfo[r][i][s][1] = 0;

				for (int j = 1; j < NA1_NUM; j++) {
					if (Na1OutWeightOrder[i][j] == INFINITY) break;
					if (Na1FWMinW[r][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j] < Na1FWOutLB[r][i][s]) {
						Na1FWOutLB[r][i][s] = Na1FWMinW[r][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j];
						if (Na1FWMinWOver[r][Na1InOutLink[i][j]][s]) Na1FWOutLBInfo[r][i][s][0] = 1;
						else Na1FWOutLBInfo[r][i][s][0] = 0;
						Na1FWOutLBInfo[r][i][s][1] = j;
					}
					else if (Na1FWMinW[r][Na1InOutLink[i][j]][s] + Na1OutWeightOrder[i][j] == Na1FWOutLB[r][i][s]
						&& (!Na1FWOutLBInfo[r][i][s][0]) && Na1FWMinWOver[r][Na1InOutLink[i][j]][s]) {
						Na1FWOutLBInfo[r][i][s][0] = 1;
						Na1FWOutLBInfo[r][i][s][1] = j;
					}
				}
			}
		}		
	}
	for (int r = UpdateBWRoundNa1; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				if (tag1) {
					Na1BWLB[r] = Na1BWMinW[r][i][s];
					tag1 = false;
				}
				else if (Na1BWMinW[r][i][s] < Na1BWLB[r]) Na1BWLB[r] = Na1BWMinW[r][i][s];
			}
		}		
	}

	tmp_r = (UpdateBWRoundNa1 < UpdateFWRoundNa1) ? UpdateBWRoundNa1 : UpdateFWRoundNa1;
	for (int R = tmp_r; R < Rnum; R++) {
		tmp_NALB = 0xffff;
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				tmp_NALB = tmp_NALB < Na1FWMinW[R - 1][i][s] + Na1RoundNPInfo[i][1] ? tmp_NALB : Na1FWMinW[R - 1][i][s] + Na1RoundNPInfo[i][1];
			}
		}
		for (int r = 2; r <= R - 1; r++) {
			for (int s = 0; s < SBox_NUM; s++) {
				for (int i = 0; i < NA1_NUM; i++) {
					tmp_NALB = tmp_NALB < Na1BWMinW[r][i][s] + Na1FWOutLB[R - 1 - r][i][s] ? tmp_NALB : Na1BWMinW[r][i][s] + Na1FWOutLB[R - 1 - r][i][s];
				}
			}
		}
		if (tmp_NALB > NaLB[R][0]) NaLB[R][0] = tmp_NALB;
	}


	//Na2:
	for (int r = UpdateFWRoundNa2; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int i = 0; i < NA2_NUM; i++) {
			for (int s = 0; s < NA2_SBoxNUM; s++) {
				if (tag1) {
					Na2FWLB[r] = Na2FWMinW[r][i][s];
					tag1 = false;
				}
				else if (Na2FWMinW[r][i][s] < Na2FWLB[r]) Na2FWLB[r] = Na2FWMinW[r][i][s];

				tag2 = true;
				for (int j = 0; j < NA2_NUM; j++) {
					if (Na2OutWeightOrder[i][j] == INFINITY) break;
					if (tag2) {
						Na2FWOutLB[r][i][s] = Na2FWMinW[r][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j];
						if (Na2FWMinWOver[r][Na2InOutLink[i][j]][s]) Na2FWOutLBInfo[r][i][s][0] = 1;
						else Na2FWOutLBInfo[r][i][s][0] = 0;
						Na2FWOutLBInfo[r][i][s][1] = j;
						tag2 = false;
					}
					else if (((Na2FWMinW[r][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]) < Na2FWOutLB[r][i][s])) {
						Na2FWOutLB[r][i][s] = Na2FWMinW[r][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j];
						if (Na2FWMinWOver[r][Na2InOutLink[i][j]][s]) Na2FWOutLBInfo[r][i][s][0] = 1;
						else Na2FWOutLBInfo[r][i][s][0] = 0;
						Na2FWOutLBInfo[r][i][s][1] = j;
					}
					else if (((Na2FWMinW[r][Na2InOutLink[i][j]][s] + Na2OutWeightOrder[i][j]) == Na2FWOutLB[r][i][s])
						&& (!Na2FWOutLBInfo[r][i][s][0]) && (Na2FWMinWOver[r][Na2InOutLink[i][j]][s])) {
						Na2FWOutLBInfo[r][i][s][0] = 1;
						Na2FWOutLBInfo[r][i][s][1] = j;
					}
				}
			}
		}
	}

	for (int r = UpdateBWRoundNa2; r <= Rnum - 1; r++) {
		tag1 = true;
		for (int i = 0; i < NA2_NUM; i++) {
			for (int s = 0; s < NA2_SBoxNUM; s++) {
				if (tag1) {
					Na2BWLB[r] = Na2BWMinW[r][i][s];
					tag1 = false;
				}
				else if (Na2BWMinW[r][i][s] < Na2BWLB[r]) Na2BWLB[r] = Na2BWMinW[r][i][s];
			}
		}
	}

	tmp_r = (UpdateBWRoundNa2 < UpdateFWRoundNa2) ? UpdateBWRoundNa2 : UpdateFWRoundNa2;
	for (int R = tmp_r; R < Rnum; R++) {
		tmp_NALB = 0xffff;
		for (int s = 0; s < SBox_NUM; s++) {
			for (int i = 0; i < NA1_NUM; i++) {
				tmp_NALB = tmp_NALB < Na2FWMinW[R - 1][i][s] + Na2RoundNPInfo[i][1] ? tmp_NALB : Na2FWMinW[R - 1][i][s] + Na2RoundNPInfo[i][1];
			}
		}
		for (int r = 2; r <= R - 1; r++) {
			for (int s = 0; s < SBox_NUM; s++) {
				for (int i = 0; i < NA1_NUM; i++) {
					tmp_NALB = tmp_NALB < Na2BWMinW[r][i][s] + Na2FWOutLB[R - 1 - r][i][s] ? tmp_NALB : Na2BWMinW[r][i][s] + Na2FWOutLB[R - 1 - r][i][s];
				}
			}
		}
		if (tmp_NALB > NaLB[R][1]) NaLB[R][1] = tmp_NALB;
	}

	for (int i = 1; i >= 0; i--) NaLB[Rnum][i] = (NaLB[Rnum][i] < NaLB[Rnum][i + 1]) ? NaLB[Rnum][i] : NaLB[Rnum][i + 1];
}

void GenBnUP(int NaTag) {
	//Extending the best (r-1)-round trails yields an upper bound on Bn
	NPRnum = BNPRnum;
	if (NaTag == 0) {
		NaIndex = 0; NaBWIndex = 0; 
	}
	else if (NaTag == 1) {
		NaIndex = 0; NaBWIndex = 1; 
	}
	else {
		NaIndex = 1; NaBWIndex = 2; 
	}
	
	__m128i sbx_out = _mm_setzero_si128();
	__m128i sbox_in1 = _mm_setzero_si128();
	__m128i sbox_in2 = _mm_setzero_si128();

	if (NPRnum == Rnum - 1) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (BestTrail[Rnum - 3].m128i_u8[i]) {
				sbox_in1 = _mm_xor_si128(sbox_in1, PTable[i][BestTrail[Rnum - 3].m128i_u8[i]]);
			}
		}
	}
	else sbox_in1 = BestTrail[Rnum - 2];

	if (NPRnum == 1) {
		sbox_in2 = _mm_setzero_si128();
		for (int i = 0; i < SBox_NUM; i++) {
			if (BestTrail[1].m128i_u8[i]) {
				sbox_in2 = _mm_xor_si128(sbox_in2, INVPTable[i][BestTrail[1].m128i_u8[i]]);
			}
		}
	}
	else {
		sbox_in2 = BestTrail[0];
	}

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
#if(TYPE)
	string fileName = "PRESENT_Linear.txt";
#else	
	string fileName = "PRESENT_Diff.txt";
#endif

	stringstream message;
	message << "Pre-Seach Round: Na1 and Na2:" << Na12UBRnum << "  Na3:" << Na3UBRnum << endl;
	logToFile(fileName, message.str());

	BestB[1] = weight[1]; GenBnTag = false;
	int NextBnInNA; 
	double RecordTotalTime = 0;
	clock_t start, End;

	for (int i = 2; i <= RNUM; i++) {	
		Rnum = i;		
		if (i == 2) Bn = BestB[i - 1] + weight[1];

		cout << "Round NUM: " << dec << i << endl;
		message.str("");
		message << "RNUM_" << Rnum << " :\nBeginBn:" << Bn << endl;

		if (i > 2) {
			FindBn = true;
			InitLB();
			cout << "Bn: " << Bn << endl;
			start = clock();
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
				FindBnNa1 = false; FindBnNa2 = false; FindBnNa3 = false;
				Round_NA1();
				Round_NA2();
				Round_NA3();
				if (!FindBn) Bn += weight[WeightLen - 2];
			}
			End = clock();
		}

		printf("Final Bn:%f\ntime: %fs, %fmin\n", (double)Bn, ((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
		FileOutputTrail();

		message << "BestBn:" << Bn << "\nSearch Time: " << ((double)End - (double)start) / CLOCKS_PER_SEC << " s,  " << (((double)End - (double)start) / CLOCKS_PER_SEC) / 60 << " min\n";

		BestB[i] = Bn;
		RecordTotalTime += (double)End - (double)start;

		int GenBnNPRnum;
		if (i < RNUM) {
			if (i <= Na12UBRnum) {
				if (!FindBnNa1 && !FindBnNa2 && !FindBnNa3) {
					if (NextBnInNA == 0) FindBnNa1 = true;
					else if (NextBnInNA == 1) FindBnNa2 = true;
					else  FindBnNa3 = true;
				}
			}
			FindBn = false; GenBnTag = true;
			Rnum++;
			GenBnUP(0); 
			if (GenBnDir) {				
				memcpy(GenBnBestTrail, BestTrail, Rnum * STATE_LEN);
				memcpy(GenBnBestw, Best_w, Rnum * sizeof(int));
				GenBnNPRnum = BNPRnum;
			}
			else {
				memcpy(&GenBnBestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
				memcpy(&GenBnBestw[1], Best_w, (Rnum - 1) * sizeof(int));
				GenBnBestTrail[0] = BestTrail[Rnum - 1];
				GenBnBestw[0] = Best_w[Rnum - 1];
				GenBnNPRnum = BNPRnum + 1;
			}
			NextBnInNA = (BnInNA < GenBnInNA) ? BnInNA : GenBnInNA;
			Rnum--; GenBnTag = false;
		}

		if (i <= Na12UBRnum) {
			int TmpBn = Bn;
			start = clock();
			if (!FindBnNa1) {
				if (i == 2) {
					Bn = NaLB[i][0];
					FindBn = false;
					while (!FindBnNa1) {
						Round_NA1();
						if (!FindBnNa1) Bn += weight[WeightLen - 2];
					}
				}
				else {
					Bn = Na1PreNxtBn;
					memcpy(BestTrail, GenBnNa1BestTrail, Rnum * STATE_LEN);
					memcpy(Best_w, GenBnNa1Bestw, Rnum * sizeof(int));
					BNPRnum = GenBnNa1NPRnum;
					FindBn = true;
					Round_NA1();
					UpdateLB();
				}
				if (i < Na12UBRnum) {
					FindBn = false; GenBnTag = true;
					Rnum++;
					GenBnUP(1);
					if (GenBnDir) {		
						memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa1NPRnum = BNPRnum;
					}
					else {
						memcpy(&GenBnNa1BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
						memcpy(&GenBnNa1Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
						GenBnNa1BestTrail[0] = BestTrail[Rnum - 1];
						GenBnNa1Bestw[0] = Best_w[Rnum - 1];
						GenBnNa1NPRnum = BNPRnum + 1;
					}
					Rnum--;
					Na1PreNxtBn = Bn; 
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
					memcpy(GenBnNa1BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa1Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa1NPRnum = BNPRnum;
				}
				else {
					memcpy(&GenBnNa1BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
					memcpy(&GenBnNa1Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
					GenBnNa1BestTrail[0] = BestTrail[Rnum - 1];
					GenBnNa1Bestw[0] = Best_w[Rnum - 1];
					GenBnNa1NPRnum = BNPRnum + 1;
				}
				Rnum--;
				Na1PreNxtBn = Bn; 
				GenBnTag = false;
			}
			
			if (!FindBnNa2) {
				if (i == 2) {
					Bn = NaLB[i][1];
					FindBn = false;
					while (!FindBnNa2) {
						Round_NA2();
						if (i > 2) UpdateLB();
						if (!FindBnNa2) Bn += weight[WeightLen - 2];
					}
				}
				else {
					Bn = Na2PreNxtBn;
					FindBn = true;
					memcpy(BestTrail, GenBnNa2BestTrail, Rnum * STATE_LEN);
					memcpy(Best_w, GenBnNa2Bestw, Rnum * sizeof(int));
					BNPRnum = GenBnNa2NPRnum;
					Round_NA2();
					UpdateLB();
				}				
				if (i < Na12UBRnum) {
					FindBn = false; GenBnTag = true;
					Rnum++;
					GenBnUP(2);
					if (GenBnDir) {			
						memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
						memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
						GenBnNa2NPRnum = BNPRnum;
					}
					else {
						memcpy(&GenBnNa2BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
						memcpy(&GenBnNa2Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
						GenBnNa2BestTrail[0] = BestTrail[Rnum - 1];
						GenBnNa2Bestw[0] = Best_w[Rnum - 1];
						GenBnNa2NPRnum = BNPRnum + 1;
					}
					Rnum--;
					Na2PreNxtBn = Bn; 
					GenBnTag = false;
				}
			}
			else if(i < Na12UBRnum) {
				memcpy(BestTrail, GenBnNa2BestTrail, Rnum* STATE_LEN);
				memcpy(Best_w, GenBnNa2Bestw, Rnum * sizeof(int));
				BNPRnum = GenBnNa2NPRnum;
				Bn = Na2PreNxtBn;
				FindBn = false; GenBnTag = true;
				Rnum++;
				GenBnUP(2);
				if (GenBnDir) {		
					memcpy(GenBnNa2BestTrail, BestTrail, Rnum * STATE_LEN);
					memcpy(GenBnNa2Bestw, Best_w, Rnum * sizeof(int));
					GenBnNa2NPRnum = BNPRnum;
				}
				else {
					memcpy(&GenBnNa2BestTrail[1], BestTrail, (Rnum - 1) * STATE_LEN);
					memcpy(&GenBnNa2Bestw[1], Best_w, (Rnum - 1) * sizeof(int));
					GenBnNa2BestTrail[0] = BestTrail[Rnum - 1];
					GenBnNa2Bestw[0] = Best_w[Rnum - 1];
					GenBnNa2NPRnum = BNPRnum + 1;
				}
				Rnum--;
				Na2PreNxtBn = Bn;
				GenBnTag = false;
			}
			

			if (i <= Na3UBRnum) {
				Bn = NaLB[i][2]; FindBn = false;
				while (!FindBnNa3) {
					Bn += weight[WeightLen - 2];
					Round_NA3();
				}
			}
			End = clock();
			printf("PreSearch time: %fs, %fmin\n\n",((double)End - (double)start) / CLOCKS_PER_SEC, (((double)End - (double)start) / CLOCKS_PER_SEC) / 60);
			message << "PreSearch time: " << ((double)End - (double)start) / CLOCKS_PER_SEC << " s,  " << (((double)End - (double)start) / CLOCKS_PER_SEC) / 60 << " min\n\n";
			Bn = TmpBn;
			RecordTotalTime += (double)End - (double)start;
		}

		initial_AllTrail();

		for (int kk = 1; kk >= 0; kk--) NaLB[Rnum][kk] = (NaLB[Rnum][kk] < NaLB[Rnum][kk + 1]) ? NaLB[Rnum][kk] : NaLB[Rnum][kk + 1];
		if (i < RNUM) {
			memcpy(BestTrail, GenBnBestTrail, (Rnum + 1) * STATE_LEN);
			memcpy(Best_w, GenBnBestw, (Rnum + 1) * sizeof(int));
			BNPRnum = GenBnNPRnum;
		}

		printf("\nTotal Time: %fs, %fmin\n\n", (RecordTotalTime) / CLOCKS_PER_SEC, (RecordTotalTime / CLOCKS_PER_SEC) / 60);
		message << "\nTotal Time: " << (RecordTotalTime) / CLOCKS_PER_SEC << " s,  " << (RecordTotalTime / CLOCKS_PER_SEC) / 60 << " min\n\n";
		logToFile(fileName, message.str());
	}

	printf("B: ");
	message.str("");
	message << "BestB:\n";
	for (int i = 1; i <= RNUM; i++) {
		printf("%f ", (double)BestB[i]);
		message << BestB[i] << ", ";
	}
	logToFile(fileName, message.str());

	return;
}


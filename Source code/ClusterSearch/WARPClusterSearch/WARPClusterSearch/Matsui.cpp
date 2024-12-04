 #include<iostream>
#include<ctime>
#include<sstream>
#include<fstream>
#include<iomanip>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
#pragma comment(linker,"/STACK:1024000000,1024000000") 
using namespace std;

ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1];        
int t_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];    
int Best_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) TMPX[RNUM + 1]; 
int Extern2RMask[RNUM + 1];
ALIGNED_TYPE_(__m128i, 16) TMPX2R[RNUM + 1]; 
ALIGNED_TYPE_(__m128i, 16) TMPX2R_ASN[RNUM + 1]; 

#if(TYPE)
int BestB[RNUM + 1] = { 0,0,1,2,3,4,6,8,11,14,17,22,28,34,40,47,52,57,61,66,70,75,79,82,85,89,93,97,101,106,109,114,120,126,131,135,140,145,150,154,159,162 };
#else
int BestB[RNUM + 1] = { 0, 0, 2, 4, 6, 8, 12, 16, 22, 28, 34, 44, 56, 68, 80, 94, 104, 114, 122, 132, 140, 150, 158, 164, 170, 178, 186, 194, 202, 212, 218, 228, 240, 252, 262, 270, 280, 290, 300, 308, 318, 324 };
#endif
int BestASN[RNUM + 1] = { 0,0,1,2,3,4,6,8,11,14,17,22,28,34,40,47,52,57,61,66,70,75,79,82,85,89,93,97,101,106,109,114,120,126,131,135,140,145,150,154,159,162 };

__m128i TrailIState1;
__m128i TrailIState2;
__m128i TrailOState1;
__m128i TrailOState2;

int Rnum;
int Bn;
int BASN;
int Trail_Weight;
int RecordTrailNUM;
bool MergeTag;
bool OutputTrailTag;
bool MeetASP;

__m128i Mask = _mm_setzero_si128();
ALIGNED_TYPE_(__m128i, 16) MaskRound[RNUM + 1];
int MaskRoundBG;
int LastASN1, LastASN2;

void logToFile(const string& fileNameR, const string& message) {
	ofstream file(fileNameR, ios::app);
	if (!file.is_open()) {
		cerr << "Unable to open file: " << fileNameR << endl;
		return;
	}
	file << message;
	file.close();
}

class WeightForCluster { 
public:
	int Trail_NUM;
	WeightForCluster() : Trail_NUM(1) {};
	~WeightForCluster() {};

	void UpdateNUM() { Trail_NUM++; };
	void UpdateNUMPlus(int data) { Trail_NUM += data; }
	void InitNUM(int data) { Trail_NUM = data; }
};

map<int, WeightForCluster> RecordWeightTrailNUM;

class ValueorASPInfoForCluster {
public:
	int WeightUBorLB; 
	ValueorASPInfoForCluster(int data) :WeightUBorLB(data) {};
	~ValueorASPInfoForCluster() {};

	void UpdateWeightUBorLB(int data) {
		WeightUBorLB = data;
	}

};

int EndRoundForNo;
map<pair<pair<u64, u64>, int>, ValueorASPInfoForCluster> ValueNoMapCluster; 
map<pair<pair<u16, u16>, int>, ValueorASPInfoForCluster> ASPMapCluster_ASN;
map<pair<pair<u16, u16>, int>, ValueorASPInfoForCluster> ASPNoMapCluster_ASN;
map<pair<pair<pair<u16, u16>, pair<u8, u16>>, int>, ValueorASPInfoForCluster> ASPNoMapCluster_ASN_J;
map<pair<pair<pair<u16, u16>, pair<u8, u16>>, int>, ValueorASPInfoForCluster> ASPMapCluster_ASN_J;

class ValueForCluster {
public:
	int WeightUB; 
	map<int, int> WeightValueAndNUM;
	ValueForCluster(int dataUB) :WeightUB(dataUB) {
		WeightValueAndNUM.clear();
	}
	~ValueForCluster() {};

	bool SearchTag(int WeightForLink) {
		if (Bn - WeightForLink > WeightUB) return true; //Search: There may be more trails which satisfy the conditions
		else {
			for (auto itor = WeightValueAndNUM.begin(); itor != WeightValueAndNUM.end(); itor++) {
				if (WeightForLink + itor->first <= Bn) {
					auto it = RecordWeightTrailNUM.find(WeightForLink + itor->first);
					int tmp_Record = RecordTrailNUM;
					RecordTrailNUM += itor->second;
					if (it != RecordWeightTrailNUM.end()) it->second.UpdateNUMPlus(itor->second);
					else {
						WeightForCluster NewWeightRecord; NewWeightRecord.InitNUM(itor->second);
						RecordWeightTrailNUM.insert(make_pair(WeightForLink + itor->first, NewWeightRecord));
					}

				}
			}
			if (MergeTag) {
				for (auto it = RecordWeightTrailNUM.rbegin(); it != RecordWeightTrailNUM.rend(); it++) {
					int record = it->second.Trail_NUM; int recordValue = it->first; it->second.Trail_NUM = 0;
					while (record) {
						if (record % 2) {
							auto tmpIt = RecordWeightTrailNUM.find(recordValue);
							if (tmpIt != RecordWeightTrailNUM.end()) {
								tmpIt->second.UpdateNUM();
							}
							else {
								WeightForCluster NewWeightRecord;
								RecordWeightTrailNUM.insert(make_pair(recordValue, NewWeightRecord));
							}
						}
						record /= 2; recordValue--;
					}
				}
			}			
			return false;
		}
	}

	void ClearWeightValue() {
		WeightValueAndNUM.clear();
	}

	void InsertOrUpdate(int FindWeight, int TrailNUM) {
		WeightValueAndNUM[FindWeight] = TrailNUM;
		return;
	}

	void UpdateWeightUB(int data) { WeightUB = data; }
};


map<pair<pair<u64, u64>, int>, ValueForCluster> ValueMapCluster; 

void FileOutputTrail() {
#if(TYPE)	
	string fileName = "WARP_Linear_ClusterSearch_Trail.txt";
#else
	string fileName = "WARP_Diff_ClusterSearch_Trail.txt";
#endif

	
#if(TYPE)
	//Linear
	ALIGNED_TYPE_(__m128i, 16) SI[RNUM];
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM];
	memset(SI, 0, sizeof(SI));
	memset(SO, 0, sizeof(SO));
	for (int r = 0; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			SO[r].m128i_u8[i] = BestTrail[r + 1].m128i_u8[INVSBoxPermutation[i]];
		}
	}	

	__m128i tmpSI = _mm_setzero_si128();
	for (int i = 0; i < SBox_NUM; i++) {
		SI[Rnum - 1].m128i_u8[INVState2Permutation[i]] = SO[Rnum - 2].m128i_u8[i];
		tmpSI.m128i_u8[INVState2Permutation[i]] = TrailOState1.m128i_u8[i];
	}
	SI[Rnum - 1] = _mm_xor_si128(SI[Rnum - 1], TrailIState1);
	SI[0] = _mm_xor_si128(tmpSI, _mm_shuffle_epi8(SO[1], INVSBoxPermutationSSE));
	
	__m128i tmp;
	for (int r = 1; r < Rnum - 1; r++) {
		tmp = _mm_setzero_si128();
		for (int sbx_per = 0; sbx_per < SBox_NUM; sbx_per++) {
			tmp.m128i_u8[State1Permutation[sbx_per]] = BestTrail[r].m128i_u8[sbx_per];
		}
		SI[r] = _mm_xor_si128(BestTrail[r + 2], tmp);
	}

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Trail_Weight << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(SI[Rnum - 1 - r].m128i_u8[k]);
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(SO[Rnum - 1 - r].m128i_u8[k]);
		}
		message << "  w: " << dec << Best_w[Rnum - 1 - r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());
#else
	//Diff
	ALIGNED_TYPE_(__m128i, 16) SI[RNUM];
	ALIGNED_TYPE_(__m128i, 16) SO[RNUM];
	memset(SI, 0, sizeof(SI));
	memset(SO, 0, sizeof(SO));
	memcpy(SI, &BestTrail[1], Rnum * STATE_LEN);
	
	SO[0] = _mm_shuffle_epi8(_mm_xor_si128(_mm_shuffle_epi8(TrailIState2, SBoxPermutationSSE), BestTrail[2]), INVSBoxPermutationSSE);
	SO[Rnum - 1] = _mm_shuffle_epi8(_mm_xor_si128(TrailOState2, _mm_shuffle_epi8(BestTrail[Rnum - 1], State1PermutationSSE)), INVSBoxPermutationSSE);
	__m128i tmp;
	for (int r = 1; r < Rnum - 1; r++) {
		tmp = _mm_setzero_si128();
		for (int sbx_per = 0; sbx_per < SBox_NUM; sbx_per++) {
			tmp.m128i_u8[State1Permutation[sbx_per]] = SI[r - 1].m128i_u8[sbx_per];
		}
		tmp = _mm_xor_si128(SI[r + 1], tmp);
		for (int i = 0; i < SBox_NUM; i++) {
			if (tmp.m128i_u8[i]) {
				SO[r].m128i_u8[INVSBoxPermutation[i]] = tmp.m128i_u8[i];
			}
		}
	}

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Trail_Weight << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(SI[r].m128i_u8[k]);
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << static_cast<int>(SO[r].m128i_u8[k]);
		}
		message << "  w: " << dec << Best_w[r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());
#endif
}

inline void WeightTrailInfoReorganize() {
	for (auto it = RecordWeightTrailNUM.rbegin(); it != RecordWeightTrailNUM.rend(); it++) {
		int record = it->second.Trail_NUM; int recordValue = it->first; it->second.Trail_NUM = 0;
		while (record) {
			if (record % 2) {
				auto tmpIt = RecordWeightTrailNUM.find(recordValue);
				if (tmpIt != RecordWeightTrailNUM.end()) {
					tmpIt->second.UpdateNUM();
				}
				else {
					WeightForCluster NewWeightRecord;
					RecordWeightTrailNUM.insert(make_pair(recordValue, NewWeightRecord));
				}
			}
			record /= 2; recordValue--;
		}
	}
}

// the search for AS
void FWRound_i_ASN(STATE s, __m128i sbx_out) {	
	int tmp_sbxout = sbx_out.m128i_u8[s.sbx_a[s.j]];
	int sbx_nr_w = 0; int lb = 1;
	if (tmp_sbxout) lb = 0;          
	u8 tmp2R = TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]];
	for (int i = lb; i <= 1; i++) {
		sbx_out.m128i_u8[s.sbx_a[s.j]] = i; sbx_nr_w = i;
		if (sbx_nr_w && !s.sbx_tag[s.j]) {
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = 1;
		}
		else if (!s.sbx_tag[s.j]) {
			TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = 0;
		}
		else if (s.sbx_tag[s.j]) {
			sbx_nr_w = 0;
		}
		if (s.W + s.w + s.nr_sbx_num + sbx_nr_w + BestASN[Rnum - s.rnum - 1] > BASN) continue;

		if (s.rnum >= MaskRoundBG) {
			if (s.j == s.sbx_num || s.sbx_tag[s.j + 1]) {
				int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(sbx_out, MaskRound[s.rnum]), Mask)));
				if (asn) continue;
			}
			else {
				__m128i tmpsbxout = sbx_out;
				for (int sa = s.j + 1; sa <= s.sbx_num; sa++) {
					if (s.sbx_tag[sa]) break;
					else tmpsbxout.m128i_u8[s.sbx_a[sa]] = 0;
				}
				int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(tmpsbxout, MaskRound[s.rnum]), Mask)));
				if (asn) continue;
			}			
		}

		if (s.rnum + 1 == Rnum) {
			if (s.j == s.sbx_num) {
				int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(sbx_out, Mask)));
				if (asn != LastASN1) continue; 
				__m128i rl_sbxout = _mm_or_si128(_mm_shuffle_epi8(sbx_out, SBoxPermutationSSE), TMPX[s.rnum]);
				asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(rl_sbxout, MaskRound[0]), Mask)));
				if (asn != LastASN2) continue; 
				MeetASP = true; BASN = s.W + s.w + LastASN1; 
			}
			else {
				FWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				if (MeetASP) return;
			}
		}
		else {
			int tmp_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX2R_ASN[s.rnum], Mask));
			int asn = _mm_popcnt_u32(tmp_mask ^ Extern2RMask[s.rnum]);
			if (s.W + s.w + s.nr_sbx_num + sbx_nr_w + asn + BestASN[Rnum - s.rnum - 2] > BASN) continue;
			if (s.j == s.sbx_num) {
				Trail[s.rnum + 1] = sbx_out; 
				TMPX[s.rnum + 1] = _mm_shuffle_epi8(Trail[s.rnum + 1], State1PermutationSSE);
				Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
				STATE nxt_s = UpdateStateRoundI_ASN(s, sbx_nr_w);
				if (nxt_s.W + nxt_s.w + nxt_s.nr_sbx_num + BestASN[Rnum - nxt_s.rnum - 1] > BASN) continue;
				if (nxt_s.w) FWRound_i_ASN(nxt_s, TMPX[s.rnum]);
				else { 		
					if (nxt_s.rnum >= MaskRoundBG) {
						int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(TMPX[s.rnum], MaskRound[nxt_s.rnum]), Mask)));
						if (asn) continue;
					}
					Trail[nxt_s.rnum + 1] = TMPX[s.rnum]; 
					TMPX[nxt_s.rnum + 1] = _mm_shuffle_epi8(Trail[nxt_s.rnum + 1], State1PermutationSSE);
					Extern2RMask[nxt_s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[nxt_s.rnum + 1], Mask));
					STATE nxtnxt_s = UpdateStateRoundI_ASN(nxt_s, 0);
					if (nxtnxt_s.rnum == Rnum) {
						if (nxtnxt_s.w != LastASN1) continue; 
						__m128i rl_sbxout = _mm_or_si128(_mm_shuffle_epi8(TMPX[s.rnum], SBoxPermutationSSE), TMPX[nxt_s.rnum]);
						asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(rl_sbxout, MaskRound[0]), Mask)));
						if (asn != LastASN2) continue; 
						MeetASP = true; BASN = nxtnxt_s.W + nxtnxt_s.w;  
					}
					else if (nxtnxt_s.W + nxtnxt_s.w + nxtnxt_s.nr_sbx_num + BestASN[Rnum - nxtnxt_s.rnum - 1] <= BASN) {
						FWRound_i_ASN(nxtnxt_s, TMPX[nxt_s.rnum]);
					}
				}
				if (MeetASP) return;
			}
			else {
				FWRound_i_ASN(UpdateStateRoundI_j_ASN(s, sbx_nr_w), sbx_out);
				if (MeetASP) return;
			}
		}
	}
	TMPX2R_ASN[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R;
	return;
}

void FWRound_Last2Round(STATE s, __m128i sbx_out) {
	int tmp_w_record1 = 0;
	__m128i tmp_sbxin = _mm_shuffle_epi8(Trail[Rnum - 1], SBoxPermutationSSE);

	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail[Rnum].m128i_u8[i]) {
			if (tmp_sbxin.m128i_u8[i]) {
				if (DDTorLAT[tmp_sbxin.m128i_u8[i]][Trail[Rnum].m128i_u8[i] ^ sbx_out.m128i_u8[i]] != INFINITY)
					tmp_w_record1 += DDTorLATMinusMin[tmp_sbxin.m128i_u8[i]][Trail[Rnum].m128i_u8[i] ^ sbx_out.m128i_u8[i]];
				else return;
			}
			else if (sbx_out.m128i_u8[i] != Trail[Rnum].m128i_u8[i]) return;
		}
		else {
			if (DDTorLAT[tmp_sbxin.m128i_u8[i]][sbx_out.m128i_u8[i]] != INFINITY) {
				tmp_w_record1 += DDTorLATMinusMin[tmp_sbxin.m128i_u8[i]][sbx_out.m128i_u8[i]];
			}
			else return;
		}
	}
	t_w[Rnum - 2] = s.w + tmp_w_record1;
	int tmp_w_record2 = 0;
	tmp_sbxin = _mm_shuffle_epi8(Trail[Rnum], SBoxPermutationSSE);
#if(TYPE)
	for (int i = 0; i < SBox_NUM; i++) {
		if (TrailIState1.m128i_u8[i]) {
			if (tmp_sbxin.m128i_u8[i]) {
				if (DDTorLAT[tmp_sbxin.m128i_u8[i]][TrailIState1.m128i_u8[i] ^ TMPX[Rnum - 1].m128i_u8[i]] != INFINITY) {
					tmp_w_record2 += DDTorLAT[tmp_sbxin.m128i_u8[i]][TrailIState1.m128i_u8[i] ^ TMPX[Rnum - 1].m128i_u8[i]];
				}
				else return;
					
			}
			else if (TMPX[Rnum - 1].m128i_u8[i] != TrailIState1.m128i_u8[i]) return;
		}
		else {
			if (DDTorLAT[tmp_sbxin.m128i_u8[i]][TMPX[Rnum - 1].m128i_u8[i]] != INFINITY) {
				tmp_w_record2 += DDTorLAT[tmp_sbxin.m128i_u8[i]][TMPX[Rnum - 1].m128i_u8[i]];
			}
			else return;
		}
	}
#else
	for (int i = 0; i < SBox_NUM; i++) {
		if (TrailOState2.m128i_u8[i]) {
			if (tmp_sbxin.m128i_u8[i]) {
				if (DDTorLAT[tmp_sbxin.m128i_u8[i]][TrailOState2.m128i_u8[i] ^ TMPX[Rnum - 1].m128i_u8[i]] != INFINITY)
					tmp_w_record2 += DDTorLAT[tmp_sbxin.m128i_u8[i]][TrailOState2.m128i_u8[i] ^ TMPX[Rnum - 1].m128i_u8[i]];
				else return;
			}
			else if (TMPX[Rnum - 1].m128i_u8[i] != TrailOState2.m128i_u8[i]) return;
		}
		else {
			if (DDTorLAT[tmp_sbxin.m128i_u8[i]][TMPX[Rnum - 1].m128i_u8[i]] != INFINITY) {
				tmp_w_record2 += DDTorLAT[tmp_sbxin.m128i_u8[i]][TMPX[Rnum - 1].m128i_u8[i]];
			}
			else return;
		}
	}
#endif

	t_w[Rnum - 1] = tmp_w_record2;
	Trail_Weight = s.W + s.w + tmp_w_record1 + tmp_w_record2;
	RecordTrailNUM++;
	if (OutputTrailTag) {
		memcpy(BestTrail, Trail, (Rnum + 1) * STATE_LEN);
		memcpy(Best_w, t_w, Rnum * sizeof(int));
		FileOutputTrail();
	}
	auto itor = RecordWeightTrailNUM.find(Trail_Weight);
	if (itor != RecordWeightTrailNUM.end()) itor->second.UpdateNUM();
	else {
		WeightForCluster NewWeightRecord;
		RecordWeightTrailNUM.insert(make_pair(Trail_Weight, NewWeightRecord));
	}

	if(MergeTag) WeightTrailInfoReorganize();
	return;
}

void FWRound_i(STATE s, __m128i sbx_out) {
	int tmp_sbxout = sbx_out.m128i_u8[s.sbx_a[s.j]];
	int sbx_nr_w = 0, sbx_nr_num = 0; 
	u8 tmp2R = TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]];
	u16 tmp_asp1, tmp_asp2, tmp_searchOutput;
	if (tmp_sbxout) {
		if (s.j != s.sbx_num) {
			tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum - 1], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum], Mask));
			tmp_searchOutput = 0;
			for (int ASB_Output = 0; ASB_Output < s.j; ASB_Output++) if (sbx_out.m128i_u8[s.sbx_a[ASB_Output]]) tmp_searchOutput ^= (1 << ASB_Output);
		}
		if (s.W + s.w + DDTorLATMinusMin[s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw + BestB[Rnum - s.rnum - 1] > Bn) goto FWSearchForActiveOutput;
		sbx_out.m128i_u8[s.sbx_a[s.j]] = 0; TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = 0;
		if (s.j == s.sbx_num) {
			Trail[s.rnum + 1] = sbx_out;
			TMPX[s.rnum + 1] = _mm_shuffle_epi8(Trail[s.rnum + 1], State1PermutationSSE);
			Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
			STATE nxt_s = UpdateStateRoundI(s, DDTorLATMinusMin[s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w);
			if (nxt_s.W + nxt_s.w + nxt_s.nr_minw + BestB[Rnum - nxt_s.rnum - 1] > Bn) goto FWSearchForActiveOutput;
			if (nxt_s.rnum == Rnum - 1) { 
				FWRound_Last2Round(nxt_s, TMPX[s.rnum]); 
			}
			else if (nxt_s.w) {
				MeetASP = false;
				BASN = ((Bn - nxt_s.W - nxt_s.w - nxt_s.nr_minw) / weight[1]) + nxt_s.sbx_num + nxt_s.nr_sbx_num + 1;
				tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum + 1], Mask));
				auto itNo = ASPNoMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum));
				if (itNo != ASPNoMapCluster_ASN.end()) {
					if (BASN <= itNo->second.WeightUBorLB) goto FWSearchForActiveOutput; 
				}
				auto itYes = ASPMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum)); 
				bool SearchTag = true;
				if (itYes != ASPMapCluster_ASN.end()) {
					if (BASN >= itYes->second.WeightUBorLB) { MeetASP = true; SearchTag = false;  } 
				}				
				if (SearchTag) {
					STATE nxt_asn_s = GenStateRI_ASN(nxt_s);
					FWRound_i_ASN(nxt_asn_s, TMPX[s.rnum]);

					if (MeetASP) {
						if (itYes != ASPMapCluster_ASN.end()) {
							itYes->second.UpdateWeightUBorLB(BASN);
						}
						else {
							ValueorASPInfoForCluster new_info(BASN);
							ASPMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum), new_info));
						}
					}
					else {
						if (itNo != ASPNoMapCluster_ASN.end()) {
							itNo->second.UpdateWeightUBorLB(BASN);
						}
						else {
							ValueorASPInfoForCluster new_info(BASN);
							ASPNoMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum), new_info));
						}
					}					
				}	
				if (MeetASP) {
					if (s.rnum >= 4 && s.rnum <= (Rnum - 4)) {
						u64 tmp_value1 = (Trail[s.rnum].m128i_u64[1] << 4) ^ Trail[s.rnum].m128i_u64[0];
						u64 tmp_value2 = (Trail[s.rnum + 1].m128i_u64[1] << 4) ^ Trail[s.rnum + 1].m128i_u64[0];
						if (s.rnum <= (Rnum - EndRoundForNo)) {
							auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
							if (itorNo != ValueNoMapCluster.end() && (Bn - nxt_s.W <= itorNo->second.WeightUBorLB)) goto FWSearchForActiveOutput; 
						}						

						auto itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
						if (itorYes == ValueMapCluster.end() || itorYes->second.SearchTag(nxt_s.W)) {
							int Tmp_RecordTrailNUM = RecordTrailNUM; map<int, WeightForCluster> TmpRecordWeightTrailNUM = RecordWeightTrailNUM;
							FWRound_i(nxt_s, TMPX[s.rnum]);

							if (Tmp_RecordTrailNUM != RecordTrailNUM) {
								if (itorYes == ValueMapCluster.end()) {
									ValueForCluster NewValueForCluster(Bn - nxt_s.W);
									ValueMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum), NewValueForCluster));
									itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
								}
								else {
									itorYes->second.UpdateWeightUB(Bn - nxt_s.W);
								}
								if (MergeTag) {
									bool MinusTag = false; itorYes->second.ClearWeightValue();
									for (auto it1 = RecordWeightTrailNUM.rbegin(); it1 != RecordWeightTrailNUM.rend(); it1++) {
										if (it1->second.Trail_NUM && !MinusTag) {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
												itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
											}
										}
										else if (it1->second.Trail_NUM && MinusTag) {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
												MinusTag = false;
											}
											else {
												itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
											}
										}
										else if (!it1->second.Trail_NUM && MinusTag) {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
												itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
											}
										}
										else {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM) {
												itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
												MinusTag = true;
											}
										}
									}
								}
								else {
									for (auto it1 = RecordWeightTrailNUM.begin(); it1 != RecordWeightTrailNUM.end(); it1++) {
										auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
										if (it2 == TmpRecordWeightTrailNUM.end()) {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, it1->second.Trail_NUM);
										}
										else if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM < it1->second.Trail_NUM) {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, (it1->second.Trail_NUM - it2->second.Trail_NUM));
										}
									}
								}
							}
							else if(s.rnum <= (Rnum - EndRoundForNo)) {
								auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
								if (itorNo != ValueNoMapCluster.end()) {
									itorNo->second.UpdateWeightUBorLB(Bn - nxt_s.W);
								}
								else {
									ValueorASPInfoForCluster new_ValueNoForCluster(Bn - nxt_s.W);
									ValueNoMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum), new_ValueNoForCluster));
								}
							}
						}
					}
					else {
						FWRound_i(nxt_s, TMPX[s.rnum]);
					}
				}
			}
			else {
				Trail[nxt_s.rnum + 1] = TMPX[s.rnum];
				TMPX[nxt_s.rnum + 1] = _mm_shuffle_epi8(Trail[nxt_s.rnum + 1], State1PermutationSSE);
				Extern2RMask[nxt_s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[nxt_s.rnum + 1], Mask)); 
				STATE nxtnxt_s = UpdateStateRoundI(nxt_s, 0, 0);
				if (nxtnxt_s.rnum == Rnum - 1) {
					FWRound_Last2Round(nxtnxt_s, TMPX[nxt_s.rnum]);
				}
				else if (nxtnxt_s.W + nxtnxt_s.w + nxtnxt_s.nr_minw + BestB[Rnum - nxtnxt_s.rnum - 1] <= Bn) {
					MeetASP = false;
					BASN = ((Bn - nxtnxt_s.W - nxtnxt_s.w - nxtnxt_s.nr_minw) / weight[1]) + nxtnxt_s.sbx_num + nxtnxt_s.nr_sbx_num + 1;
					tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[nxt_s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[nxt_s.rnum + 1], Mask));
					auto itNo = ASPNoMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - nxt_s.rnum));
					if (itNo != ASPNoMapCluster_ASN.end()) {
						if (BASN <= itNo->second.WeightUBorLB) goto FWSearchForActiveOutput; 
					}
					auto itYes = ASPMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - nxt_s.rnum)); bool SearchTag = true;
					if (itYes != ASPMapCluster_ASN.end()) {
						if (BASN >= itYes->second.WeightUBorLB) { MeetASP = true; SearchTag = false; } 
					}
					if (SearchTag) {
						STATE nxt_asn_s = GenStateRI_ASN(nxtnxt_s);
						FWRound_i_ASN(nxt_asn_s, TMPX[nxt_asn_s.rnum - 1]);
						if (MeetASP) {
							if (itYes != ASPMapCluster_ASN.end()) {
								itYes->second.UpdateWeightUBorLB(BASN);
							}
							else {
								ValueorASPInfoForCluster new_info(BASN);
								ASPMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - nxt_s.rnum), new_info));
							}
						}
						else {
							if (itNo != ASPNoMapCluster_ASN.end()) {
								itNo->second.UpdateWeightUBorLB(BASN);
							}
							else {
								ValueorASPInfoForCluster new_info(BASN);
								ASPNoMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - nxt_s.rnum), new_info));
							}
						}
					}
					if (MeetASP) {
						if (nxt_s.rnum >= 4 && nxt_s.rnum <= (Rnum - 4)) {
							u64 tmp_value1 = (Trail[nxt_s.rnum].m128i_u64[1] << 4) ^ Trail[nxt_s.rnum].m128i_u64[0];
							u64 tmp_value2 = (Trail[nxt_s.rnum + 1].m128i_u64[1] << 4) ^ Trail[nxt_s.rnum + 1].m128i_u64[0];
							if (nxt_s.rnum <= (Rnum - EndRoundForNo)) {
								auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum));
								if (itorNo != ValueNoMapCluster.end() && (Bn - nxtnxt_s.W <= itorNo->second.WeightUBorLB)) goto FWSearchForActiveOutput; //²»´æÔÚ
							}				

							auto itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum));

							if (itorYes == ValueMapCluster.end() || itorYes->second.SearchTag(nxtnxt_s.W)) {
								int Tmp_RecordTrailNUM = RecordTrailNUM; map<int, WeightForCluster> TmpRecordWeightTrailNUM = RecordWeightTrailNUM;

								FWRound_i(nxtnxt_s, TMPX[nxtnxt_s.rnum - 1]);

								if (Tmp_RecordTrailNUM != RecordTrailNUM) {
									if (itorYes == ValueMapCluster.end()) {
										ValueForCluster NewValueForCluster(Bn - nxtnxt_s.W);
										ValueMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum), NewValueForCluster));
										itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum));
									}
									else {
										itorYes->second.UpdateWeightUB(Bn - nxtnxt_s.W);
									}
									if (MergeTag) {
										bool MinusTag = false; itorYes->second.ClearWeightValue();
										for (auto it1 = RecordWeightTrailNUM.rbegin(); it1 != RecordWeightTrailNUM.rend(); it1++) {
											if (it1->second.Trail_NUM && !MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, 1);
												}
											}
											else if (it1->second.Trail_NUM && MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													MinusTag = false;
												}
												else {
													itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, 1);
												}
											}
											else if (!it1->second.Trail_NUM && MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, 1);
												}
											}
											else {
												//!it1->second.Trail_NUM && !MinusTag
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM) {
													itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, 1);
													MinusTag = true;
												}
											}
										}
									}
									else {
										for (auto it1 = RecordWeightTrailNUM.begin(); it1 != RecordWeightTrailNUM.end(); it1++) {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 == TmpRecordWeightTrailNUM.end()) {
												itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, it1->second.Trail_NUM);
											}
											else if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM < it1->second.Trail_NUM) {
												itorYes->second.InsertOrUpdate(it1->first - nxtnxt_s.W, (it1->second.Trail_NUM - it2->second.Trail_NUM));
											}
										}
									}
								}
								else if(nxt_s.rnum <= (Rnum - EndRoundForNo)) {
									auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum));
									if (itorNo != ValueNoMapCluster.end()) {
										itorNo->second.UpdateWeightUBorLB(Bn - nxtnxt_s.W);
									}
									else {
										ValueorASPInfoForCluster new_ValueNoForCluster(Bn - nxtnxt_s.W);
										ValueNoMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxtnxt_s.rnum), new_ValueNoForCluster));
									}
								}
							}
						}
						else {
							FWRound_i(nxtnxt_s, TMPX[nxtnxt_s.rnum - 1]);
						}
					}						
					}
				}
			
		}
		else {
			MeetASP = false;
			BASN = ((Bn - (s.W + s.w + DDTorLATMinusMin[s.sbx_in[s.j]][tmp_sbxout] + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
 			auto itNo_j = ASPNoMapCluster_ASN_J.find(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1));
			if (itNo_j != ASPNoMapCluster_ASN_J.end()) {
				if (BASN <= itNo_j->second.WeightUBorLB)  goto FWSearchForActiveOutput;
			}
			auto itYes_j = ASPMapCluster_ASN_J.find(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1)); bool SearchTag = true;
			if (itYes_j != ASPMapCluster_ASN_J.end()) {
				if (BASN >= itYes_j->second.WeightUBorLB) { MeetASP = true; SearchTag = false; }
			}
			if (SearchTag) {
				FWRound_i_ASN(GenStateRI_j_ASN(s, sbx_nr_num), sbx_out);
				if (MeetASP) {
					if (itYes_j != ASPMapCluster_ASN_J.end()) {
						itYes_j->second.UpdateWeightUBorLB(BASN);
					}
					else {
						ValueorASPInfoForCluster new_info(BASN);
						ASPMapCluster_ASN_J.insert(make_pair(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1), new_info));
					}
				}
				else {
					if (itNo_j != ASPNoMapCluster_ASN_J.end()) {
						itNo_j->second.UpdateWeightUBorLB(BASN);
					}
					else {
						ValueorASPInfoForCluster new_info(BASN);
						ASPNoMapCluster_ASN_J.insert(make_pair(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1), new_info));
					}
				}
			}		
			if (MeetASP) {
				FWRound_i(UpdateStateRoundI_j(s, DDTorLATMinusMin[s.sbx_in[s.j]][tmp_sbxout], sbx_nr_w), sbx_out);
			}
		}

	}

FWSearchForActiveOutput:
	sbx_out.m128i_u8[s.sbx_a[s.j]] = 1;
	if (!s.sbx_tag[s.j]) {  
		sbx_nr_num = 1; TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = 1;
		if (s.rnum >= MaskRoundBG) {
			if (s.j == s.sbx_num || s.sbx_tag[s.j + 1]) {
				int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(sbx_out, MaskRound[s.rnum]), Mask)));
				if (asn) {
					TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R; return;					
				}
			}
			else {
				__m128i tmpsbxout = sbx_out;
				for (int sa = s.j + 1; sa <= s.sbx_num; sa++) {
					if (s.sbx_tag[sa]) break;
					else tmpsbxout.m128i_u8[s.sbx_a[sa]] = 0;
				}
				int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpgt_epi8(_mm_and_si128(tmpsbxout, MaskRound[s.rnum]), Mask)));
				if (asn) {
					TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R; return;
				}
			}
		}
	}
	MeetASP = false;
	BASN = ((Bn - (s.W + s.w + s.nr_minw)) / weight[1]) + s.sbx_num + s.nr_sbx_num + 1;
	if (s.j != s.sbx_num && !s.sbx_tag[s.j]) {
		tmp_searchOutput ^= (1 << s.j);
		auto itNo_j = ASPNoMapCluster_ASN_J.find(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1));
		if (itNo_j != ASPNoMapCluster_ASN_J.end()) {
			if (BASN <= itNo_j->second.WeightUBorLB) { TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R; return; }
		}
		auto itYes_j = ASPMapCluster_ASN_J.find(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1)); bool SearchTag = true;
		if (itYes_j != ASPMapCluster_ASN_J.end()) {
			if (BASN >= itYes_j->second.WeightUBorLB) { MeetASP = true; SearchTag = false; }
		}
		if (SearchTag) {
			STATE s_asn = GenStateRI_j_ASN(s, sbx_nr_num);
			FWRound_i_ASN(s_asn, sbx_out);
			if (MeetASP) {
				if (itYes_j != ASPMapCluster_ASN_J.end()) {
					itYes_j->second.UpdateWeightUBorLB(BASN);
				}
				else {
					ValueorASPInfoForCluster new_info(BASN);
					ASPMapCluster_ASN_J.insert(make_pair(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1), new_info));
				}
			}
			else {
				if (itNo_j != ASPNoMapCluster_ASN_J.end()) {
					itNo_j->second.UpdateWeightUBorLB(BASN);
				}
				else {
					ValueorASPInfoForCluster new_info(BASN);
					ASPNoMapCluster_ASN_J.insert(make_pair(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), make_pair(s.j, tmp_searchOutput)), Rnum - s.rnum + 1), new_info));
				}
			}
		}		
	}
	else if (s.j != s.sbx_num) {
		MeetASP = true;
	}
	else if (s.rnum + 1 != Rnum) { //s.j == s.sbx_num
		Trail[s.rnum + 1] = sbx_out;
		TMPX[s.rnum + 1] = _mm_shuffle_epi8(Trail[s.rnum + 1], State1PermutationSSE);
		Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
		BASN -= (s.sbx_num + 1);
		tmp_asp1 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum], Mask)); tmp_asp2 = _mm_movemask_epi8(_mm_cmpgt_epi8(Trail[s.rnum + 1], Mask));
		auto itNo = ASPNoMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum));
		if (itNo != ASPNoMapCluster_ASN.end()) {
			if (BASN <= itNo->second.WeightUBorLB) { TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R; return; } 
		}
		auto itYes = ASPMapCluster_ASN.find(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum)); bool SearchTag = true;
		if (itYes != ASPMapCluster_ASN.end()) {
			if (BASN >= itYes->second.WeightUBorLB) { MeetASP = true; SearchTag = false; } 
		}
		if (SearchTag) {
			STATE s_asn = GenStateRI_ASN_FW(s);
			FWRound_i_ASN(s_asn, TMPX[s.rnum]);
			if (MeetASP) {
				if (itYes != ASPMapCluster_ASN.end()) {
					itYes->second.UpdateWeightUBorLB(BASN);
				}
				else {
					ValueorASPInfoForCluster new_info(BASN);
					ASPMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum), new_info));
				}
			}
			else {
				if (itNo != ASPNoMapCluster_ASN.end()) {
					itNo->second.UpdateWeightUBorLB(BASN);
				}
				else {
					ValueorASPInfoForCluster new_info(BASN);
					ASPNoMapCluster_ASN.insert(make_pair(make_pair(make_pair(tmp_asp1, tmp_asp2), Rnum - s.rnum), new_info));
				}
			}
		}		
	}
	else MeetASP = true;

	if (!MeetASP) {
		TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R; return;
	}

	for (int i = 0; i < SBox_SIZE; i++) {
		if (s.W + s.w + FWWeightOrderWMinusMin[s.sbx_in[s.j]][i] + s.nr_minw + BestB[Rnum - s.rnum - 1] > Bn) break;
		sbx_out.m128i_u8[s.sbx_a[s.j]] = (FWWeightOrderValue[s.sbx_in[s.j]][i] ^ tmp_sbxout);
		if (!sbx_out.m128i_u8[s.sbx_a[s.j]]) continue;
		sbx_nr_w = FWWeightMinandMax[sbx_out.m128i_u8[s.sbx_a[s.j]]][0];
		if (s.sbx_tag[s.j]) sbx_nr_w -= weight[1];
		if (s.j == s.sbx_num) {
			Trail[s.rnum + 1] = sbx_out;
			TMPX[s.rnum + 1] = _mm_shuffle_epi8(Trail[s.rnum + 1], State1PermutationSSE);
			Extern2RMask[s.rnum + 1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[s.rnum + 1], Mask)); 
			STATE nxt_s = UpdateStateRoundI(s, FWWeightOrderWMinusMin[s.sbx_in[s.j]][i], sbx_nr_w);
			if (nxt_s.W + nxt_s.w + nxt_s.nr_minw + BestB[Rnum - nxt_s.rnum - 1] > Bn) continue;
			if (nxt_s.rnum == Rnum - 1) {
				FWRound_Last2Round(nxt_s, TMPX[s.rnum]);
			}
			else {
				if (s.rnum >= 4 && s.rnum <= (Rnum - 4)) {
					u64 tmp_value1 = (Trail[s.rnum].m128i_u64[1] << 4) ^ Trail[s.rnum].m128i_u64[0];
					u64 tmp_value2 = (Trail[s.rnum + 1].m128i_u64[1] << 4) ^ Trail[s.rnum + 1].m128i_u64[0];
					if (s.rnum <= (Rnum - EndRoundForNo)) {
						auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
						if (itorNo != ValueNoMapCluster.end() && (Bn - nxt_s.W <= itorNo->second.WeightUBorLB)) continue;
					}					
					auto itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
					if (itorYes == ValueMapCluster.end() || itorYes->second.SearchTag(nxt_s.W)) {
						int Tmp_RecordTrailNUM = RecordTrailNUM; map<int, WeightForCluster> TmpRecordWeightTrailNUM = RecordWeightTrailNUM;
						FWRound_i(nxt_s, TMPX[s.rnum]);

						if (Tmp_RecordTrailNUM != RecordTrailNUM) {
							if (itorYes == ValueMapCluster.end()) {
								ValueForCluster NewValueForCluster(Bn - nxt_s.W);
								ValueMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum), NewValueForCluster));
								itorYes = ValueMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
							}
							else {
								itorYes->second.UpdateWeightUB(Bn - nxt_s.W);
							}
							if (MergeTag) {
								bool MinusTag = false; itorYes->second.ClearWeightValue();
								for (auto it1 = RecordWeightTrailNUM.rbegin(); it1 != RecordWeightTrailNUM.rend(); it1++) {
									if (it1->second.Trail_NUM && !MinusTag) {
										auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
										if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
										}
									}
									else if (it1->second.Trail_NUM && MinusTag) {
										auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
										if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
											MinusTag = false;
										}
										else {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
										}
									}
									else if (!it1->second.Trail_NUM && MinusTag) {
										auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
										if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
										}
									}
									else {
										auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
										if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM) {
											itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, 1);
											MinusTag = true;
										}
									}
								}
							}
							else {
								for (auto it1 = RecordWeightTrailNUM.begin(); it1 != RecordWeightTrailNUM.end(); it1++) {
									auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
									if (it2 == TmpRecordWeightTrailNUM.end()) {
										itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, it1->second.Trail_NUM);
									}
									else if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM < it1->second.Trail_NUM) {
										itorYes->second.InsertOrUpdate(it1->first - nxt_s.W, (it1->second.Trail_NUM - it2->second.Trail_NUM));
									}
								}
							}
						}
						else if (s.rnum <= (Rnum - EndRoundForNo)) {
							auto itorNo = ValueNoMapCluster.find(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum));
							if (itorNo != ValueNoMapCluster.end()) {
								itorNo->second.UpdateWeightUBorLB(Bn - nxt_s.W);
							}
							else {
								ValueorASPInfoForCluster new_ValueNoForCluster(Bn - nxt_s.W);
								ValueNoMapCluster.insert(make_pair(make_pair(make_pair(tmp_value1, tmp_value2), nxt_s.rnum), new_ValueNoForCluster));
							}
						}
					}
				}
				else {
					FWRound_i(nxt_s, TMPX[s.rnum]);
				}
			}
		}
		else {
			FWRound_i(UpdateStateRoundI_j(s, FWWeightOrderWMinusMin[s.sbx_in[s.j]][i], sbx_nr_w), sbx_out);
		}
	}
	TMPX2R[s.rnum].m128i_u8[SBoxPermutation[s.sbx_a[s.j]]] = tmp2R;
	return;
}

void Round() {
	initial_AllTrail();
#if(TYPE)
	Trail[1] = TrailOState2; Trail[Rnum] = _mm_shuffle_epi8(TrailIState2, INVSBoxPermutationSSE);
	//Input:
	for (int i = 0; i < SBox_NUM; i++) TMPX[0].m128i_u8[INVState2Permutation[i]] = TrailOState1.m128i_u8[i];
#else
	Trail[1] = TrailIState1; Trail[Rnum] = TrailOState1;
	//Input:
	TMPX[0] = _mm_shuffle_epi8(TrailIState2, SBoxPermutationSSE);
#endif
	TMPX[1] = _mm_shuffle_epi8(Trail[1], State1PermutationSSE);
	Extern2RMask[1] = _mm_movemask_epi8(_mm_cmpeq_epi8(TMPX[1], Mask));
	STATE s_r1 = UpdateStateRound1();
	//Output:
	memset(MaskRound, 0, sizeof(MaskRound)); 
	LastASN1 = 0; LastASN2 = 0;
#if(TYPE)
	__m128i tmpTrailOut = TrailIState1;
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail[Rnum].m128i_u8[i]) {
			LastASN1++;
		}
		else {
			MaskRound[Rnum - 1].m128i_u8[i] = 0xf;
		}
		if (tmpTrailOut.m128i_u8[i]) {
			LastASN2++;
			MaskRound[0].m128i_u8[i] = 0xf;
		}
		else {
			MaskRound[Rnum].m128i_u8[i] = 0xf;
		}
	}
#else
	for (int i = 0; i < SBox_NUM; i++) {
		if (Trail[Rnum].m128i_u8[i]) {
			LastASN1++;
		}
		else {
			MaskRound[Rnum - 1].m128i_u8[i] = 0xf;
		}
		if (TrailOState2.m128i_u8[i]) {
			LastASN2++;
			MaskRound[0].m128i_u8[i] = 0xf;
		}
		else {
			MaskRound[Rnum].m128i_u8[i] = 0xf;
		}
	}
#endif
	MaskRoundBG = Rnum - 1;
	__m128i tmpRecord;
	while (1) {
		tmpRecord = _mm_shuffle_epi8(MaskRound[MaskRoundBG], SBoxPermutationSSE);
		tmpRecord = _mm_and_si128(tmpRecord, MaskRound[MaskRoundBG + 1]);
		int count = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(tmpRecord, Mask)));
		if (count == SBox_NUM) break;
		MaskRoundBG--;
		MaskRound[MaskRoundBG] = _mm_shuffle_epi8(tmpRecord, INVState1PermutationSSE);
		if (MaskRoundBG == 1) break;
	}
	RecordTrailNUM = 0; 
	FWRound_i(s_r1, TMPX[0]);
	return;
}

void TransInputTrail(__m128i input1, __m128i input2, bool tag) {
	__m128i tmpOutput1 = _mm_setzero_si128(); __m128i tmpOutput2 = _mm_setzero_si128();
#if(TYPE)
	if (tag) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (input1.m128i_u8[i]) {
				tmpOutput1.m128i_u8[i] = FWWeightOrderValue[input1.m128i_u8[i]][0];
			}
		}
		for (int i = 0; i < SBox_NUM; i++) {
			if (input2.m128i_u8[i]) {
				tmpOutput2.m128i_u8[INVState2Permutation[i]] = input2.m128i_u8[i];
			}
		}
		tmpOutput1 = _mm_xor_si128(tmpOutput1, tmpOutput2);
		TrailIState1 = tmpOutput1; TrailIState2 = input1;
	}
	else {
		for (int i = 0; i < SBox_NUM; i++) {
			tmpOutput1.m128i_u8[i] = input1.m128i_u8[i] & 0xf;
			tmpOutput2.m128i_u8[i] = input1.m128i_u8[i] >> 4;
		}
		TrailIState1 = tmpOutput1; TrailIState2 = tmpOutput2;
	}
#else
	if (tag) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (input1.m128i_u8[i]) {
				tmpOutput1.m128i_u8[i] = FWWeightOrderValue[input1.m128i_u8[i]][0];
			}
		}		
		for (int i = 0; i < SBox_NUM; i++) {
			if (input2.m128i_u8[i]) {
				tmpOutput2.m128i_u8[INVSBoxPermutation[i]] = input2.m128i_u8[i];
			}
		}
		tmpOutput1 = _mm_xor_si128(tmpOutput1, tmpOutput2);
		TrailIState1 = input1; TrailIState2 = tmpOutput1;
	}
	else {
		for (int i = 0; i < SBox_NUM; i++) {
			tmpOutput1.m128i_u8[i] = input1.m128i_u8[i] & 0xf;
			tmpOutput2.m128i_u8[i] = input1.m128i_u8[i] >> 4;
		}
		TrailIState1 = tmpOutput1; TrailIState2 = tmpOutput2;
	}
#endif
}

void TransOutputTrail(__m128i input1, __m128i input2, bool tag) {
	__m128i tmpOutput1 = _mm_setzero_si128(); __m128i tmpOutput2 = _mm_setzero_si128();
#if(TYPE)
	if (tag) {	
		__m128i tmpOutput3 = _mm_setzero_si128();
		for (int i = 0; i < SBox_NUM; i++) {
			if (input2.m128i_u8[i]) {
				tmpOutput3.m128i_u8[INVSBoxPermutation[i]] = input2.m128i_u8[i];
				tmpOutput1.m128i_u8[State2Permutation[i]] = FWWeightOrderValue[input2.m128i_u8[i]][0];
			}
			if (input1.m128i_u8[i]) {
				tmpOutput2.m128i_u8[State2Permutation[INVSBoxPermutation[i]]] = input1.m128i_u8[i];
			}
		}
		tmpOutput1 = _mm_xor_si128(tmpOutput1, tmpOutput2);
	}
	else {
		for (int i = 0; i < SBox_NUM; i++) {
			tmpOutput1.m128i_u8[i] = input1.m128i_u8[i] >> 4;
			tmpOutput2.m128i_u8[i] = input1.m128i_u8[i] & 0xf;
		}
		TrailOState1 = tmpOutput1; TrailOState2 = tmpOutput2;
	}
#else
	if (tag) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (input1.m128i_u8[i]) {
				tmpOutput1.m128i_u8[State1Permutation[i]] = input1.m128i_u8[i];
			}
		}
		for (int i = 0; i < SBox_NUM; i++) {
			if (input2.m128i_u8[i]) {
				tmpOutput2.m128i_u8[SBoxPermutation[i]] = FWWeightOrderValue[input2.m128i_u8[i]][0];
			}
		}
		tmpOutput1 = _mm_xor_si128(tmpOutput1, tmpOutput2);
		TrailOState1 = input2; TrailOState2 = tmpOutput1;
	}
	else {
		for (int i = 0; i < SBox_NUM; i++) {
			tmpOutput1.m128i_u8[i] = input1.m128i_u8[i] >> 4;
			tmpOutput2.m128i_u8[i] = input1.m128i_u8[i] & 0xf;
		}
		__m128i tmpOutput3 = _mm_setzero_si128();
		for (int i = 0; i < SBox_NUM; i++) {
			tmpOutput3.m128i_u8[INVState2Permutation[i]] = tmpOutput1.m128i_u8[i];
		}
		TrailOState1 = tmpOutput3; TrailOState2 = tmpOutput2;
	}
#endif
}

void matsui() {
	WeightOrderTables();
	__m128i tmp1 = _mm_setzero_si128();
	__m128i tmp2 = _mm_setzero_si128();
	__m128i tmp3 = _mm_setzero_si128();
	__m128i TrailInput = _mm_setzero_si128(); __m128i TrailOutput = _mm_setzero_si128();
#if(TYPE)
	string fileName = "WARP_Linear_ClusterSearch.txt";
	for (int i = 0; i <= RNUM; i++) BestB[i] *= 2;
#else	
	string fileName = "WARP_Diff_ClusterSearch.txt";
#endif

	// input info:
	Rnum = 18;                  // the number of rounds for distinguisher
	int Margin = 0;
	Bn = BestB[Rnum] + Margin;    // the upper bound on the weight of trails
	MergeTag = false;             // whether to use the merging weights strategy
	OutputTrailTag = false;		  // Whether to output trails that satisfy the conditions for search, only all trails can be output if no strategy is used, otherwise only some trails can be output
	EndRoundForNo = 10;
	// Other input information that needs to be changed, including differential or linear cluster searches, and the state size of KNOT, is changed in GlobleVariables.h.	

	//the input and output for the distinguisher
#if(TYPE)
	
#else
	if (Rnum == 18) {
		//18
		tmp3 = _mm_setzero_si128();
		tmp3.m128i_u16[2] = 0xaa75;
		tmp3.m128i_u16[3] = 0xa000;
		tmp3.m128i_u16[5] = 0x005a;
		tmp3.m128i_u16[6] = 0xa0af;
		TransInputTrail(tmp3, tmp2, false);
		TrailInput = _mm_xor_si128(TrailIState1, _mm_slli_epi16(TrailIState2, 4));
		tmp3 = _mm_setzero_si128();
		tmp3.m128i_u16[0] = 0x0a00;
		tmp3.m128i_u16[2] = 0x000a;
		tmp3.m128i_u16[3] = 0x5a00;
		tmp3.m128i_u16[5] = 0x0aa0;
		tmp3.m128i_u16[6] = 0x5000;
		tmp3.m128i_u16[7] = 0x000a;
		TransOutputTrail(tmp3, tmp2, false);
		TrailOutput = _mm_setzero_si128();
		for (int i = 0; i < SBox_NUM; i++) {
			if (TrailOState1.m128i_u8[i]) {
				TrailOutput.m128i_u8[State2Permutation[i]] = TrailOState1.m128i_u8[i];
			}
		}
		TrailOutput = _mm_xor_si128(_mm_slli_epi16(TrailOutput, 4), TrailOState2);
	}
#endif

	stringstream message;
	message << "Rnum:" << Rnum << ",  Bn:" << BestB[Rnum] << ", UBofWeight:" << Bn << ", Margin : " << Margin << ", use the merging weights strategy: " << MergeTag << "\n";
	message << "Input:\n0x";
	for (int k = 0xf; k >= 0; k--) {
		message << hex << setw(2) << setfill('0') << static_cast<int>(TrailInput.m128i_u8[k]);
	}
	message << "\nOutput:\n0x";
	for (int k = 0xf; k >= 0; k--) {
		message << hex << setw(2) << setfill('0') << static_cast<int>(TrailOutput.m128i_u8[k]);
	}
	message << "\n\n";
	logToFile(fileName, message.str());


	clock_t Start = clock();
	Round();
	clock_t End = clock();


	double ComputeWeightForCluster = 0; int ComputeTrailNUM = 0;
	for (auto itor = RecordWeightTrailNUM.begin(); itor != RecordWeightTrailNUM.end(); itor++) {
		ComputeWeightForCluster += pow(2, (double)(-1 * itor->first)) * itor->second.Trail_NUM;
		ComputeTrailNUM += itor->second.Trail_NUM;
		printf("Weight: %d, NUM: %d\n", itor->first, itor->second.Trail_NUM);
		message << dec << "Weight: " << itor->first << ", NUM: " << itor->second.Trail_NUM << "\n";
	}
	printf("Trail_NUM: %d, TrailClusterWeight: %f", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))));
	printf("\nTotal Time: %fs, %fmin\n", ((double)(End - Start)) / CLOCKS_PER_SEC, (((double)(End - Start)) / CLOCKS_PER_SEC) / 60);

	message << dec << "Trail_NUM: " << ComputeTrailNUM << ", ClusterWeight: " << (-1 * (log(ComputeWeightForCluster) / log(2))) << endl;
	message << dec << "Total Time: " << ((double)(End - Start)) / CLOCKS_PER_SEC << "s, " << (((double)(End - Start)) / CLOCKS_PER_SEC) / 60 << "min\n\n";
	logToFile(fileName, message.str());
	return;
}

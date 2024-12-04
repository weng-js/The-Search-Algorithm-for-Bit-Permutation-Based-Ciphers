#include<iostream>
#include<string>
#include<sstream>
#include<fstream>
#include<iomanip>
#include<nmmintrin.h>
#include<vector>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
using namespace std;

ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
int t_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1][State_NUM];    //256bit 2*__m128i  384 3*__m128i 512 4*__m128i 
int Best_w[RNUM];

#if(Block_SIZE==256)
#if(TYPE)
int BestB[RNUM + 1] = { 0, 1, 2, 4, 6, 8, 10, 13, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53,
		56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119, 122,
		125, 128, 131, 134, 137, 140 };
#else
int BestB[RNUM + 1] = { 0, 2, 4, 7, 10, 14, 18, 25, 32, 40, 49, 55, 60, 66, 71, 76, 82, 87, 92, 98,
		103, 108, 114, 119, 124, 130, 135, 140, 146, 151, 156, 162, 167, 172, 178, 183, 188, 194, 199,
		204, 210, 215, 220, 226, 231, 236, 242, 247, 252, 258, 263, 268, 274 };
#endif
#elif(Block_SIZE==384)
#if(TYPE)
int BestB[RNUM + 1] = { 0, 1, 2, 4, 6, 8, 10, 13, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53,
		56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119,
		122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176,
		179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, };
#else
int BestB[RNUM + 1] = { 0, 2, 4, 7, 10, 14, 18, 25, 32, 40, 49, 55, 60, 66, 71, 76, 82, 87, 92, 98,
		103, 108, 114, 119, 124, 130, 135, 140, 146, 151, 156, 162, 167, 172, 178, 183, 188, 194, 199,
		204, 210, 215, 220, 226, 231, 236, 242, 247, 252, 258, 263, 268, 274, 279, 284, 290, 295, 300,
		306, 311, 316, 322, 327, 332, 338, 343, 348, 354, 359, 364, 370, 375, 380, 386, 391, 396, 402, };
#endif
#else
#if(TYPE)
int BestB[RNUM + 1] = { 0, 1, 2, 4, 6, 8, 10, 13, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53,
		56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110, 113, 116, 119,
		122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 176,
		179, 182, 185, 188, 191, 194, 197, 200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233,
		236, 239, 242, 245, 248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278, 281, 284 };
#else
int BestB[RNUM + 1] = { 0, 2, 4, 7, 10, 14, 18, 25, 32, 40, 49, 55, 60, 66, 71, 76, 82, 87, 92, 98,
		103, 108, 114, 119, 124, 130, 135, 140, 146, 151, 156, 162, 167, 172, 178, 183, 188, 194, 199,
		204, 210, 215, 220, 226, 231, 236, 242, 247, 252, 258, 263, 268, 274, 279, 284, 290, 295, 300,
		306, 311, 316, 322, 327, 332, 338, 343, 348, 354, 359, 364, 370, 375, 380, 386, 391, 396, 402,
		407, 412, 418, 423, 428, 434, 439, 444, 450, 455, 460, 466, 471, 476, 482, 487, 492, 498, 503, 508, 514, 519, 524, 530 };
#endif
#endif

int Rnum;
int Bn;
int Trail_Weight;
int ASNUB_ER;
bool MergeTag;
bool OutputTrailTag;
long long RecordTrailNUM;

__m128i count_asn = _mm_setzero_si128();
__m128i MASK1 = _mm_set1_epi8(0xf);
__m128i MASK2 = _mm_set1_epi8(0xf0);
ALIGNED_TYPE_(__m128i, 16) MaskRound_1[RNUM + 1][State_NUM];
ALIGNED_TYPE_(__m128i, 16) MaskRound_2[RNUM + 1][State_NUM];
int MaskRoundBG;
int TrailOutputASN;
u8  TrailOutputASBO[SBox_NUM];


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
	long long Trail_NUM;
	WeightForCluster() : Trail_NUM(1) {};
	~WeightForCluster() {};

	void UpdateNUM() { Trail_NUM++; };
	void UpdateNUMPlus(long long data) { Trail_NUM += data; }
	void InitNUM(long long data) { Trail_NUM = data; }
};

map<int, WeightForCluster> RecordWeightTrailNUM;

class ValueForCluster {
	// Using this class only outputs some trails
public:
	int WeightUB;
	map<int, long long> WeightValueAndNUM;
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
					long long tmp_Record = RecordTrailNUM;
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

	void InsertOrUpdate(int FindWeight, long long TrailNUM) {
		WeightValueAndNUM[FindWeight] = TrailNUM;
		return;
	}

	void UpdateWeightUB(int data) { WeightUB = data; }
};

class ValueNoForCluster {
	//An upper bound on the weight at which the condition is not satisfied. 
	//If the weight exceeds this upper bound, a satisfying conditions trail may exist
public:
	int WeightUB;
	ValueNoForCluster(int data) :WeightUB(data) {};
	~ValueNoForCluster() {};
	void UpdateWeightUB(int data) {
		WeightUB = data;
	}
};


map<pair<pair<pair<u8, u8>, pair<u8, u8>>, int>, ValueForCluster> ValueMapCluster;			// the info of the first active sbox and the second active sbox, and the round
map<pair<pair<u8, u8>, int>, ValueNoForCluster> ValueNoMapCluster_asbx1;					// the info of the first active sbox, and the round
map<pair<pair<pair<u8, u8>, pair<u8, u8>>, int>, ValueNoForCluster> ValueNoMapCluster_sbx2; // the info of the first active sbox and the second active sbox, and the round
map<pair<pair<pair<pair<u8, u8>, pair<u8, u8>>, pair<u8, u8>>, int>, ValueNoForCluster> ValueNoMapCluster_asbx3; // the info of the first, the second and the third active sbox, and the round
map<pair<pair<pair<pair<u8, u8>, pair<u8, u8>>, pair<pair<u8, u8>, pair<u8, u8>>>, int>, ValueNoForCluster> ValueNoMapCluster_asbx4; // the info of the first, the second, the third and the fourth active sbox, and the round


void FileOutputTrail() {
#if(TYPE==0)
	string fileName = "KNOT" + to_string(Block_SIZE) + "_Diff_ClusterSearch_Trail.txt";
#elif(TYPE==1)
	string fileName = "KNOT" + to_string(Block_SIZE) + "_Linear_ClusterSearch_Trail.txt";
#endif

	ALIGNED_TYPE_(__m128i, 16) SO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	ALIGNED_TYPE_(__m128i, 16) PO[RNUM][State_NUM];        //256bit 2*__m128i  384 3*__m128i 512 4*__m128i
	memset(SO, 0, RNUM * STATE_LEN);
	memset(PO, 0, RNUM * STATE_LEN);
	memcpy(PO, BestTrail, Rnum * STATE_LEN);
	memcpy(SO[Rnum - 1], BestTrail[Rnum], STATE_LEN);

	for (int r = 1; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if ((PO[r][Sbox_loc[i][0]].m128i_u8[Sbox_loc[i][1]] >> Sbox_loc[i][2]) & 0xf) {
				for (int k = 0; k < State_NUM; k++) {
					SO[r - 1][k] = _mm_xor_si128(SO[r - 1][k], INVPTable[i][(PO[r][Sbox_loc[i][0]].m128i_u8[Sbox_loc[i][1]] >> Sbox_loc[i][2]) & 0xf][k]);
				}
			}
		}
	}

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Trail_Weight << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				message << hex << setw(2) << setfill('0') << static_cast<int>(PO[r][s].m128i_u8[k]);
			}
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int s = State_NUM - 1; s >= 0; s--) {
			for (int k = 0xf; k >= 0; k--) {
				message << hex << setw(2) << setfill('0') << static_cast<int>(SO[r][s].m128i_u8[k]);
			}
		}
		message << "  w: " << dec << Best_w[r] << "\n\n";
	}
	message << "\n\n";
	logToFile(fileName, message.str());
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

void FWRound_n(STATE s) {
	s.W += s.w;
	Trail_Weight = s.W;
	RecordTrailNUM++;
	if (OutputTrailTag) {
		t_w[s.rnum - 1] = s.w;
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
	if (MergeTag) WeightTrailInfoReorganize();
	return;
}


void FWRound_i(STATE s, __m128i sbx_out[]) {
	int asn;
	for (int i = 0; i < SBox_SIZE; i++) {
		if (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + BestB[Rnum - s.rnum] > Bn) {
			if (i > 0) for (int k = 0; k < State_NUM; k++) sbx_out[k] = _mm_xor_si128(sbx_out[k], FWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i - 1][k]);
			break;
		}
		asn = 0; // the number of active sbox for next round
		for (int k = 0; k < State_NUM; k++) {
			sbx_out[k] = _mm_xor_si128(sbx_out[k], FWSPTableXor[s.sbx_a[s.j]][s.sbx_in[s.j]][i][k]);
			asn += _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(sbx_out[k], MASK1), count_asn)))
				+ _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(sbx_out[k], MASK2), count_asn)));
		}
		asn = SBox_NUM - asn;

		if (s.rnum >= MaskRoundBG) {
			int tmp_asn = 0;
			for (int k = 0; k < State_NUM; k++) {
				tmp_asn += _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(sbx_out[k], MaskRound_1[s.rnum][k]), count_asn)))
					+ _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(sbx_out[k], MaskRound_2[s.rnum][k]), count_asn)));
			}
			if (tmp_asn != SBox_NUM) continue;
		}

		if ((s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + BestB[Rnum - s.rnum - 1]) <= Bn && asn <= ASNUB_ER) {
			if (s.j == s.sbx_num) {
				STATE s_nr = FWupdate_state_row(s, FWWeightOrderW[s.sbx_in[s.j]][i], sbx_out);
				if (s_nr.W + s_nr.w + BestB[Rnum - s.rnum - 1] <= Bn) {
					if (s.rnum + 1 == Rnum) {
						if (s_nr.w != -1) FWRound_n(s_nr);
					}
					else if (s.rnum >= 4 && s.rnum <= (Rnum - 4) && asn <= 4) {
						if (asn == 1) {
							auto itorNo = ValueNoMapCluster_asbx1.find(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), s_nr.rnum));
							if (itorNo != ValueNoMapCluster_asbx1.end() && (Bn - s_nr.W <= itorNo->second.WeightUB)) continue;
							long long Tmp_RecordTrailNUM = RecordTrailNUM;

							ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
							memset(tmp_out, 0, STATE_LEN);
							FWRound_i(s_nr, tmp_out);
							if (Tmp_RecordTrailNUM == RecordTrailNUM) {
								if (itorNo != ValueNoMapCluster_asbx1.end()) {
									itorNo->second.UpdateWeightUB(Bn - s_nr.W);
								}
								else {
									ValueNoForCluster new_ValueNoForCluster(Bn - s_nr.W);
									ValueNoMapCluster_asbx1.insert(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), s_nr.rnum), new_ValueNoForCluster));
								}
							}
						}
						else if (asn == 2) {
							auto itorNo = ValueNoMapCluster_sbx2.find(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), s_nr.rnum));
							if (itorNo != ValueNoMapCluster_sbx2.end() && (Bn - s_nr.W <= itorNo->second.WeightUB)) continue;
							auto itorYes = ValueMapCluster.find(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), s_nr.rnum));

							if (itorYes == ValueMapCluster.end() || itorYes->second.SearchTag(s_nr.W)) {
								long long Tmp_RecordTrailNUM = RecordTrailNUM;
								map<int, WeightForCluster> TmpRecordWeightTrailNUM = RecordWeightTrailNUM;

								ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
								memset(tmp_out, 0, STATE_LEN);
								FWRound_i(s_nr, tmp_out);

								if (Tmp_RecordTrailNUM == RecordTrailNUM) {
									if (itorNo != ValueNoMapCluster_sbx2.end()) {
										itorNo->second.UpdateWeightUB(Bn - s_nr.W);
									}
									else {
										ValueNoForCluster new_ValueNoForCluster(Bn - s_nr.W);
										ValueNoMapCluster_sbx2.insert(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), s_nr.rnum), new_ValueNoForCluster));
									}
								}
								else {
									if (itorYes == ValueMapCluster.end()) {
										ValueForCluster NewValueForCluster(Bn - s_nr.W);
										ValueMapCluster.insert(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), s_nr.rnum), NewValueForCluster));
										itorYes = ValueMapCluster.find(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), s_nr.rnum));
									}
									else {
										itorYes->second.UpdateWeightUB(Bn - s_nr.W);
									}

									itorYes->second.ClearWeightValue();
									if (MergeTag) {
										bool MinusTag = false; 
										for (auto it1 = RecordWeightTrailNUM.rbegin(); it1 != RecordWeightTrailNUM.rend(); it1++) {
											if (it1->second.Trail_NUM && !MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													itorYes->second.InsertOrUpdate(it1->first - s_nr.W, 1);
												}
											}
											else if (it1->second.Trail_NUM && MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													MinusTag = false;
												}
												else {
													itorYes->second.InsertOrUpdate(it1->first - s_nr.W, 1);
												}
											}
											else if (!it1->second.Trail_NUM && MinusTag) {
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 == TmpRecordWeightTrailNUM.end() || !it2->second.Trail_NUM) {
													itorYes->second.InsertOrUpdate(it1->first - s_nr.W, 1);
												}
											}
											else {
												//!it1->second.Trail_NUM && !MinusTag
												auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
												if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM) {
													//¿ÉÒÔ²åÈë
													itorYes->second.InsertOrUpdate(it1->first - s_nr.W, 1);
													MinusTag = true;
												}
											}
										}
									}
									else {
										for (auto it1 = RecordWeightTrailNUM.begin(); it1 != RecordWeightTrailNUM.end(); it1++) {
											auto it2 = TmpRecordWeightTrailNUM.find(it1->first);
											if (it2 == TmpRecordWeightTrailNUM.end()) {
												itorYes->second.InsertOrUpdate(it1->first - s_nr.W, it1->second.Trail_NUM);
											}
											else if (it2 != TmpRecordWeightTrailNUM.end() && it2->second.Trail_NUM < it1->second.Trail_NUM) {
												itorYes->second.InsertOrUpdate(it1->first - s_nr.W, (it1->second.Trail_NUM - it2->second.Trail_NUM));
											}
										}
									}
									

								}

							}
						}
						else if (asn == 3) {
							auto itorNo = ValueNoMapCluster_asbx3.find(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), make_pair(s_nr.sbx_a[2], s_nr.sbx_in[2])), s_nr.rnum));
							if (itorNo != ValueNoMapCluster_asbx3.end() && (Bn - s_nr.W <= itorNo->second.WeightUB)) continue;
							long long Tmp_RecordTrailNUM = RecordTrailNUM;

							ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
							memset(tmp_out, 0, STATE_LEN);
							FWRound_i(s_nr, tmp_out);

							if (Tmp_RecordTrailNUM == RecordTrailNUM) {
								if (itorNo != ValueNoMapCluster_asbx3.end()) {
									itorNo->second.UpdateWeightUB(Bn - s_nr.W);
								}
								else {
									ValueNoForCluster new_ValueNoForCluster(Bn - s_nr.W);
									ValueNoMapCluster_asbx3.insert(make_pair(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1])), make_pair(s_nr.sbx_a[2], s_nr.sbx_in[2])), s_nr.rnum), new_ValueNoForCluster));
								}
							}

						}
						else if (asn == 4) {
							auto itorNo = ValueNoMapCluster_asbx4.find(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1]))
								, make_pair(make_pair(s_nr.sbx_a[2], s_nr.sbx_in[2]), make_pair(s_nr.sbx_a[3], s_nr.sbx_in[3]))), s_nr.rnum));
							if (itorNo != ValueNoMapCluster_asbx4.end() && (Bn - s_nr.W <= itorNo->second.WeightUB)) continue;
							long long Tmp_RecordTrailNUM = RecordTrailNUM;

							ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
							memset(tmp_out, 0, STATE_LEN);
							FWRound_i(s_nr, tmp_out);

							if (Tmp_RecordTrailNUM == RecordTrailNUM) {
								if (itorNo != ValueNoMapCluster_asbx4.end()) {
									itorNo->second.UpdateWeightUB(Bn - s_nr.W);
								}
								else {
									ValueNoForCluster new_ValueNoForCluster(Bn - s_nr.W);
									ValueNoMapCluster_asbx4.insert(make_pair(make_pair(make_pair(make_pair(make_pair(s_nr.sbx_a[0], s_nr.sbx_in[0]), make_pair(s_nr.sbx_a[1], s_nr.sbx_in[1]))
										, make_pair(make_pair(s_nr.sbx_a[2], s_nr.sbx_in[2]), make_pair(s_nr.sbx_a[3], s_nr.sbx_in[3]))), s_nr.rnum), new_ValueNoForCluster));
								}
							}

						}

					}
					else {
						ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM];
						memset(tmp_out, 0, STATE_LEN);
						FWRound_i(s_nr, tmp_out);
					}
				}
			}
			else FWRound_i(update_state_sbx(s, FWWeightOrderW[s.sbx_in[s.j]][i]), sbx_out);
		}

	}
	return;
}

void Round(__m128i TrailI[], __m128i TrailO[]) {
	initial_AllTrail();
	memcpy(Trail[0], TrailI, STATE_LEN);
	memcpy(Trail[Rnum], TrailO, STATE_LEN);
	memset(MaskRound_1, 0, sizeof(MaskRound_1));
	memset(MaskRound_2, 0, sizeof(MaskRound_2));
	ALIGNED_TYPE_(__m128i, 16) TmpOutput[State_NUM];
	memset(TmpOutput, 0, sizeof(TmpOutput));


	STATE s = GenStateForRound1();
	memset(TrailOutputASBO, 0, sizeof(TrailOutputASBO)); TrailOutputASN = 0;
	for (int i = 0; i < State_NUM; i++) {
		for (int j = 0; j < 16; j++) {
			if (TrailO[i].m128i_u8[j]) {
				if (TrailO[i].m128i_u8[j] & 0xf) {
					TrailOutputASBO[(i << 5) | (j << 1)] = TrailO[i].m128i_u8[j] & 0xf;
					TrailOutputASN++;
					for (int k = 0; k < State_NUM; k++) {
						TmpOutput[k] = _mm_xor_si128(TmpOutput[k], INVPTable[(i << 5) | (j << 1)][0xf][k]);
					}
				}
				else {
					MaskRound_1[Rnum - 1][i].m128i_u8[j] = 0xf;
				}
				if (TrailO[i].m128i_u8[j] & 0xf0) {
					TrailOutputASBO[(i << 5) | (j << 1) | 1] = TrailO[i].m128i_u8[j] >> 4;
					TrailOutputASN++;
					for (int k = 0; k < State_NUM; k++) {
						TmpOutput[k] = _mm_xor_si128(TmpOutput[k], INVPTable[(i << 5) | (j << 1) | 1][0xf][k]);
					}
				}
				else {
					MaskRound_2[Rnum - 1][i].m128i_u8[j] = 0xf0;
				}
			}
			else {
				MaskRound_1[Rnum - 1][i].m128i_u8[j] = 0xf;
				MaskRound_2[Rnum - 1][i].m128i_u8[j] = 0xf0;
			}
		}
	}

	MaskRoundBG = Rnum - 1; int record_asn = 0;
	ALIGNED_TYPE_(__m128i, 16) TmpInput[State_NUM];
	while (record_asn < SBox_NUM) {
		record_asn = 0; MaskRoundBG--;
		memcpy(TmpInput, TmpOutput, sizeof(TmpInput));
		memset(TmpOutput, 0, sizeof(TmpOutput));
		for (int i = 0; i < State_NUM; i++) {
			for (int j = 0; j < 16; j++) {
				if (TmpInput[i].m128i_u8[j]) {
					if (TmpInput[i].m128i_u8[j] & 0xf) {
						record_asn++;
						for (int k = 0; k < State_NUM; k++) {
							TmpOutput[k] = _mm_xor_si128(TmpOutput[k], INVPTable[(i << 5) | (j << 1)][0xf][k]);
						}
					}
					else {
						MaskRound_1[MaskRoundBG][i].m128i_u8[j] = 0xf;
					}
					if (TmpInput[i].m128i_u8[j] & 0xf0) {
						record_asn++;
						for (int k = 0; k < State_NUM; k++) {
							TmpOutput[k] = _mm_xor_si128(TmpOutput[k], INVPTable[(i << 5) | (j << 1) | 1][0xf][k]);
						}
					}
					else {
						MaskRound_2[MaskRoundBG][i].m128i_u8[j] = 0xf0;
					}
				}
				else {
					MaskRound_1[MaskRoundBG][i].m128i_u8[j] = 0xf;
					MaskRound_2[MaskRoundBG][i].m128i_u8[j] = 0xf0;
				}
			}
		}
		if (record_asn == SBox_NUM) {
			MaskRoundBG++; break;
		}
	}

	RecordTrailNUM = 0;
	ALIGNED_TYPE_(__m128i, 16) tmp_out[State_NUM]; memset(tmp_out, 0, sizeof(tmp_out));
	FWRound_i(s, tmp_out);
	return;
}

void matsui() {
	GenTables();
#if(TYPE)
	string fileName = "KNOT" + to_string(Block_SIZE) + "_Linear_ClusterSearch.txt";
	for (int i = 0; i <= RNUM; i++) BestB[i] *= 2;
#else	
	string fileName = "KNOT" + to_string(Block_SIZE) + "_Diff_ClusterSearch.txt";
#endif
	RecordWeightTrailNUM.clear();
	ALIGNED_TYPE_(__m128i, 16) TrailI[State_NUM];
	ALIGNED_TYPE_(__m128i, 16) TrailO[State_NUM];
	memset(TrailI, 0, sizeof(TrailI));
	memset(TrailO, 0, sizeof(TrailO));

	// input info:
	Rnum = RNUM;                  // the number of rounds for distinguisher
	ASNUB_ER = 3;                 // the maximum number of active sbox per round
	int Margin = 25;
	Bn = BestB[Rnum] + Margin;    // the upper bound on the weight of trails
	MergeTag = true;             // whether to use the merging weights strategy
	OutputTrailTag = false;		  // Whether to output trails that satisfy the conditions for search, only all trails can be output if no strategy is used, otherwise only some trails can be output
	// Other input information that needs to be changed, including differential or linear cluster searches, and the state size of KNOT, is changed in GlobleVariables.h.	

	//the input and output for the distinguisher
#if(Block_SIZE==256)
#if(TYPE)
	if (Rnum == 49) {
		//Rnum:49
		//00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03
		//00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 93 00 00 00 05 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x03;
		TrailI[1].m128i_u8[3] = 0xc0;
		//TrailO[0].m128i_u8[4] = 0x05;
		//TrailO[0].m128i_u8[8] = 0x93;
		//0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 03 00 00 00 05 00 00 00 00
		TrailO[0].m128i_u8[4] = 0x05;
		TrailO[0].m128i_u8[8] = 0x03;
	}
	else if (Rnum == 12) {
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 93 00 00 00 05 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x03;
		TrailI[1].m128i_u8[3] = 0xc0;
		TrailO[0].m128i_u8[4] = 0x05;
		TrailO[0].m128i_u8[8] = 0x93;
	}
#else
	if (Rnum == 49) {
		//Rnum:49
		//90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 90 00 00 00 00 00 00 00 00
		//00 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 a0 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[8] = 0x90;
		TrailI[1].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[7] = 0xa0;
		TrailO[1].m128i_u8[12] = 0xc0;
	}
	else if (Rnum == 5) {
		//Rnum:5
		// 0x90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 00 00 00 00 00 00 00 00 70 00 00 00 00
		TrailI[1].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[4] = 0x70;
		TrailO[1].m128i_u8[0] = 0xc0;
	}
	else if (Rnum == 12) {
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00 00 00 09
		// 0x00 0c 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x09;
		TrailI[0].m128i_u8[11] = 0x0a;
		TrailO[0].m128i_u8[9] = 0x0a;
		TrailO[1].m128i_u8[14] = 0x0c;
	}
	else if (Rnum == 52) {
		// 0x90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 90 00 00 00 00 00 00 00 00
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 0a 00 00 00 00 00 00 00 00 00 00 0c 00
		TrailI[0].m128i_u8[8] = 0x90;
		TrailI[1].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[1] = 0x0c;
		TrailO[0].m128i_u8[12] = 0x0a;
	}
	else if (Rnum == 6) {
		TrailI[0].m128i_u8[3] = 0x20;
		TrailI[0].m128i_u8[15] = 0xc0;
		TrailO[0].m128i_u8[1] = 0x0c;
		TrailO[0].m128i_u8[12] = 0x0a;
	}
#endif
#elif(Block_SIZE==384)
#if(TYPE)
	if (Rnum == 73) {
		//Rnum:49
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 93 00 00 00 05 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x03;
		TrailI[0].m128i_u8[4] = 0xc0;
		TrailO[0].m128i_u8[4] = 0x05;
		TrailO[0].m128i_u8[8] = 0x93;
	}
	else if (Rnum == 12) {
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 93 00 00 00 05 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x03;
		TrailI[0].m128i_u8[4] = 0xc0;
		TrailO[0].m128i_u8[4] = 0x05;
		TrailO[0].m128i_u8[8] = 0x93;
	}
#else
	if (Rnum == 73) {
		//Rnum:49
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 09 00 00 0a
		// 0x00 00 00 00 00 00 00 00 70 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x0a;
		TrailI[0].m128i_u8[3] = 0x09;
		TrailO[1].m128i_u8[2] = 0x10;
		TrailO[2].m128i_u8[7] = 0x70;
	}
	else if (Rnum == 5) {
		//Rnum:5
		// 0x90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 70 00 00 00 00

		TrailI[2].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[4] = 0x70;
		TrailO[1].m128i_u8[15] = 0xc0;
	}
	else if (Rnum == 12) {
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 09 00 00 0a
		// 0x00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0c
		TrailI[0].m128i_u8[0] = 0x0a;
		TrailI[0].m128i_u8[3] = 0x09;
		TrailO[0].m128i_u8[0] = 0x0c;
		TrailO[2].m128i_u8[13] = 0x0a;
	}
	else if (Rnum == 76) {
		//0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  
		//  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		//  00 00 00 00 00 00 00 00 00 00 00 00 09 00 00 0a
		//0x00 00 00 07 00 00 00 00 00 00 00 00 00 00 00 00
		//  00 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
		//  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x0a;
		TrailI[0].m128i_u8[3] = 0x09;
		TrailO[1].m128i_u8[7] = 0x01;
		TrailO[2].m128i_u8[12] = 0x07;
	}
#endif
#elif(Block_SIZE==512) //512
#if(TYPE)
	if (Rnum == RNUM) {
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 c0 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		//   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
		//   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 93  00 00 00 00 00 00 00 05 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[0] = 0x03;
		TrailI[3].m128i_u8[3] = 0xc0;
		TrailO[0].m128i_u8[8] = 0x05;
		TrailO[1].m128i_u8[0] = 0x93;
	}
#else
	if (Rnum == 97) {
		// 0x90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  
		//   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 90 00 00 00 00
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0
		//   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[4] = 0x90;
		TrailI[3].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[15] = 0xa0;
		TrailO[2].m128i_u8[0] = 0xc0;
	}
	else if (Rnum == 5) {
		//Rnum:5
		// 0x90000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
		// 0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000c00000000000000000000000700000000000000000

		TrailI[2].m128i_u8[15] = 0x90;
		TrailO[0].m128i_u8[4] = 0x70;
		TrailO[1].m128i_u8[15] = 0xc0;
	}
	else if (Rnum == 12) {
		// 0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000007000000009
		// 0x00000000000000000000000000c000000000000000000000000000000000a0000000000000000000000000000000000000000000000000000000000000000000
		TrailI[0].m128i_u8[0] = 0x0a;
		TrailI[0].m128i_u8[3] = 0x09;
		TrailO[0].m128i_u8[0] = 0x0c;
		TrailO[2].m128i_u8[13] = 0x0a;
	}
	else if (Rnum == 100) {
		// 0x90 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  
		//   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 90 00 00 00 00
		// 0x00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  00 00 00 00 00 00 0c 00 00 00 00 00 00 00 00 00 
		//   00 00 00 00 00 00 00 0a 00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
		TrailI[0].m128i_u8[4] = 0x90;
		TrailI[3].m128i_u8[15] = 0x90;
		TrailO[1].m128i_u8[8] = 0x0a;
		TrailO[2].m128i_u8[9] = 0x0c;

	}

#endif
#endif

	stringstream message;
	message << "Rnum:" << Rnum << ",  Bn:" << BestB[Rnum] << ", UBofWeight:" << Bn << ", Margin : " << Margin << ", RASNUB_ER : " << ASNUB_ER << ", use the merging weights strategy: " << MergeTag << "\n";
	message << "Input:\n0x";
	for (int s = State_NUM - 1; s >= 0; s--) {
		for (int k = 0xf; k >= 0; k--) {
			message << hex << setw(2) << setfill('0') << static_cast<int>(TrailI[s].m128i_u8[k]);
			if (k % 2 == 0) message << " ";
		}
	}
	message << "\nOutput:\n0x";
	for (int s = State_NUM - 1; s >= 0; s--) {
		for (int k = 0xf; k >= 0; k--) {
			message << hex << setw(2) << setfill('0') << static_cast<int>(TrailO[s].m128i_u8[k]);
			if (k % 2 == 0) message << " ";
		}
	}
	message << "\n\n";
	logToFile(fileName, message.str());

	//cluster search
	clock_t Start = clock();
	Round(TrailI, TrailO);
	clock_t End = clock();


	message.str("");
	double ComputeWeightForCluster = 0; long long ComputeTrailNUM = 0;
	for (auto itor = RecordWeightTrailNUM.begin(); itor != RecordWeightTrailNUM.end(); itor++) {
		ComputeWeightForCluster += pow(2, (-1 * itor->first)) * itor->second.Trail_NUM;
		ComputeTrailNUM += itor->second.Trail_NUM;
		printf("Weight: %d, NUM: %lld\n", itor->first, itor->second.Trail_NUM);
		message << dec << "Weight: " << itor->first << ", NUM: " << itor->second.Trail_NUM << "\n";
	}
	printf("Trail_NUM: %lld, ClusterWeight: %f", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))));
	printf("\nTotal Time: %fs, %fmin\n", ((double)(End - Start)) / CLOCKS_PER_SEC, (((double)(End - Start)) / CLOCKS_PER_SEC) / 60);

	message << dec << "Trail_NUM: " << ComputeTrailNUM << ", ClusterWeight: " << (-1 * (log(ComputeWeightForCluster) / log(2))) << endl;
	message << dec << "Total Time: " << ((double)(End - Start)) / CLOCKS_PER_SEC << "s, " << (((double)(End - Start)) / CLOCKS_PER_SEC) / 60 << "min\n\n";
	logToFile(fileName, message.str());

	return;
}

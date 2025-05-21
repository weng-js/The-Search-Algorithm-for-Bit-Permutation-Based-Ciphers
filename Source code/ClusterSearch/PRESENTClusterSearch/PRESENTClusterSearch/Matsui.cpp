 #include<iostream>
#include<emmintrin.h>
#include<nmmintrin.h>
#include<ctime>
#include<string>
#include<sstream>
#include<fstream>
#include<iomanip>
#include "GenTable.h"
#include "State.h"
#include "matsui.h"
#include "GlobleVariables.h"
using namespace std;

ALIGNED_TYPE_(__m128i, 16) Trail[RNUM + 1];        
int t_w[RNUM];
ALIGNED_TYPE_(__m128i, 16) BestTrail[RNUM + 1];     
int Best_w[RNUM];

#if(TYPE)
int BestB[RNUM + 1] = { 0, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120 }; //linear
#else
int BestB[RNUM + 1] = { 0, 2, 4, 8, 12, 20, 24, 28, 32, 36, 41, 46, 52, 56, 62, 66, 70, 74, 78, 82, 86, 90, 96, 100, 106, 110, 116, 120, 124, 128, 132, 136, }; //diff
#endif

int Rnum;
int Bn;
int Trail_Weight;
int ASNUB_ER;
long long RecordTrailNUM;
bool MergeTag;
bool OutputTrailTag;

__m128i count_asn = _mm_setzero_si128();
ALIGNED_TYPE_(__m128i, 16) MaskRound[RNUM + 1]; 
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
				if (WeightForLink + itor->first <= Bn && itor->second) {
					auto it = RecordWeightTrailNUM.find(WeightForLink + itor->first);
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

			if (!(RecordTrailNUM % 10000000)) {
				double ComputeWeightForCluster = 0; long long ComputeTrailNUM = 0;
				for (auto it = RecordWeightTrailNUM.begin(); it != RecordWeightTrailNUM.end(); it++) {
					ComputeWeightForCluster += pow(2, (-1 * it->first)) * it->second.Trail_NUM;
					ComputeTrailNUM += it->second.Trail_NUM;
					printf("Weight: %d, NUM: %lld\n", it->first, it->second.Trail_NUM);
				}
#if(TYPE)
				printf("Trail_NUM: %lld, TrailClusterWeight: %f,  %f\n\n", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))), (-1 * (log(ComputeWeightForCluster) / log(2))) / 2);
#else
				printf("Trail_NUM: %d, TrailClusterWeight: %f\n\n", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))));
#endif
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

map<pair<u64, int>, ValueForCluster> ValueMapCluster; // the info of the first active sbox and the second active sbox, and the round
map<pair<u64, int>, ValueNoForCluster> ValueNoMapCluster; 

void FileOutputTrail() {
#if(TYPE==0)
	string fileName = "result/PRESENT_Diff_ClusterSearch_Trail.txt";
#elif(TYPE==1)
	string fileName = "result/PRESENT_Linear_ClusterSearch_Trail.txt";
#endif

	ALIGNED_TYPE_(__m128i, 16) SO[RNUM + 1];
	memset(SO, 0, sizeof(SO));

	for (int r = 1; r < Rnum; r++) {
		for (int i = 0; i < SBox_NUM; i++) {
			if (BestTrail[r].m128i_u8[i]) {
				SO[r - 1] = _mm_xor_si128(SO[r - 1], INVPTable[i][BestTrail[r].m128i_u8[i]]);
			}
		}
	}
	SO[Rnum - 1] = BestTrail[Rnum];

	stringstream message;
	message << "\nRNUM_" << Rnum << ":  Bn:" << Trail_Weight << endl;
	for (int r = 0; r < Rnum; r++) {
		message << "PO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << (int)BestTrail[r].m128i_u8[k];
		}
		message << "\nSO[" << dec << setw(2) << setfill('0') << r + 1 << "]: 0x";
		for (int k = 0xf; k >= 0; k--) {
			message << hex << (int)SO[r].m128i_u8[k];
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
	s.W			+= s.w;
	Trail_Weight = s.W;
	RecordTrailNUM++;
	if (OutputTrailTag) {
		t_w[s.rnum - 1] = s.w;
		memcpy(BestTrail, Trail, (Rnum + 1) * STATE_LEN);
		memcpy(Best_w, t_w, Rnum * sizeof(int));
		FileOutputTrail();
	}
	auto itor = RecordWeightTrailNUM.find(s.W);
	if (itor != RecordWeightTrailNUM.end()) itor->second.UpdateNUM();
	else {
		WeightForCluster NewWeightRecord;
		RecordWeightTrailNUM.insert(make_pair(s.W, NewWeightRecord));
	}
	if (MergeTag) WeightTrailInfoReorganize();

	if (!(RecordTrailNUM % 10000000)) {
		cout << "Find!" << Trail_Weight << " " << s.W << ":\n";
		double ComputeWeightForCluster = 0; long long ComputeTrailNUM = 0;
		for (auto it = RecordWeightTrailNUM.begin(); it != RecordWeightTrailNUM.end(); it++) {
			ComputeWeightForCluster += pow(2, (-1 * it->first)) * it->second.Trail_NUM;
			ComputeTrailNUM += it->second.Trail_NUM;
			printf("Weight: %d, NUM: %lld\n", it->first, it->second.Trail_NUM);
		}
#if(TYPE)
		printf("Trail_NUM: %lld, TrailClusterWeight: %f,  %f\n\n", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))), (-1 * (log(ComputeWeightForCluster) / log(2))) / 2);
#else
		printf("Trail_NUM: %d, TrailClusterWeight: %f\n\n", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))));
#endif
	}

	return;
}

void FWRound_i(STATE s, __m128i sbx_out) {
	int group_minw = 0;
	for (int i = 0; i < SBox_SIZE; i++) {
		if (s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + BestB[Rnum - s.rnum] > Bn) break;		
		sbx_out = _mm_xor_si128(sbx_out, FWSPTable[s.sbx_a[s.j]][s.sbx_in[s.j]][i]);
		int asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(sbx_out, count_asn)));
		asn = SBox_NUM - asn;

		if ((s.W + s.w + FWWeightOrderW[s.sbx_in[s.j]][i] + asn * weight[1] + BestB[Rnum - s.rnum - 1] > Bn) || asn > ASNUB_ER) continue;

		if (s.rnum >= MaskRoundBG) {
			int tmp_asn = 0;
			tmp_asn = _mm_popcnt_u32(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(sbx_out, MaskRound[s.rnum]), count_asn)));
			if (tmp_asn != SBox_NUM) continue;
		}

		if (FWjudge_state_ri(s, FWWeightOrderW[s.sbx_in[s.j]][i], sbx_out, group_minw)) {
			if (s.j == s.sbx_num) {
				STATE nxt_s = FWupdate_state_row(s, FWWeightOrderW[s.sbx_in[s.j]][i], group_minw, sbx_out);
				if (s.rnum + 1 == Rnum) {
					if (nxt_s.w != -1) FWRound_n(nxt_s);
				}
				else if (nxt_s.W + nxt_s.w + (nxt_s.g_num + 1) * weight[1] + BestB[Rnum - s.rnum - 2] <= Bn) {
					if (nxt_s.rnum >= 4 && nxt_s.rnum <= (Rnum - 4)) {
						auto itorNo = ValueNoMapCluster.find(make_pair((sbx_out.m128i_u64[1] << 4) ^ sbx_out.m128i_u64[0], nxt_s.rnum));
						if (itorNo != ValueNoMapCluster.end() && (Bn - nxt_s.W <= itorNo->second.WeightUB)) continue; 
						int Tmp_RecordTrailNUM = RecordTrailNUM;
						if (asn <= 4) {						
							auto itorYes = ValueMapCluster.find(make_pair((sbx_out.m128i_u64[1] << 4) ^ sbx_out.m128i_u64[0], nxt_s.rnum));
							if (itorYes == ValueMapCluster.end() || itorYes->second.SearchTag(nxt_s.W)) {
								 map<int, WeightForCluster> TmpRecordWeightTrailNUM = RecordWeightTrailNUM;
								__m128i tmp_out = _mm_setzero_si128();
								FWRound_i(nxt_s, tmp_out);

								if (Tmp_RecordTrailNUM != RecordTrailNUM) {
									if (itorYes == ValueMapCluster.end()) {
										ValueForCluster NewValueForCluster(Bn - nxt_s.W);
										ValueMapCluster.insert(make_pair(make_pair((sbx_out.m128i_u64[1] << 4) ^ sbx_out.m128i_u64[0], nxt_s.rnum), NewValueForCluster));
										itorYes = ValueMapCluster.find(make_pair((sbx_out.m128i_u64[1] << 4) ^ sbx_out.m128i_u64[0], nxt_s.rnum));
									}
									else {
										itorYes->second.UpdateWeightUB(Bn - nxt_s.W);
									}
									itorYes->second.ClearWeightValue();
									if (MergeTag) {
										bool MinusTag = false;
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
							}
						}
						else {							
							__m128i tmp_out = _mm_setzero_si128();
							FWRound_i(nxt_s, tmp_out);											
						}
						if (Tmp_RecordTrailNUM == RecordTrailNUM) {
							if (itorNo != ValueNoMapCluster.end()) {
								itorNo->second.UpdateWeightUB(Bn - nxt_s.W);
							}
							else {
								ValueNoForCluster new_ValueNoForCluster(Bn - nxt_s.W);
								ValueNoMapCluster.insert(make_pair(make_pair((sbx_out.m128i_u64[1] << 4) ^ sbx_out.m128i_u64[0], nxt_s.rnum), new_ValueNoForCluster));
							}
						}
					}
					else {
						__m128i tmp_out = _mm_setzero_si128();
						FWRound_i(nxt_s, tmp_out);
					}
				}
			}
			else {
				FWRound_i(FWupdate_state_sbx(s, FWWeightOrderW[s.sbx_in[s.j]][i], group_minw), sbx_out);
			}
		}
	}
	return;
}

void Round(__m128i TrailInput, __m128i TrailOutput) {
	initial_AllTrail();
	Trail[0] = TrailInput; Trail[Rnum] = TrailOutput;
	memset(TrailOutputASBO, 0, sizeof(TrailOutputASBO));
	memset(MaskRound, 0, sizeof(MaskRound));
	TrailOutputASN = 0;

	STATE s = GenStateForRound1(TrailInput);
	__m128i TmpOutput = _mm_setzero_si128();

	for (int i = 0; i < 16; i++) {
		if (TrailOutput.m128i_u8[i]) {
			TrailOutputASBO[i] = TrailOutput.m128i_u8[i];
			TrailOutputASN++;
			TmpOutput = _mm_xor_si128(TmpOutput, INVPTable[i][0xf]);
		}
		else {
			MaskRound[Rnum - 1].m128i_u8[i] = 0xf;
		}
	}

	MaskRoundBG = Rnum - 1; 
	__m128i TmpInput; int record_asn = 0;
	while (record_asn < SBox_NUM) {
		record_asn = 0; MaskRoundBG--;
		TmpInput = TmpOutput; TmpOutput = _mm_setzero_si128();
		for (int j = 0; j < 16; j++) {
			if (TmpInput.m128i_u8[j]) {
				record_asn++;
				TmpOutput = _mm_xor_si128(TmpOutput, INVPTable[j][0xf]);
			}
			else {
				MaskRound[MaskRoundBG].m128i_u8[j] = 0xf;
			}
		}
		if (record_asn == SBox_NUM) {
			MaskRoundBG++; break;
		}
	}

	RecordTrailNUM = 0;
	__m128i sbx_out = _mm_setzero_si128();
	FWRound_i(s, sbx_out);
	return;
}

void matsui() {
	GenTables();
#if(TYPE)
	string fileName = "result/PRESENT_Linear_ClusterSearch.txt";
#else
	string fileName = "result/PRESENT_Diff_ClusterSearch.txt";	
#endif
	RecordWeightTrailNUM.clear();
	__m128i TrailIn = _mm_setzero_si128();
	__m128i TrailOut = _mm_setzero_si128();

	// input info:
	Rnum = 31;                  // the number of rounds for distinguisher
	ASNUB_ER = 4;                 // the maximum number of active sbox per round
	int Margin = 24;
	Bn = BestB[Rnum] + Margin;    // the upper bound on the weight of trails
	MergeTag = true;             // whether to use the merging weights strategy
	OutputTrailTag = false;		  // Whether to output trails that satisfy the conditions for search, only all trails can be output if no strategy is used, otherwise only some trails can be output
	// Other input information that needs to be changed, including differential or linear cluster searches, is changed in GlobleVariables.h.	

	//the input and output for the distinguisher
#if(!TYPE)
	//RNUM:14 round
	TrailIn.m128i_u8[2] = 0x7;
	TrailIn.m128i_u8[14] = 0x7;
	TrailOut.m128i_u8[0] = 0x5;
	TrailOut.m128i_u8[3] = 0x5;
#else
	//RNUM:31 round 
	TrailIn.m128i_u8[5] = 0xa;
	TrailOut.m128i_u8[5] = 0xb;
#endif

	stringstream message;
	message << "Rnum:" << Rnum << ",  Bn:" << BestB[Rnum] << ", UBofWeight:" << Bn << ", Margin : " << Margin << ", RASNUB_ER : " << ASNUB_ER << ", use the merging weights strategy: " << MergeTag << "\n";
	message << "Input:\n0x";
	for (int k = 0xf; k >= 0; k--) {
		message << hex << static_cast<int>(TrailIn.m128i_u8[k]);
	}
	message << "\nOutput:\n0x";
	for (int k = 0xf; k >= 0; k--) {
		message << hex << static_cast<int>(TrailOut.m128i_u8[k]);
	}
	message << "\n\n";
	logToFile(fileName, message.str());

	clock_t Start = clock();
	Round(TrailIn, TrailOut);
	clock_t End = clock();


	message.str("");
	double ComputeWeightForCluster = 0; long long ComputeTrailNUM = 0;
	for (auto itor = RecordWeightTrailNUM.begin(); itor != RecordWeightTrailNUM.end(); itor++) {
		ComputeWeightForCluster += pow(2, (double)(-1 * itor->first)) * itor->second.Trail_NUM;
		ComputeTrailNUM += itor->second.Trail_NUM;
		printf("Weight: %d, NUM: %lld\n", itor->first, itor->second.Trail_NUM);
		message << dec << "Weight: " << itor->first << ", NUM: " << itor->second.Trail_NUM << "\n";
	}
	printf("Trail_NUM: %lld, TrailClusterWeight: %f", ComputeTrailNUM, (-1 * (log(ComputeWeightForCluster) / log(2))));
	printf("\nTotal Time: %fs, %fmin\n", ((double)(End - Start)) / CLOCKS_PER_SEC, (((double)(End - Start)) / CLOCKS_PER_SEC) / 60);

	message << dec << "Trail_NUM: " << ComputeTrailNUM << ", ClusterWeight: " << (-1 * (log(ComputeWeightForCluster) / log(2))) << endl;
	message << dec << "Total Time: " << ((double)(End - Start)) / CLOCKS_PER_SEC << "s, " << (((double)(End - Start)) / CLOCKS_PER_SEC) / 60 << "min\n\n";
	logToFile(fileName, message.str());
	return;
}

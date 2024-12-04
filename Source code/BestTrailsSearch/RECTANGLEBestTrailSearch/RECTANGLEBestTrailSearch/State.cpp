#include<emmintrin.h>
#include<vector>
#include<iostream>
#include<iomanip>
#include<fstream>
#include "GlobleVariables.h"
#include "State.h"

// for trail
void initial_Trail() {
	memset(Trail, 0, RNUM * STATE_LEN);
	memset(TmpBestTrail, 0, RNUM * STATE_LEN);
	memset(TmpNaBestTrail, 0, RNUM * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Tmp_Best_w, 0, sizeof(Tmp_Best_w));
	memset(TmpNaBestw, 0, sizeof(TmpNaBestw));
}

void initial_AllTrail() {
	memset(Trail, 0, RNUM * STATE_LEN);
	memset(BestTrail, 0, RNUM * STATE_LEN);
	memset(TmpBestTrail, 0, RNUM * STATE_LEN);
	memset(TmpNaBestTrail, 0, RNUM * STATE_LEN);
	memset(t_w, 0, sizeof(t_w));
	memset(Best_w, 0, sizeof(Best_w));
	memset(Tmp_Best_w, 0, sizeof(Tmp_Best_w));
	memset(TmpNaBestw, 0, sizeof(TmpNaBestw));
}

// for matsui
STATE GenStateToGenBnUP_FW(__m128i sbx_out[], int NaTag) {
	STATE s(Rnum - 1, 0);
	if (NaTag == 0) {
		s.W = BestB[Rnum - 1] - Best_w[Rnum - 2];
	}
	else if (NaTag == 1) {
		s.W = NaLB[Rnum - 1][0] - Best_w[Rnum - 2];
	}
	else {
		s.W = NaLB[Rnum - 1][1] - Best_w[Rnum - 2];
	}
	s.w = Best_w[Rnum - 2]; 
	ALIGNED_TYPE_(__m128i, 16) tmp_sbx[State_NUM]; memset(tmp_sbx, 0, STATE_LEN);
	for (int j = 0; j < 16; j++) {
		if (sbx_out[0].m128i_u8[j]) {
			s.sbx_a[s.sbx_num] = j;
			s.sbx_in[s.sbx_num] = sbx_out[0].m128i_u8[j];
			for (int k = 0; k < State_NUM; k++) {
				tmp_sbx[k] = _mm_xor_si128(tmp_sbx[k], FWSPTable[s.sbx_a[s.sbx_num]][s.sbx_in[s.sbx_num]][0][k]);
			}
			s.sbx_num++;
		}
	}
	//cout << "num: " << s.sbx_num << " ";
	s.sbx_num--;
	int TmpBn = 0; int count = 0;
	for (int j = 0; j < 16; j++) {
		if (tmp_sbx[0].m128i_u8[j]) {
			TmpBn += FWWeightMinandMax[tmp_sbx[0].m128i_u8[j]][0];
			count++;
		}
	}
	//cout <<"TmpBn: " << TmpBn << endl;
	if (NaTag == 0) Bn = BestB[Rnum - 1] + TmpBn; //初始化
	else if (NaTag == 1) Bn = NaLB[Rnum - 1][0] + TmpBn;
	else {
		if (count >= 2) Bn = NaLB[Rnum - 1][1] + TmpBn;
		else  Bn = NaLB[Rnum - 1][1] + 2 * weight[2];
	}
	return s;
}

STATE GenStateToGenBnUP_BW(__m128i sbx_out[], int NaTag) {
	STATE s(2, 0);
	if (NaTag == 0) {
		s.W = BestB[Rnum - 1] - Best_w[0];
	}
	else if (NaTag == 1) {
		s.W = NaLB[Rnum - 1][0] - Best_w[0];
	}
	else {
		s.W = NaLB[Rnum - 1][1] - Best_w[0];
	}
	s.w = Best_w[0];
	for (int j = 0; j < 16; j++) {
		if (sbx_out[0].m128i_u8[j]) {
			s.sbx_a[s.sbx_num] = j;
			s.sbx_in[s.sbx_num] = sbx_out[0].m128i_u8[j];
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	return s;
}

STATE FWupdate_state_row(STATE s, double w, __m128i sbx_out[]) { //
	s.w += w;
	//记录重量和最优迹
	t_w[s.rnum - 1] = s.w;
	memcpy(Trail[s.rnum], sbx_out, STATE_LEN);
	s.W += s.w;
	s.j = 0;
	s.w = 0; s.sbx_num = 0;
	for (int j = 0; j < 16; j++) {
		if (sbx_out[0].m128i_u8[j]) {
			s.sbx_a[s.sbx_num] = j;
			s.sbx_in[s.sbx_num] = sbx_out[0].m128i_u8[j];
			s.w += FWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	s.rnum++;	return s;
}

STATE BWupdate_state_row(STATE s, double w, __m128i sbx_out[]) { //
	s.w += w;
	//记录重量和最优迹
	Tmp_Best_w[s.rnum - 1] = s.w;
	memcpy(TmpBestTrail[s.rnum - 2], sbx_out, STATE_LEN); //下一轮输入，后续修改，这里是输出
	s.W += s.w;
	s.j = 0;
	s.w = 0; s.sbx_num = 0;
	for (int j = 0; j < 16; j++) {
		if (sbx_out[0].m128i_u8[j]) {
			s.sbx_a[s.sbx_num] = j;
			s.sbx_in[s.sbx_num] = sbx_out[0].m128i_u8[j];
			s.w += BWWeightMinandMax[s.sbx_in[s.sbx_num]][0];
			s.sbx_num++;
		}
	}
	s.sbx_num--;
	s.rnum--;
	return s;
}


STATE update_state_sbx(STATE s, double w) { //这里也需要更新sbx_a_nr
	s.w += w;
	s.j++;
	return s;
}

bool ArrCheck(int arr[], int high, int v) { //二分查找，high是最后一个元素的index
	int low = 0;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (arr[mid] == v) return true;
		else if (arr[mid] > v) high = mid - 1;
		else low = mid + 1;
	}
	return false;
}

void ArrInsert(int arr[], int high, int v) { //提前保证了arr是升序，且v不在arr中
	while (high > 0 && arr[high - 1] > v) {
		arr[high] = arr[high - 1];
		high--;
	}
	arr[high] = v;
}

bool Check_arr_exist(int arr[], int len, Tree t) {
	Node* p = t->lc;
	int index = 1;
	while (index <= len) {
		if (p == NULL) return false;
		while (p->rc != NULL && p->rc->index <= arr[index]) {
			p = p->rc;
		}
		if (p->index == arr[index]) {
			p = p->lc;
			index++;
		}
		else return false; //不存在
	}
	return true;
}

bool CheckArr(int arr[], int len, Tree t) { //检查数组是否需要插入
	// arr[]是新生成的数组，因此它必不会出现在树中，因此只需要检查以后续结点为0的数组
	// SearchOrder index 插入
	int tmp_arr1[Round1_ASN] = { 0 }; //用来记录循环移位后的index
	int index = 1;
	while (index <= len) {
		for (int i = index; i <= len; i++) {
			tmp_arr1[i - index] = (arr[i] - arr[index] + SBox_NUM) % SBox_NUM;
		}
		for (int i = 0; i < index; i++) {
			tmp_arr1[len - index + 1 + i] = (arr[i] - arr[index] + SBox_NUM) % SBox_NUM;
		}
		if (Check_arr_exist(tmp_arr1, len, t)) return false;
		index++;
	}
	return true;
}

void GenPattern(int arr[], int len, int h, Tree* p) { //len为数组长度，h为最多激活的ASN 初始arr[0] = 0; p是上一轮的父节点
	Node* q = (*p);
	if (len + 1 == h) {
		for (int i = arr[len - 1] + 1; i < SBox_NUM; i++) {
			arr[len] = i;
			if (CheckArr(arr, len, T)) {
				Node* s = new Node(i);
				if ((*p)->lc == NULL) {
					(*p)->lc = s;
					q = s;
				}
				else {
					q->rc = s;
					q = s;
				}
			}
		}
		return;
	}
	for (int i = arr[len - 1] + 1; i < SBox_NUM; i++) {
		arr[len] = i;
		if (CheckArr(arr, len, T)) {
			//插入该结点
			Node* s = new Node(i);
			if ((*p)->lc == NULL) {
				(*p)->lc = s;
				q = s;
			}
			else {
				q->rc = s;
				q = s;
			}
			GenPattern(arr, len + 1, h, &s);
		}
	}


}

void GenRound1Pattern() {
	T = new Node(0);
	int arr[Round1_ASN];
	arr[0] = 0;
	GenPattern(arr, 1, Round1_ASN, &T);
	//printfNA2NUM(T);
}

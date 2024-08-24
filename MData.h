#pragma once
#include <iostream>
#include <vector>
using namespace std;
class MData
{
public:
	MData(const int & batch,const int & channels,const int & width,const int &height,const double *data);
	MData(const int & batch, const int & channels, const int & width, const int & height, const double & base_value=0);
	~MData();
	double * data();
	double& operator[] (const int & index);
	double* operator+(const int & index);
	void reset();
	double * operator++();
	double * operator--();
	double * head();
	double * end() { _m_data + _m_batch_num; }
	double *batch(const int & batch);
	double *mat(const int & batch, const int & channel);
	double *row(const int & batch, const int & channel, const int & row);
	double *value(const int &batch, const int & channel, const int &row, const int &col);
	long GetDataNum();
	long GetMatnum();
	long GetRownum();
	long GetBatchIndex() { return _m_batch_num; }
	void safe_delete(double* data)
	{
		if (data == nullptr||data==NULL) return;
		delete[] data;
		data = nullptr;
	}
private:
	double* _m_data=nullptr;
	double* _m_cur = _m_data;
	unsigned long _m_data_num;
	unsigned long _m_batch_num;
	unsigned long _m_mat_num;
	unsigned long _m_row_num;
};


#include "MData.h"
MData::MData(const int & batch,const int & channels,const int & width,const int & height,const double * data)
{
	_m_row_num = width;
	_m_mat_num = _m_row_num * height;
	_m_batch_num = _m_mat_num * channels;
	_m_data_num = _m_batch_num * batch;
	_m_data = new double[_m_data_num] {0};
	_m_cur = _m_data;
	memcpy(_m_data,data,_m_data_num * sizeof(double));
}
MData::MData(const int & batch, const int & channels, const int & width, const int & height,const double & base_value)
{
	_m_row_num = width;
	_m_mat_num = _m_row_num * height;
	_m_batch_num = _m_mat_num * channels;
	_m_data_num = _m_batch_num * batch;
	_m_data = new double[_m_data_num] {base_value};
}
MData::~MData()
{
	safe_delete(_m_data);
}
double& MData::operator[] (const int & index)
{
	return _m_data[index];
}
double* MData::operator+(const int & index)
{
	return _m_data + index;
}
void MData::reset()
{
	_m_cur = _m_data;
}
double * MData::operator++()
{
	return _m_cur++;
}
double * MData::operator--()
{
	return _m_cur;
}
double * MData::head()
{
	return _m_data;
}
double *MData::batch(const int & batch)
{
	return _m_data + batch * _m_batch_num;
}
double *MData::mat(const int & batch, const int & channel)
{
	return _m_data + (batch * _m_batch_num + channel * _m_mat_num);
}
double *MData::row(const int & batch, const int & channel, const int & row)
{
	return _m_data + (batch * _m_batch_num + channel * _m_mat_num + row * _m_row_num);
}
double *MData::value(const int &batch, const int & channel, const int &row, const int &col)
{
	return _m_data + (batch *_m_batch_num + channel * _m_mat_num + row * _m_row_num + col);
}
long MData::GetDataNum()
{
	return _m_data_num;
};
long MData::GetMatnum()
{
	return _m_mat_num;
};
long MData::GetRownum()
{
	return _m_row_num;
};
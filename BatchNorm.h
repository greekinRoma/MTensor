#pragma once
#include "BasicLayer.h"
class BatchNorm: public BasicLayer
{
public:
	BatchNorm(vector<double> weights, vector<double> biases, vector<double> running_mean, vector<double> running_var, const string & weight_name, const string &bias_name, const int & in_width,const int & in_height);
	~BatchNorm();
	MTensor  forward(MTensor inp);
private:
	double * _m_bias;
	double * _m_weight;
	double * _m_running_mean;
	double * _m_running_var;
	int _m_weight_num;
	int _m_bias_num;
	int _m_channel_num;
	string _m_weight_name;
	string _m_bias_name;
	unsigned long mat_num;
};


#include "BatchNorm.h"
BatchNorm::BatchNorm(vector<double> weights, vector<double> biases, vector<double> running_mean,vector<double> running_var,const string & weight_name, const string & bias_name, const int & in_width, const int & in_height):BasicLayer()
{
	_m_weight = new double[weights.size()-2];
	_m_bias = new double[biases.size()-2];
	_m_running_mean = new double[running_mean.size() - 2];
	_m_running_var = new double[running_var.size() - 2];
	memcpy(_m_weight, weights.data()+2, (weights.size()-2) * sizeof(double));
	memcpy(_m_bias, biases.data()+2, (biases.size()-2) * sizeof(double));
	memcpy(_m_running_mean, running_mean.data() + 2, (running_mean.size() - 2) * sizeof(double));
	double eps = 1e-5;
	for (int i = 0; i < running_var.size() - 2; i++)
	{
		running_var[i + 2] = sqrt(running_var[i + 2] + eps);
	}
	memcpy(_m_running_var,running_var.data()+2,(running_var.size()-2)*sizeof(double));
	/*
	for (double weight : weights)
	{
		cout << weight << endl;
	}
	for (double bias : biases)
	{
		cout << bias << endl;
	}
	*/
	_m_weight_num = weights[1];
	_m_bias_num = biases[1];
	if (_m_weight_num != _m_bias_num) throw "%s is not equal to %s", _m_weight_name, _m_bias_name;
	_m_weight_name = weight_name;
	_m_layer_name = bias_name;
	_m_channel_num = biases[1];
	init(in_width,in_height,_m_channel_num,in_width,in_height,_m_channel_num);
	mat_num = in_width * in_height;
}
BatchNorm::~BatchNorm()
{
	delete[] _m_bias;
	delete[] _m_weight;
}
MTensor BatchNorm::forward(MTensor inp)
{
	double eps = 1e-5;
	MData * data_head = inp.GetMData();
	for (int b = 0; b < inp.GetBatchNum(); b++)
	{
		for (int c = 0; c < inp.GetChannelNum(); c++)
		{
			double*data_mat = data_head->mat(b, c);
			for (int i = 0; i < mat_num; i++)
			{
				data_mat[i] = (data_mat[i] - _m_running_mean[c])/_m_running_var[c]*_m_weight[c]+_m_bias[c];
			}
		}
	}
	return inp;
}
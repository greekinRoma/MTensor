#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "./BasicLayer.h"
#include "./loop_list.h"
using namespace std;
class Convolve2d_multi_channel:public BasicLayer
{
public:
	Convolve2d_multi_channel(vector<double> weights, const string & weight_name, const int& stride, const int &padding_height, const int & padding_width, const int &inwidth, const int &inheight,const int & groups=1);
	~Convolve2d_multi_channel();
	void show_weight();
	MTensor forward(MTensor input);
private:
	cv::Mat Conv_layer(const vector<cv::Mat> &inp,double *weight, const cv::Mat &inp_mat);
	vector<int> output_shape;
	string m_weight_name;
	int m_stride;
	int m_padding_height;
	int m_padding_width;
	int m_kernels;
	int m_weight_num;
	int m_layer_weight_num;
	int m_kernel_weight_num;
	int m_groups;
	int m_group_inp_channels;
	int m_group_out_channels;
	unsigned long m_group_layers;
	double * m_kernel_weight;
	std::vector<std::vector<cv::Mat>> m_weights;
	vector<double> m_weight;
	
	int calculate_padding(int in_dim, int stride, int kernel_size);
	int calculate_output_dim(int input_dim, int padding, int kernel_size, int stride);
};
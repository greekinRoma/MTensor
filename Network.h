#pragma once
#include<string>
#include"./Convolve2d_multi_channel.h"
#include"./BatchNorm.h"
#include"./LoadWeight.h"
#include"./Relu.h"
#include"./Relu6.h"
#include"./AdaptiveAvgPool.h"
using namespace std;
class Network
{
public:
	Network(const string & weight_dir,int width,int height);
	~Network();
	void push_conv2d(const string & weight_name, int stride, int paddingx=-1,int paddingy=-1,int group=1);
	void push_batchnorm(const string & batch_name);
	void push_relu();
	void push_relu6();
	void push_adaptiveavgpool(const int &width, const int & height);
	MTensor forward(MTensor out);
	MTensor test(MTensor out);
private:
	vector<BasicLayer*> layers;
	LoadWeight* m_weight_loader;
	Convolve2d_multi_channel* Load_Conv2d(const string & weight_name, int stride = 1 , int paddingx=-1,int paddingy=-1 , int width=256 , int height=256,int group=1);
	BatchNorm * Load_Batchnorm(const string &weight_name, const string & bias_name, const string &running_mean, const string & running_var, const int &width,const int &height);
	int m_width;
	int m_height;
};


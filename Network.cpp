#include "Network.h"
Network::Network(const string & weight_dir,int width, int height)
{
	m_weight_loader = new LoadWeight(weight_dir);
	m_width = width;
	m_height = height;
}
Network::~Network()
{
	delete m_weight_loader;
	for (BasicLayer *layer : layers)
	{
		delete layer;
	}
}
Convolve2d_multi_channel* Network::Load_Conv2d(const string & weight_name, int stride , int paddingx,int paddingy, int width, int height,int groups)
{
	return new Convolve2d_multi_channel(m_weight_loader->load_weight(weight_name), weight_name, stride, paddingx,paddingy, width, height,groups);
}
BatchNorm * Network::Load_Batchnorm(const string &weight_name, const string & bias_name,const string &running_mean,const string & running_var,const int &width,const int &height)
{
	//cout <<"ahsfdoiahspodfh"<< running_mean << endl;
	//cout << width << endl;
	return new BatchNorm(m_weight_loader->load_weight(weight_name),m_weight_loader->load_weight(bias_name),m_weight_loader->load_weight(running_mean),m_weight_loader->load_weight(running_var),weight_name,bias_name,width,height);
}
void Network::push_conv2d(const string & weight_name, int stride, int paddingx,int paddingy,int groups)
{
	BasicLayer *layer = Load_Conv2d(weight_name, stride, paddingx, paddingy, m_width, m_height,groups);
	m_width = layer->GetOutWidth();
	m_height = layer->GetOutHeight();
	layers.push_back(layer);
}
void Network::push_adaptiveavgpool(const int &width,const int & height)
{
	BasicLayer *layer = new AdaptiveAvgPool(width,height);
	layers.push_back(layer);
}
void Network::push_batchnorm(const string & batch_name)
{
	string weight_name = batch_name + "weight.bin";
	string bias_name = batch_name + "bias.bin";
	string running_mean_path = batch_name + "running_mean.bin";
	string running_var_path = batch_name + "running_var.bin";
	BasicLayer* layer = Load_Batchnorm(weight_name,bias_name,running_mean_path,running_var_path,m_width,m_height);
	m_width = layer->GetOutWidth();
	m_height = layer->GetOutHeight();
	layers.push_back(layer);
}
void Network::push_relu()
{
	BasicLayer* layer = new Relu(m_width, m_height);
	layers.push_back(layer);
}
void Network::push_relu6()
{
	BasicLayer * layer = new Relu6(m_width, m_height);
	layers.push_back(layer);
}
MTensor Network::test(MTensor out)
{
	clock_t start, end;
	
	for (BasicLayer *layer : layers)
	{
		start = clock();
		out = layer->forward(out);

		end = clock();
		std::cout << layer->GetName()<<"F1运行时间" << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

	}

	out.show();
	//cv::Mat img = out.GetMat(0, 2);
	//cv::normalize(img, img, 1., 0., cv::NORM_MINMAX);
	//cv::imshow("tmp_img", img);
	//cv::waitKey(1000000000);
	return out;
}

MTensor Network::forward(MTensor out) 
{
	for (BasicLayer* layer : layers)
	{
		out = layer->forward(out);
	}
	return out;
}

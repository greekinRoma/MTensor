#include "AdaptiveAvgPool.h"
AdaptiveAvgPool::AdaptiveAvgPool(const int & out_width, const int & out_height):BasicLayer()
{
	m_out_width = out_width;
	m_out_height = out_height;
}
AdaptiveAvgPool::~AdaptiveAvgPool()
{

}
MTensor AdaptiveAvgPool::forward(MTensor inp)
{
	int num_batch = inp.GetBatchNum();
	int num_channel = inp.GetChannelNum();
	MTensor output(num_batch,num_channel,m_out_width,m_out_height);
	int kernel_width_size = inp.Width() / m_out_width;
	int kernel_height_size = inp.Height() / m_out_height;
	unsigned int kernel_num = kernel_height_size * kernel_width_size;
	if (inp.Width() % kernel_width_size)
	{
		cout << "Error in Adaptive Avage Pooling in Width!" << endl;
		throw;
	}
	if (inp.Height() % kernel_width_size)
	{
		cout << "Error in Adaptive Avage Pooling in Width!" << endl;
	}
	double * out_double = inp.GetMData()->head();
	double * inp_double = inp.GetMData()->head();
	for (unsigned long i = 0; i < num_batch*num_channel*m_out_height*m_out_width;i++)
	{
		for (unsigned long j = 0; j < kernel_num; j++)
		{
			(*out_double) += (*inp_double);
			inp_double++;
		}
		(*out_double) = (*out_double)/kernel_num;
		out_double++;
	}
	return output;
}
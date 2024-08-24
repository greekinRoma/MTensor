#pragma once
#include"./BasicLayer.h"
class Relu:public BasicLayer
{
public:
	Relu(const int &in_width,const int &in_height);
	~Relu();
	MTensor forward(MTensor inp)
	{
		long data_num = inp.GetDataNum();
		double * data_double = inp.GetDouble();
		for (long i = 0; i < data_num; i++)
		{
			*data_double = *data_double>0?*data_double:0;
			data_double++;
		}
		return inp;
	}
};


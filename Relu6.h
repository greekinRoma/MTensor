#pragma once
#include"./BasicLayer.h"
class Relu6 :public BasicLayer
{
public:
	Relu6(const int &in_width, const int &in_height)
	{
		init(in_width, in_height, 1, in_width, in_height, 1);
		_m_layer_name = "relu6";
	};
	~Relu6();
	MTensor forward(MTensor inp)
	{
		long data_num = inp.GetDataNum();
		double * data_double = inp.GetDouble();
		for (long i = 0; i < data_num; i++)
		{
			*data_double = *data_double > 0 ? *data_double : 0;
			*data_double = *data_double < 6 ? *data_double : 6;
			data_double++;
		}
		return inp;
	}
};

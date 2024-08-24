#pragma once
#include "./BasicLayer.h"
class AdaptiveAvgPool:public BasicLayer
{
public:
	AdaptiveAvgPool(const int & out_width,const int & out_height);
	~AdaptiveAvgPool();
	MTensor forward(MTensor inp);
private:
	int m_out_width;
	int m_out_height;
};
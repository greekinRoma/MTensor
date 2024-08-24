#pragma once
#include<opencv2/opencv.hpp>
#include "./MTensor.h"
class BasicLayer
{
public:
	BasicLayer() {};
	~BasicLayer() {};
	void init(int inwidth, int inheight, int inchannels, int outwidth, int outheight, int outchannels);
	virtual MTensor forward(MTensor input) { return input; };
	int GetInchannels()
	{
		return m_inchannels;
	}
	int GetInwidth()
	{
		return m_inwidth;
	}
	int GetInHeight()
	{
		return m_inheight;
	}
	int GetOutchannels()
	{
		return m_outchannels;
	}
	int GetOutHeight()
	{
		return m_outheight;
	}
	int GetOutWidth()
	{
		return m_outwidth;
	}
	string GetName()
	{
		return _m_layer_name;
	}
protected:
	int m_inchannels;
	int m_outchannels;
	int m_inwidth;
	int m_inheight;
	int m_outwidth;
	int m_outheight;
	string _m_layer_name = "BasicLayer";
};
#include "BasicLayer.h"
void BasicLayer::init(int inwidth, int inheight,int inchannels,int outwidth,int outheight,int outchannels)
{
	m_inchannels = inchannels;
	m_outchannels = outchannels;
	m_inwidth = inwidth;
	m_inheight = inheight;
	m_outwidth = outwidth;
	m_outheight = outheight;
}
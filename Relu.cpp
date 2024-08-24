#include "Relu.h"
Relu::Relu(const int &in_width, const int &in_height):BasicLayer()
{
	init(in_width, in_height,1, in_width, in_height, 1);
	_m_layer_name = "relu";
}
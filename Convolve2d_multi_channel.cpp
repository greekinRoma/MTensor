#include "Convolve2d_multi_channel.h"
Convolve2d_multi_channel::Convolve2d_multi_channel(vector<double> weights,const string & weight_name, const int& stride, const int &padding_height,const int & padding_width,const int &inwidth,const int &inheight,const int &groups):BasicLayer()
{
	if (weights[0] != 4) throw "error in the weight of %s!!!!",weight_name;
	//cout << weights[0] << "," << weights[1] << "," << weights[2] << "," << weights[3] << "," << weights[4] << endl;
	m_kernels = (int)weights[3];
	int inchannels = (int)weights[2];
	int outchannels = (int)weights[1];
	if (padding_width == -1)
	{
		m_padding_width = calculate_padding(inwidth, stride, m_kernels);
	}
	else
	{
		m_padding_width = padding_width;
	}
	if (padding_height == -1)
	{
		m_padding_height = calculate_padding(inheight, stride, m_kernels);
	}
	else
	{
		m_padding_height = padding_height;
	}
	//cout << inwidth << m_padding_width << endl;
	int outwidth = calculate_output_dim(inwidth, m_padding_width, m_kernels, stride);
	int outheight = calculate_output_dim(inheight, m_padding_height, m_kernels, stride);
	//cout << outwidth << outheight << endl;
	m_stride = stride;
	/*算法的基本信息*/
	_m_layer_name = weight_name;
	///
	m_weight_num = weights.size() - weights[0] - 1;
	m_layer_weight_num = m_kernels * m_kernels*inchannels;
	m_kernel_weight_num = m_kernels * m_kernels;
	m_groups = groups;
	m_group_inp_channels = inchannels;
	m_group_out_channels = outchannels / groups;
	//m_group_layers = m_layer_weight_num * m_group_inp_channels;
	//cout << "jdofahs" << m_layer_weight_num << endl;
	//m_group_in_channels = in_channels / groups;
	/*初始化*/
	init(inwidth,inheight,inchannels*m_groups,outwidth,outheight,outchannels);
	m_kernel_weight = (double *)malloc(sizeof(double)*m_weight_num);
	//m_kernel_weight = new double[m_weight_num];
	memcpy(m_kernel_weight, weights.data() + (int)weights[0] + 1, sizeof(double)*m_weight_num);
}
Convolve2d_multi_channel::~Convolve2d_multi_channel()
{
	delete[] m_kernel_weight;
}
int Convolve2d_multi_channel::calculate_output_dim(int input_dim, int padding, int kernel_size, int stride)
{
	return (input_dim + padding - kernel_size) / stride + 1;
}
int Convolve2d_multi_channel::calculate_padding(int in_dim, int stride, int kernel_size) {
	int out_dim = (in_dim + stride - 1) / stride; 
	int padding = (out_dim - 1) * stride + kernel_size - in_dim;
	padding = padding > 0 ? padding : 0;
	return padding ;
}
MTensor Convolve2d_multi_channel::forward(MTensor input)
{
	input.padding(m_padding_width, m_padding_height);

	MData * _m_inp_data = input.GetMData();
	size_t batch_num = input.GetBatchNum();
	MTensor output(batch_num, m_outchannels, m_outwidth, m_outheight);
	//cout << m_outwidth << m_outheight << endl;
	//if (m_kernels == 1) cin.get();
	MData * out_data = output.GetMData();
	MData * inp_data = input.GetMData();
	double * out_double = out_data->head();
	double * inp_double = inp_data->head();
	/*总共遍历的流程*/
	unsigned long inp_mat_num = inp_data->GetMatnum();
	unsigned long out_mat_num = out_data->GetMatnum();
	unsigned long out_row = out_data->GetRownum();
	unsigned long inp_row = inp_data->GetRownum();
	unsigned long inp_channels = input.GetChannelNum();

	/*用于遍历参数的检测头*/
	double * out_channel;
	double * in_channel;
	double * out_pixel;
	double * inp_pixel;
	double * wei_pixel;
	double wei_tmp;
	vector<vector<double*>> m_loop_lists;
	vector<double*> m_loop_list;
	inp_pixel = inp_double;
	m_group_layers = inp_mat_num * m_group_inp_channels;
	for (unsigned int j = 0; j < m_groups; j++)
	{
		vector<double*> m_loop_list;
		for (unsigned int i = 0; i < m_kernel_weight_num; i++)
		{
			m_loop_list.push_back(inp_pixel + i % m_kernels + i / m_kernels * inp_data->GetRownum());
		}
		m_loop_lists.push_back(m_loop_list);
		inp_pixel = inp_pixel+ m_group_layers;
	}
	wei_pixel = m_kernel_weight;
	out_channel = out_double;
	for (unsigned int batch_index = 0; batch_index < batch_num; batch_index++)
	{
		for (unsigned int group_index = 0; group_index < m_groups; group_index++)
		{
			for (unsigned int out_channel_index = 0; out_channel_index <m_group_out_channels; out_channel_index++)
			{
				m_loop_list = m_loop_lists[group_index];
				for (int in_channel_index = 0; in_channel_index < m_group_inp_channels; in_channel_index++)
				{
					//if (m_kernel_weight_num == 1) { cout << m_kernel_weight_num << "  " << "mat_update" << out_channel - out_double << endl; cin.get(); }
					for (unsigned long weight_pixel_index = 0; weight_pixel_index < m_kernel_weight_num; weight_pixel_index++)
					{
						out_pixel = out_channel;
						inp_pixel = m_loop_list[weight_pixel_index];
						m_loop_list[weight_pixel_index] = m_loop_list[weight_pixel_index] + inp_mat_num;
						wei_tmp = *wei_pixel;
						for (unsigned long inp_pixel_index = 0; inp_pixel_index < out_mat_num; inp_pixel_index++)
						{
							//if (m_kernel_weight_num == 1) cout <<"pixel_update" << out_pixel - out_double << endl;
							(*out_pixel) += wei_tmp * (*inp_pixel);
							out_pixel++;
							//if (m_stride==1)cout <<"out_channel"<<out_row<<"          "<<m_stride<<"           "<< inp_pixel -inp_double<< endl;
							if (inp_pixel_index != 0 && (inp_pixel_index + 1) % out_row == 0) inp_pixel = inp_pixel + m_kernels + inp_row * (m_stride - 1);
							else inp_pixel += m_stride;
						}
						wei_pixel++;
						//if (m_stride == 1) cin.get();
					}
				}
				out_channel = out_channel + out_mat_num;
			}
		}
	}
	return output;
}


void Convolve2d_multi_channel::show_weight()
{
	for (double w:m_weight)
	{
		cout<<m_weight.size() <<"||||"<< w << endl;
	}
}
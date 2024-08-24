#include "MTensor.h"
MTensor::MTensor(cv::Mat  inp)
{
	//cout << inps[0].channels() << ends;
	m_batch_num = 1;
	m_channels = inp.channels();
	m_height = inp.rows;
	m_width = inp.cols;
	if (_m_data != NULL) delete _m_data;
	_m_data = NULL;
	_m_data=new MData(m_batch_num, m_channels, m_width, m_height,(double*)inp.data);
}
MTensor::MTensor(const string &file_dir,const string &file_name)
{
	LoadWeight data_loader(file_dir);
	vector<double> data_inp = data_loader.load_weight(file_name);
	m_batch_num = data_inp[1];
	m_channels = data_inp[2];
	m_width = data_inp[3];
	m_height = data_inp[4];
	_m_data = new MData(m_batch_num, m_channels, m_width, m_height, (double*)data_inp.data() + 5);
}
MTensor::MTensor(const int & batch, const int &channels, const int &width, const int &height)
{
	m_batch_num = batch;
	m_channels = channels;
	m_height = height;
	m_width = width;
	if (_m_data != NULL) delete _m_data;
	_m_data = NULL;
	_m_data = new MData(m_batch_num,m_channels,m_width,m_height);
}
MTensor::MTensor(MTensor * inp)
{
	m_batch_num = inp->GetBatchNum();
	m_channels = inp->GetChannelNum();
	m_height = inp->Height();
	m_width = inp->Width();
	if (_m_data != NULL) delete _m_data;
	_m_data = NULL;
	_m_data = new MData(m_batch_num,m_channels,m_width,m_height,inp->GetMData()->head());
}
MTensor::~MTensor()
{
	//if (_m_data != NULL) delete _m_data;
	//_m_data = NULL;
}

MData* MTensor::GetMData() { return _m_data; };
void MTensor::padding(int padding_x, int padding_y)
{
	MData *_m_padding_mat = new MData(m_batch_num, m_channels, m_width + padding_x, m_height + padding_y, 0.);
	padding_x = padding_x / 2 + padding_x % 2;
	padding_y = padding_y / 2 + padding_y % 2;
	for (int i = 0; i < m_batch_num; i++)
	{
		for (int j = 0; j < m_channels; j++)
		{
			for (int k = 0; k < m_width; k++)
			{
				memcpy(_m_padding_mat->row(i,j,k+padding_y)+padding_x,_m_data->row(i,j,k),_m_data->GetRownum()*sizeof(double));
				//cout << "dddd" << endl;
			}
		}
	}
	delete _m_data;
	m_width = m_width + padding_x;
	m_height = m_height + padding_y;
	_m_data = _m_padding_mat;
}
double MTensor::GetValue(const int & batch, const int & channel, const int & row, const int& col)
{
	return *_m_data->value(batch, channel, row, col);
}
cv::Mat MTensor::GetMat(const int &batch, const int &channel)
{
	cv::Mat mat = cv::Mat::zeros(cv::Size(m_height, m_width), CV_64FC1);
	memcpy(mat.data, _m_data->mat(batch, channel), _m_data->GetMatnum()*sizeof(double));
	return mat;
}
vector<cv::Mat> MTensor::GetBatch(const int & batch)
{
	vector<cv::Mat> outcome(m_channels, cv::Mat::zeros(cv::Size(m_height, m_width), CV_64FC1));
	for (int i = 0; i < m_channels; i++)
	{
		outcome.push_back(GetMat(batch,i));
	}
	return outcome;
}
vector<vector<cv::Mat>> MTensor::GetDataset()
{
	vector<vector<cv::Mat>> outcome(m_batch_num, vector<cv::Mat>(m_channels, cv::Mat::zeros(cv::Size(m_height, m_width), CV_64FC1)));
	for (int i = 0; i < m_batch_num; i++)
	{
		outcome.push_back(GetBatch(i));
	}
	return outcome;
}
void MTensor::show(bool is_start)
{
	for (int i = 0; i < GetBatchNum(); ++i) {
		std::cout << "Batch " << i << ":" << std::endl;
		for (int j = 0; j < GetChannelNum(); ++j) {
			std::cout << "  Channel " << j << ": "<<std::endl;
			for (int k = 0; k < GetHieghtNum(); ++k) {
				std::cout << "Row" << k << ":";
				for (int l = 0; l < GetWithNum(); ++l) {
					std::cout << GetValue(i, j, k, l) << " ";
				}
				std::cout << std::endl; 
			}
			if (is_start) cin.get();
		}
	}
}
MTensor& MTensor::operator+(const MTensor & inp)
{
	double* itr = this->GetMData()->head();
	double* tmp_itr = inp._m_data->head();
	if (_m_data->GetDataNum() != inp._m_data->GetDataNum())
	{
		cout << "The add operator is not equal!!!!!!!!!!"<<endl;
		throw;
	}
	for (unsigned long i = 0; i < _m_data->GetDataNum(); i++)
	{
		(*itr) = (*itr) + (*tmp_itr);
		itr++;
		tmp_itr++;
	}
	return *this;
}

//vector<vector<double>> MTensor::adaptive_avg_pool2d(const MTensor &inp)
//{
//	int n = inp.m_batch_num;
//	int c = inp.m_channels;
//	int h = inp.m_height;
//	int w = inp.m_width;
//	int num = inp._m_data->GetDataNum();
//	vector<vector<vector<vector<double>>>> input(n, vector<vector<vector<double>>>(c, vector<vector<double>>(h, vector<double>(w, 0))));
//	//vector<vector<vector<vector<double>>>> output(n, vector<vector<vector<double>>>(c, vector<vector<double>>(out_height, vector<double>(out_width, 0))));
//	vector<vector<double>> output(n, vector<double>(c, 0));
//	
//	double* inp_itr = inp._m_data->head();
//	for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < c; ++j) {
//			double patch_sum = 0.0;
//			for (int k = 0; k < h; ++k) {
//				for (int l = 0; l < w; ++l) {
//					//input[i][j][k][l] = *inp_itr;
//					patch_sum += *inp_itr;
//					inp_itr++;
//				}
//			}
//			double mean = patch_sum / (h * w);
//			output[i][j] = mean;
//
//		}
//	}
//	double* oup_itr = this->GetMData()->head();
//	for (int i = 0; i < n; i++) {
//		for (int j = 0; j < c; j++) {
//			*oup_itr = output[i][j];
//		}
//	}
//	return output;
//	
//	//double stride_h = h / out_height;
//	//double stride_w = w / out_width;	
//
//	//for (int y = 0; y < out_height; y++) {
//	//	for (int x = 0; x < out_width; x++) {
//	//		int start_y = floor(y * stride_h);
//	//		int end_y = ceil((y + 1) * stride_h);
//	//		int start_x = floor(x * stride_w);
//	//		int end_x = ceil((x + 1) * stride_w);
//
//	//		for (int i = 0; i < n; i++) {
//	//			for (int j = 0; j < c; j++) {
//	//				double patch_sum = 0.0;
//	//				for (int m = start_y; m < end_y; m++) {
//	//					for (int n = start_x; n < end_x; n++) {
//	//						patch_sum += input[i][j][m][n];
//	//					}
//	//				}
//	//				double mean = patch_sum / ((end_y - start_y) * (end_x = start_x));
//	//				int output_index = 
//
//	//				}
//	//			}
//	//		}
//	//	}
//	
//}

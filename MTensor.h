#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "./MData.h"
#include "./LoadWeight.h"
using namespace std;

class MTensor
{
public:
	MTensor(cv::Mat TMat);
	MTensor(const string &file_path,const string &file_name);
	MTensor(const int &batch, const int &channels, const int &width, const int &height);
	MTensor(MTensor * inp);
	~MTensor();
	const int GetBatchNum() { return m_batch_num; }
	int GetDataNum() { return _m_data->GetDataNum(); }
	int GetWithNum() { return m_width; }
	int GetHieghtNum() { return m_height; }
	const int GetChannelNum() { return m_channels; }
	long GetChannelIndex() { return _m_data->GetBatchIndex(); }
	MData* GetMData();
	double* GetDouble() { return _m_data->head(); }
	void padding(int padding_x, int padding_y);
	double GetValue(const int & batch, const int & channel, const int & row, const int& col);
	cv::Mat GetMat(const int &batch, const int &channel);
	vector<cv::Mat> GetBatch(const int & batch);
	vector<vector<cv::Mat>> GetDataset();
	void show(bool is_start=true);
	const int Width() { return m_width; };
	const int Height() { return m_height; };

	//vector<vector<double>> adaptive_avg_pool2d(const MTensor &input);
	/*жиди*/
	MTensor& operator+(const MTensor & inp);
private:
	int m_batch_num;
	int m_channels;
	int m_width;
	int m_height;
	MData* _m_data = NULL;
};


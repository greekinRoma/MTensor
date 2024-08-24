#include "LoadWeight.h"
LoadWeight::LoadWeight(const string weight_dir)
{
	m_weight_dir = weight_dir;
}
LoadWeight::~LoadWeight()
{
}
void LoadWeight::print_weight()
{

}
bool LoadWeight::load_weights()
{
	DIR * pDir;
	struct dirent * ptr;
	if (!(pDir = opendir(m_weight_dir.c_str())))
	{
		cout << "Folder does not Exist" << endl;
		return false;
	}
	while ((ptr = readdir(pDir)) != 0)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) continue;
		if (load_weight_file(m_weight_dir + "/" + ptr->d_name, ptr->d_name) == false)
		{
			printf("The %s is an error!!!", ptr->d_name);
			return false;
		}
	}
	return true;
}
bool LoadWeight::load_weight_file(const string &weight_file_path,const string & weight_name)
{
	//cout << weight_file_path << endl;
	ifstream inF;
	inF.open(weight_file_path,std::ifstream::binary);
	if (!inF.is_open())
	{
		printf("Read File: %s Error ....\n",weight_file_path.c_str());
		return false;
	}
	// 获取文件大小
	inF.seekg(0, std::ifstream::end);
	//将这个数据转化为long
	long size = inF.tellg();
	//inF的数据大小
	inF.seekg(0);
	//将输入流文件重新设置为0
	//buffer.resize(size);
	//printf("文件:[%s] 共有：%ld(字节) ...... \n", weight_file_path.c_str(),size);
	vector<double> m_buffer;
	m_buffer.resize(size / 8);
	inF.read((char*)&m_buffer[0],size);
	//cout << m_buffer.size() << endl;
	m_weights.insert({ weight_name,m_buffer });
	///*
	//for (int i=0; i < size / 8; i++)
	//{
	//	cout << m_weights[weight_name][i] << endl;
	//}
	//.*/
	
	inF.close();
	
	return true;
}
vector<double> LoadWeight::load_weight(const string &weight_name)
{
	m_weights.clear();
	if (load_weight_file(m_weight_dir + "\\" + weight_name, weight_name) == false)
	{
		printf("The %s is an error!!!", weight_name);
		throw "Load the weight of %s error!!!",weight_name;
	}
	cout << weight_name << endl;
	return m_weights[weight_name];
}
unordered_map<string, vector<double>> LoadWeight::GetWeights()
{
	return m_weights;
}
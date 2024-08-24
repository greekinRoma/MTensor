#ifndef WEIGHTLOADER_H
#define WEIGHTLOADER_H
#include<String>
#include<iostream>
#include<fstream>
#include<vector>
#include<unordered_map>
#include <iostream>
#include "./dirent.h"
using namespace std;
class LoadWeight
{
public:
	LoadWeight(const string weight_dir);
	~LoadWeight();
	bool load_weights();
	unordered_map<string, vector<double>> GetWeights();
	vector<double> load_weight(const string &weight_name);
private:
	string m_weight_dir;
	void print_weight();
	//vector<char> buffer = {};
	//double *m_buffer;
	unordered_map<string, vector<double>> m_weights;
	bool load_weight_file(const string &weight_file, const string & weight_name);
};
#endif WEIGHTLOADER_H
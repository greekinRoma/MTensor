#pragma once
struct Node
{
	Node * next_node=nullptr;
	double * data;
};
class LoopList
{
public:
	LoopList(int node_num) 
	{
		m_node_num = node_num+1;
		m_node = new Node[node_num+1];
		cur_node = m_node;
		for (int i = 0; i < node_num; i++)
		{
			cur_node->next_node = cur_node+1;
			cur_node++;
		}
		cur_node->next_node = m_node->next_node;
		cur_node = cur_node->next_node;
	};
	~LoopList()
	{
		//delete[m_node_num] m_node;
	};
	Node* operator[](const int &i)
	{
		return m_node+ i + 1;
	}
	void rest_node()
	{
		cur_node = m_node->next_node;
		for (double * node : save_origin)
		{
			//cout <<save_origin.size()<<","<< node << endl;
			//cout << cur_node << endl;
			cur_node->data = node;
			cur_node=cur_node->next_node;
		}
	}
	void clear()
	{
		save_origin.clear();
	}
	void next()
	{
		cur_node = cur_node->next_node;
	}
	void operator=(double * double_node)
	{
		cur_node->data = double_node;
	}
	void push(double * double_node)
	{
		save_origin.push_back(double_node);
	}
	int Get_num()
	{
		return m_node_num;
	}
	double * Get()
	{
		return cur_node->data;
	}
private:
	Node * m_node;
	Node * cur_node;
	int m_node_num;
	vector<double*> save_origin;
};
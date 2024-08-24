#include <iostream>
#include "Network.h"
#include "MTensor.h"

float dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2);
vector<double> adaptive_avg_pool2d(const MTensor& inp);

int main()
{
	std::cout << "Let's Start!!!!!!!!! \n";
	//string weight_dir = "C:\\Users\\YHT\\Desktop\\yuhaitao\\ConsoleApplication1\\ConsoleApplication1\\mobilenetv2";
	//string img_dir = "C:\\Users\\YHT\\Desktop\\yuhaitao\\ConsoleApplication1\\ConsoleApplication1\\save_img";
	string weight_dir = "C:\\Users\\27227\\Desktop\\ConsoleApplication1\\ConsoleApplication1\\ConsoleApplication1\\mobilenetv2";
	string img_dir = "C:\\Users\\27227\\Desktop\\ConsoleApplication1\\ConsoleApplication1\\ConsoleApplication1\\save_img";
	MTensor inp_tensor(img_dir, "img.dat");

	//self.features[0][1]部分
	Network net1(weight_dir, inp_tensor.Width(), inp_tensor.Height());
	
	net1.push_conv2d("features.0.0.weight.bin", 2);
	net1.push_batchnorm("features.0.1.");
	net1.push_relu6();
	net1.push_conv2d("features.1.conv.0.weight.bin", 1, -1, -1, 32);
	net1.push_batchnorm("features.1.conv.1.");
	net1.push_relu6();
	net1.push_conv2d("features.1.conv.3.weight.bin", 1);
	net1.push_batchnorm("features.1.conv.4.");

	//self.features[2]部分
	net1.push_conv2d("features.2.conv.0.weight.bin", 1);
	net1.push_batchnorm("features.2.conv.1.");
	net1.push_relu6();
	net1.push_conv2d("features.2.conv.3.weight.bin", 2, -1, -1, 96);
	net1.push_batchnorm("features.2.conv.4.");
	net1.push_relu6();
	net1.push_conv2d("features.2.conv.6.weight.bin", 1);
	net1.push_batchnorm("features.2.conv.7.");

	MTensor add_out_2_7 = net1.forward(inp_tensor);
	MTensor out_2_7(&add_out_2_7);
	//out_2_7.show();

	//self.features[3]部分
	Network net2(weight_dir, out_2_7.Width(), out_2_7.Height());
	net2.push_conv2d("features.3.conv.0.weight.bin", 1);
	net2.push_batchnorm("features.3.conv.1.");
	net2.push_relu6();
	net2.push_conv2d("features.3.conv.3.weight.bin", 1, -1, -1, 144);
	net2.push_batchnorm("features.3.conv.4.");
	net2.push_relu6();
	net2.push_conv2d("features.3.conv.6.weight.bin", 1);
	net2.push_batchnorm("features.3.conv.7.");
	//add_out_2_7.show();
	MTensor add_out_3_7 = net2.forward(out_2_7);
	//out_3_7.show();
	MTensor add_out_3 = add_out_3_7 + add_out_2_7;
	MTensor out_3(&add_out_3);
	//out_3.show();

	//self.features[4]部分
	Network net3(weight_dir, out_3.Width(), out_3.Height());
	net3.push_conv2d("features.4.conv.0.weight.bin", 1);
	net3.push_batchnorm("features.4.conv.1.");
	net3.push_relu6();
	net3.push_conv2d("features.4.conv.3.weight.bin", 2, -1, -1, 144);
	net3.push_batchnorm("features.4.conv.4.");
	net3.push_relu6();
	net3.push_conv2d("features.4.conv.6.weight.bin", 1);
	net3.push_batchnorm("features.4.conv.7.");

	MTensor add_out_4_7 = net3.forward(out_3);
	MTensor out_4_7(&add_out_4_7);

	//self.features[5]部分
	Network net4(weight_dir, out_4_7.Width(), out_4_7.Height());
	net4.push_conv2d("features.5.conv.0.weight.bin", 1);
	net4.push_batchnorm("features.5.conv.1.");
	net4.push_relu6();
	net4.push_conv2d("features.5.conv.3.weight.bin", 1, -1, -1, 192);
	net4.push_batchnorm("features.5.conv.4.");
	net4.push_relu6();
	net4.push_conv2d("features.5.conv.6.weight.bin", 1);
	net4.push_batchnorm("features.5.conv.7.");

	MTensor add_out_5_7 = net4.forward(out_4_7);
	MTensor add_out_5 = add_out_4_7 + add_out_5_7;
	MTensor out_5(&add_out_5);
	//out_5.show();

	//self.features[6]部分
	Network net5(weight_dir, out_5.Width(), out_5.Height());
	net5.push_conv2d("features.6.conv.0.weight.bin", 1);
	net5.push_batchnorm("features.6.conv.1.");
	net5.push_relu6();
	net5.push_conv2d("features.6.conv.3.weight.bin", 1, -1, -1, 192);
	net5.push_batchnorm("features.6.conv.4.");
	net5.push_relu6();
	net5.push_conv2d("features.6.conv.6.weight.bin", 1);
	net5.push_batchnorm("features.6.conv.7.");

	MTensor add_out_6_7 = net5.forward(out_5);
	MTensor add_out_6 = add_out_6_7 + add_out_5;
	MTensor out_6(&add_out_6);
	//out_6.show();

	//self.features[7]部分
	Network net6(weight_dir, out_6.Width(), out_6.Height());
	net6.push_conv2d("features.7.conv.0.weight.bin", 1);
	net6.push_batchnorm("features.7.conv.1.");
	net6.push_relu6();
	net6.push_conv2d("features.7.conv.3.weight.bin", 2, -1, -1, 192);
	net6.push_batchnorm("features.7.conv.4.");
	net6.push_relu6();
	net6.push_conv2d("features.7.conv.6.weight.bin", 1);
	net6.push_batchnorm("features.7.conv.7.");
	
	MTensor add_out_7_7 = net6.forward(out_6);
	MTensor out_7_7(&add_out_7_7);

	//self.features[8]部分
	Network net7(weight_dir, out_7_7.Width(), out_7_7.Height());
	net7.push_conv2d("features.8.conv.0.weight.bin", 1);
	net7.push_batchnorm("features.8.conv.1.");
	net7.push_relu6();
	net7.push_conv2d("features.8.conv.3.weight.bin", 1, -1, -1, 384);
	net7.push_batchnorm("features.8.conv.4.");
	net7.push_relu6();
	net7.push_conv2d("features.8.conv.6.weight.bin", 1);
	net7.push_batchnorm("features.8.conv.7.");

	MTensor add_out_8_7 = net7.forward(out_7_7);
	MTensor add_out_8 = add_out_7_7 + add_out_8_7;
	MTensor out_8(&add_out_8);
	//out_8.show();

	//self.features[9]部分
	Network net8(weight_dir, out_8.Width(), out_8.Height());
	net8.push_conv2d("features.9.conv.0.weight.bin", 1);
	net8.push_batchnorm("features.9.conv.1.");
	net8.push_relu6();
	net8.push_conv2d("features.9.conv.3.weight.bin", 1, -1, -1, 384);
	net8.push_batchnorm("features.9.conv.4.");
	net8.push_relu6();
	net8.push_conv2d("features.9.conv.6.weight.bin", 1);
	net8.push_batchnorm("features.9.conv.7.");

	MTensor add_out_9_7 = net8.forward(out_8);
	MTensor add_out_9 = add_out_9_7 + add_out_8;
	MTensor out_9(&add_out_9);
	//out_9.show();

	//self.features[10]部分
	Network net9(weight_dir, out_9.Width(), out_9.Height());
	net9.push_conv2d("features.10.conv.0.weight.bin", 1);
	net9.push_batchnorm("features.10.conv.1.");
	net9.push_relu6();
	net9.push_conv2d("features.10.conv.3.weight.bin", 1, -1, -1, 384);
	net9.push_batchnorm("features.10.conv.4.");
	net9.push_relu6();
	net9.push_conv2d("features.10.conv.6.weight.bin", 1);
	net9.push_batchnorm("features.10.conv.7.");

	MTensor add_out_10_7 = net9.forward(out_9);
	MTensor add_out_10 = add_out_10_7 + add_out_9;
	MTensor out_10(&add_out_10);
	//out_10.show();

	//self.features[11]部分
	Network net10(weight_dir, out_10.Width(), out_10.Height());
	net10.push_conv2d("features.11.conv.0.weight.bin", 1);
	net10.push_batchnorm("features.11.conv.1.");
	net10.push_relu6();
	net10.push_conv2d("features.11.conv.3.weight.bin", 1, -1, -1, 384);
	net10.push_batchnorm("features.11.conv.4.");
	net10.push_relu6();
	net10.push_conv2d("features.11.conv.6.weight.bin", 1);
	net10.push_batchnorm("features.11.conv.7.");

	MTensor add_out_11_7 = net10.forward(out_10);
	MTensor out_11_7(&add_out_11_7);

	//self.features[12]部分
	Network net11(weight_dir, out_11_7.Width(), out_11_7.Height());
	net11.push_conv2d("features.12.conv.0.weight.bin", 1);
	net11.push_batchnorm("features.12.conv.1.");
	net11.push_relu6();
	net11.push_conv2d("features.12.conv.3.weight.bin", 1, -1, -1, 576);
	net11.push_batchnorm("features.12.conv.4.");
	net11.push_relu6();
	net11.push_conv2d("features.12.conv.6.weight.bin", 1);
	net11.push_batchnorm("features.12.conv.7.");

	MTensor add_out_12_7 = net11.forward(out_11_7);
	MTensor add_out_12 = add_out_11_7 + add_out_12_7;
	MTensor out_12(&add_out_12);
	//out_12.show();

	//self.features[13]部分
	Network net12(weight_dir, out_12.Width(), out_12.Height());
	net12.push_conv2d("features.13.conv.0.weight.bin", 1);
	net12.push_batchnorm("features.13.conv.1.");
	net12.push_relu6();
	net12.push_conv2d("features.13.conv.3.weight.bin", 1, -1, -1, 576);
	net12.push_batchnorm("features.13.conv.4.");
	net12.push_relu6();
	net12.push_conv2d("features.13.conv.6.weight.bin", 1);
	net12.push_batchnorm("features.13.conv.7.");

	MTensor add_out_13_7 = net12.forward(out_12);
	MTensor add_out_13 = add_out_13_7 + add_out_12;
	MTensor out_13(&add_out_13);
	//out_13.show();

	//self.features[14]部分
	Network net13(weight_dir, out_13.Width(), out_13.Height());
	net13.push_conv2d("features.14.conv.0.weight.bin", 1);
	net13.push_batchnorm("features.14.conv.1.");
	net13.push_relu6();
	net13.push_conv2d("features.14.conv.3.weight.bin", 2, -1, -1, 576);
	net13.push_batchnorm("features.14.conv.4.");
	net13.push_relu6();
	net13.push_conv2d("features.14.conv.6.weight.bin", 1);
	net13.push_batchnorm("features.14.conv.7.");

	MTensor add_out_14_7 = net13.forward(out_13);
	MTensor out_14_7(&add_out_14_7);

	//self.features[15]部分
	Network net14(weight_dir, out_14_7.Width(), out_14_7.Height());
	net14.push_conv2d("features.15.conv.0.weight.bin", 1);
	net14.push_batchnorm("features.15.conv.1.");
	net14.push_relu6();
	net14.push_conv2d("features.15.conv.3.weight.bin", 1, -1, -1, 960);
	net14.push_batchnorm("features.15.conv.4.");
	net14.push_relu6();
	net14.push_conv2d("features.15.conv.6.weight.bin", 1);
	net14.push_batchnorm("features.15.conv.7.");

	MTensor add_out_15_7 = net14.forward(out_14_7);
	MTensor add_out_15 = add_out_15_7 + add_out_14_7;
	MTensor out_15(&add_out_15);
	//out_15.show();

	//self.features[16]部分
	Network net15(weight_dir, out_15.Width(), out_15.Height());
	net15.push_conv2d("features.16.conv.0.weight.bin", 1);
	net15.push_batchnorm("features.16.conv.1.");
	net15.push_relu6();
	net15.push_conv2d("features.16.conv.3.weight.bin", 1, -1, -1, 960);
	net15.push_batchnorm("features.16.conv.4.");
	net15.push_relu6();
	net15.push_conv2d("features.16.conv.6.weight.bin", 1);
	net15.push_batchnorm("features.16.conv.7.");

	MTensor add_out_16_7 = net15.forward(out_15);
	MTensor add_out_16 = add_out_16_7 + add_out_15;
	MTensor out_16(&add_out_16);
	//out_16.show();

	//self.features[17]部分
	Network net16(weight_dir, out_16.Width(), out_16.Height());
	net16.push_conv2d("features.17.conv.0.weight.bin", 1);
	net16.push_batchnorm("features.17.conv.1.");
	net16.push_relu6();
	net16.push_conv2d("features.17.conv.3.weight.bin", 1, -1, -1, 960);
	net16.push_batchnorm("features.17.conv.4.");
	net16.push_relu6();
	net16.push_conv2d("features.17.conv.6.weight.bin", 1);
	net16.push_batchnorm("features.17.conv.7.");

	MTensor add_out_17_7 = net16.forward(out_16);
	MTensor out_17_7(&add_out_17_7);
	//out_17_7.show();

	//self.conv部分
	Network net17(weight_dir, out_17_7.Width(), out_17_7.Height());
	net17.push_conv2d("conv.0.weight.bin", 1);
	net17.push_batchnorm("conv.1.");
	net17.push_relu6();

	MTensor add_out = net17.forward(out_17_7);
	MTensor out(&add_out);
	out = out + add_out;
	out.show();

	
//	//最后部分的推理内容
//	std::ifstream weight_file(weight_dir + "\\classifier.weight.bin", std::ios::binary);
//	if (!weight_file.is_open()) {
//		std::cerr << "Failed to open weight.bin" << std::endl;
//		return -1;
//	}
//
//	// 读取 bias.bin 文件
//	std::ifstream bias_file(weight_dir + "\\classifier.bias.bin", std::ios::binary);
//	if (!bias_file.is_open()) {
//		std::cerr << "Failed to open bias.bin" << std::endl;
//		return -1;
//	}
//
//	// 从 weight.bin 读取权重数据
//	std::vector<std::vector<double>> weights;
//	int num_rows, num_cols;
//	weight_file.read(reinterpret_cast<char*>(&num_rows), sizeof(int));
//	weight_file.read(reinterpret_cast<char*>(&num_cols), sizeof(int));
//
//	weights.resize(num_rows, std::vector<double>(num_cols));
//	for (int i = 0; i < num_rows; ++i) {
//		weight_file.read(reinterpret_cast<char*>(weights[i].data()), num_cols * sizeof(double));
//	}
//
//	// 从 bias.bin 读取偏置数据
//	std::vector<double> bias(num_rows);
//	bias_file.read(reinterpret_cast<char*>(bias.data()), num_rows * sizeof(double));
//
//	// 关闭文件
//	weight_file.close();
//	bias_file.close();
//
//	std::vector<double> input = adaptive_avg_pool2d(out);
//
//	std::vector<double> output_vector(num_rows);
//	for (int i = 0; i < num_rows; ++i) {
//		double dot_result = dotProduct(input, weights[i]);
//		output_vector[i] = dot_result + bias[i];
//	}
//
//	// 输出结果向量
//	std::cout << "Output vector:" << std::endl;
//	for (double val : output_vector) {
//		std::cout << val << " ";
//	}
//	std::cout << std::endl;
//	//std::cout << out_3_7.GetBatchNum() << std::endl;
//	//std::cout << out_3_7.GetChannelNum() << std::endl;
//	//std::cout << out_3_7.GetWithNum() << std::endl;
//	//std::cout << out_3_7.GetHieghtNum() << std::endl;
//
//	//cv::Mat out_mat = out.GetMat(0, 0);
//	//cv::imshow("out_mat", out_mat);
//	//cv::waitKey(10000);
//}
//
//float dotProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
//	double result = 0.0f;
//	for (size_t i = 0; i < vec1.size(); ++i) {
//		result += vec1[i] * vec2[i];
//	}
//	return result;
//}
//
//vector<double> adaptive_avg_pool2d(const MTensor& inp)
//{
//	int n = inp.m_batch_num;
//	int c = inp.m_channels;
//	int h = inp.m_height;
//	int w = inp.m_width;
//	int num = inp._m_data->GetDataNum();
//	vector<vector<vector<vector<double>>>> input(n, vector<vector<vector<double>>>(c, vector<vector<double>>(h, vector<double>(w, 0))));
//	//vector<vector<vector<vector<double>>>> output(n, vector<vector<vector<double>>>(c, vector<vector<double>>(out_height, vector<double>(out_width, 0))));
//	vector<double> output(c, 0);
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
//			for (int index = 0; index < c; index++) {
//				output[index] = mean;
//			}
//		}
//	}
//	return output;
}
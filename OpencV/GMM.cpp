#include "GMM.h"
#include <limits>
#include <iostream>
using namespace std;

Component::Component()
{
	mean = cv::Vec3d::all(0);
	cov = cv::Matx33d::zeros();
	inverseCov = cv::Matx33d::zeros();
	covDeterminant = 0;
	totalPiexelCount = 0;
}

// copy
Component::Component(const cv::Mat &modelComponent)
{
	mean = modelComponent(cv::Rect(0, 0, 3, 1));
	cov = modelComponent(cv::Rect(3, 0, 9, 1)).reshape(1, 3);
	covDeterminant = cv::determinant(cov);
	inverseCov = cov.inv();
}



int Component::getCompPixelCount() const
{
	return this->totalPiexelCount;
}


/*************
* 将高斯分布以参数向量的方式输出
*/
cv::Mat Component::exportModel() const
{
	cv::Mat meanMat = cv::Mat(mean.t()); // 3*1
	cv::Mat covMat = cv::Mat(cov).reshape(1, 1); // 改为1通道，1行, 9*1
	cv::Mat model;
	cv::hconcat(meanMat, covMat, model); // 将两个矩阵水平合并
	return model;// 12*1
}


GMM::GMM(int componentsCount)
	: components(componentsCount), coefs(componentsCount), totalSampleCount(0)
{
}


/*********
* Model 是混合高斯模型，其实是一个一维向量，含若干个分量，每个分量13个参数
* 复制高斯混合模型
*/
GMM GMM::matToGMM(const cv::Mat &model)
{
	//一个像素的（唯一对应）高斯模型的参数个数或者说一个高斯模型的参数个数  
	//一个像素RGB三个通道值，故3个均值，3*3个协方差，共用一个权值  
	const int paraNumOfComponent = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/; // 平均数，协方差，权重， 这是模型参数的数量
	if ((model.type() != CV_64FC1) || (model.rows != 1) || (model.cols % paraNumOfComponent != 0))
		CV_Error(CV_StsBadArg, "model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount");
	int componentCount = model.cols / paraNumOfComponent; // 计算分量数量
	GMM result(componentCount);
	for (int i = 0; i < componentCount; i++) {
		cv::Mat componentModel = model(cv::Rect(13*i, 0, paraNumOfComponent, 1));
		result.coefs[i] = componentModel.at<double>(0, 0);
		result.components[i] = Component(componentModel(cv::Rect(1, 0, 12, 1)));
	}
	return result;
}

/******
* 将高斯模型转化为矩阵 (13*n)*1,依此是系数，协方差，权重
*/
cv::Mat GMM::GMMtoMat() const
{
	cv::Mat result;
	for (size_t i = 0; i < components.size(); i++) {
		cv::Mat coefMat(1, 1, CV_64F, cv::Scalar(coefs[i]));
		cv::Mat componentMat = components[i].exportModel();
		cv::Mat combinedMat;
		cv::hconcat(coefMat, componentMat, combinedMat);
		if (result.empty()) {
			result = combinedMat; // 13*1
		}
		else {
			cv::hconcat(result, combinedMat, result); // 拼接矩阵到result
		}
	}
	return result;
}


/**********
* 获得分量的数量
*/
int GMM::getComponentsCount() const
{
	return components.size();
}


//计算一个像素（由color=（B,G,R）三维double型向量来表示）属于这个GMM混合高斯模型的概率。  
//也就是把这个像素像素属于componentsCount个高斯模型的概率与对应的权值相乘再相加，  
//具体见论文的公式（10）。结果从res返回。  
//这个相当于计算Gibbs能量的第一个能量项（取负后）。

double GMM::operator()(const cv::Vec3d color) const
{
	double res = 0;
	for (size_t i = 0; i < components.size(); i++) {
		if (coefs[i] > 0)
			res += coefs[i] * components[i](color);
	}
	return res;
}

/****************
* 多元高斯分布计算概率密度
*/
double Component::operator()(const cv::Vec3d &color) const
{
	double PI = 3.1415926;
	double n = 3;
	cv::Vec3d diff = color - mean;
	double res = 1.0 / (pow(2 * PI, n / 2)*sqrt(covDeterminant)) * exp(-0.5 * (diff.t() * inverseCov * diff)(0));
	return res;
}

/**************
* 查看该像素属于5个高斯分量中的哪个分量,返回最大那个
*/
int GMM::mostPossibleComponent(const cv::Vec3d color) const
{
	int maxid = 0;
	double maxPossible = 0;
	for (int i = 0; i < components.size(); i++) {
		if (components[i](color) >= maxPossible) {
			maxid = i;
			maxPossible = components[i](color);
		}
	}
	return maxid;
}

/****
* 初始化混合高斯模型，清零
*/
void GMM::initLearning()
{
	for (int ci = 0; ci < components.size(); ci++) {
		components[ci].initLearning();
		coefs[ci] = 0;
	}
	totalSampleCount = 0;
}

/****
* 初始化高斯模型，清零
*/
void Component::initLearning()
{
	mean = cv::Vec3d::all(0);
	cov = cv::Matx33d::zeros();
	totalPiexelCount = 0;
}


void Component::addPixel(cv::Vec3d color)
{
	mean += color;
	cov += color * color.t(); // 多元协方差
	totalPiexelCount++;
}

void GMM::addPixel(int ci, const cv::Vec3d color)
{
	components[ci].addPixel(color);
	totalSampleCount++;
}


void GMM::endLearning()
{
	for (int ci = 0; ci < components.size(); ci++) {
		int n = components[ci].getCompPixelCount();
		if (n == 0) {
			coefs[ci] = 0;
		}
		else {
			coefs[ci] = (double)n / totalSampleCount;// 权重按照高斯模型摄入的像素数量决定
			components[ci].endLearning();
		}
	}
}


void Component::endLearning()
{
	const double variance = 0.01;
	mean /= totalPiexelCount;
	cov = (1.0 / totalPiexelCount) * cov;
	cov -= mean * mean.t();
	const double det = cv::determinant(cov);
	if (det <= std::numeric_limits<double>::epsilon()) {
		cov += variance * cv::Matx33d::eye();
	}
	covDeterminant = cv::determinant(cov);
	inverseCov = cov.inv();
}
#pragma once
//
// Created by geshuaiqi on 6/1/18.
//

#ifndef GMM_H
#define GMM_H
#include <opencv2/core/core.hpp>
#include <vector>

// 分量
class Component
{
public:
	Component();
	Component(const cv::Mat &modelComponent);
	cv::Mat exportModel() const;
	void initLearning();
	void addPixel(cv::Vec3d color);
	void endLearning();
	double operator()(const cv::Vec3d &color) const;
	int getCompPixelCount() const;
private:
	cv::Vec3d mean;			// 一个像素RGB三个通道，因此三个均值，3*3个协方差，共用一个权值
	cv::Matx33d cov;		// 协方差
	cv::Matx33d inverseCov; // 协方差的逆矩阵
	double covDeterminant;  // 协方差的行列式
	int totalPiexelCount;
};

// 混合高斯模型
class GMM
{
public:
	GMM(int componentsCount = 5);
	void initLearning();
	void addPixel(int compID, const cv::Vec3d color);
	void endLearning();
	static GMM matToGMM(const cv::Mat &model);
	cv::Mat GMMtoMat() const;
	int getComponentsCount() const;
	double operator()(const cv::Vec3d color) const;
	int mostPossibleComponent(const cv::Vec3d color) const;

private:
	std::vector<Component> components;
	std::vector<double> coefs; // Pi_k
	int totalSampleCount;

};


#endif //GMM_H
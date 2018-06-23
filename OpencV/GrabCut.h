#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
class GMM;
template <typename captype, typename tcaptype, typename flowtype> class Graph;
class GrabCut2D
{
public:
	void GrabCut( const cv::Mat &img, cv::Mat &mask, cv::Rect rect,
		cv::Mat &bgdModel,cv::Mat &fgdModel,
		int iterCount, int mode );  

	~GrabCut2D(void);
private:
    void initMaskWithRect(cv::Mat &mask, cv::Size size, cv::Rect rect);
    void initGMMs(const cv::Mat &img, const cv::Mat &mask, GMM &fgdGMM, GMM &bgdGMM);
    double computeBeta(const cv::Mat &img);
    void computeEdgeWeights( const cv::Mat& img, cv::Mat& leftW, cv::Mat& upleftW,
                       cv::Mat& upW, cv::Mat& uprightW, double beta, double gamma);
    void assignGMMsComponents( const cv::Mat& img, const cv::Mat& mask,
                          const GMM& bgdGMM, const GMM& fgdGMM, cv::Mat& compIdxs);
    void learnGMMs( const cv::Mat& img, const cv::Mat& mask, const cv::Mat& compIdxs,
                          GMM& bgdGMM, GMM& fgdGMM);
    void buildGraph( const cv::Mat& img, const cv::Mat& mask, const GMM& bgdGMM,
                           const GMM& fgdGMM, double lambda, const cv::Mat& leftW,
                           const cv::Mat& upleftW, const cv::Mat& upW, const cv::Mat& uprightW,
                           Graph<double, double, double>& graph );
    void estimateSegmentation( Graph<double, double, double>& graph, cv::Mat& mask );

};


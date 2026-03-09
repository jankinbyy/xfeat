#ifndef XFEAT_H
#define XFEAT_H

#include "XFModel.h"
#include "InterpolateSparse2d.h"
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <tuple>
#include <filesystem>
#include <Eigen/Dense>

namespace XFeat
{
    class XFDetector
    {
    public:
        XFDetector(int _top_k=4096, float _detection_threshold=0.05, bool use_cuda=true);
        void detectAndCompute(torch::Tensor &x, std::unordered_map<std::string, torch::Tensor> &result);
        void match(torch::Tensor &feats1, torch::Tensor &feats2, torch::Tensor &idx0, torch::Tensor &idx1, float _min_cossim=-1.0);
        void match_xfeat(cv::Mat &img1, cv::Mat &img2, cv::Mat &mkpts_0, cv::Mat &mkpts_1);
        void xfeat_keypoints_descritors(cv::Mat &img1,std::unordered_map<std::string, at::Tensor> &out1);
        void matchWithFLANN(const cv::Mat &desc1,const cv::Mat &desc2,const cv::Mat &kpts1,const cv::Mat &kpts2,cv::Mat &mkpts_0,cv::Mat &mkpts_1,float ratio);
        cv::Mat tensorToCVMat(const torch::Tensor &desc);
        torch::Tensor parseInput(cv::Mat &img);
        std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor &x);
        cv::Mat tensorToMat(const torch::Tensor &tensor);
        
    private:
        torch::Tensor getKptsHeatmap(torch::Tensor &kpts, float softmax_temp=1.0);
        torch::Tensor NMS(torch::Tensor &x, float threshold = 0.05, int kernel_size = 5);
        std::string getWeightsPath(std::string weights);
        
        std::string weights;
        int top_k;
        float min_cossim;
        float detection_threshold;
        torch::DeviceType device_type;
        std::shared_ptr<XFeatModel> model;
        std::shared_ptr<InterpolateSparse2d> bilinear, nearest;

        // BriefDatabase db;
        // BriefVocabulary *voc;
    };

} // namespace XFeat


#endif // XFEAT_H
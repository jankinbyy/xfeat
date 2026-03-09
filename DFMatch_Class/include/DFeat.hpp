#pragma once
#include "dnn/hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>
//#include <omp.h>

using namespace std;
struct KeyPoint
{
    int x, y;
    float score;
};
struct pair_hash
{
    template <class T1, class T2> size_t operator()(pair<T1, T2> const& pair) const
    {
        size_t h1 = hash<T1>()(pair.first);
        size_t h2 = hash<T2>()(pair.second);
        return h1 ^ h2;
    }
};
using Grid = std::unordered_map<std::pair<int, int>, std::vector<KeyPoint>, pair_hash>;

class DFeat
{
public:
    DFeat(std::string model_path_sp, std::string model_path_lg);
    ~DFeat() {}
    int Match(std::vector<cv::KeyPoint>& keypoint_1, Eigen::MatrixXd& desc_1, std::vector<cv::KeyPoint>& keypoint_2, Eigen::MatrixXd& desc_2, std::vector<cv::DMatch>& matches);
    void MatchCos(const Eigen::MatrixXd& descs1_full, const Eigen::MatrixXd& descs2_full, std::vector<cv::DMatch>& matches, double minScore = 0.9);
    void filter_matches(Eigen::MatrixXd& scores,                    // log assignment matrix, shape [M, N]
                        std::vector<std::pair<int, int>>& matches,  // output: matched pairs (i, j)
                        std::vector<double>& mscores                // output: matching scores (exp of log score)
    );
    Eigen::MatrixXd sigmoid_log_double_softmax(Eigen::MatrixXd& sim, Eigen::VectorXd& z0, Eigen::VectorXd& z1);
    Eigen::VectorXd log_softmax(Eigen::VectorXd& x);
    Eigen::MatrixXd log_softmax_cols(Eigen::MatrixXd& x);
    Eigen::MatrixXd log_softmax_rows(Eigen::MatrixXd& x);
    cv::Mat change_img_size(int target_height, int target_width, const cv::Mat& in_img);
    std::vector<cv::Point2f> NormalizeKeypoints(std::vector<cv::KeyPoint> kpts, int h, int w);
    int DetectAndCompute(cv::Mat& bgr_mat_, std::vector<cv::KeyPoint>& _key_point, Eigen::MatrixXd& _desc);
    void InterpDescriptor(const float* descMat, float* descriptor, float ptx, float pty);
    int32_t read_image_2_tensor_as_gray_dfeat(cv::Mat& bgr_mat, hbDNNTensor* input_tensor);
    int prepare_tensor(hbDNNTensor* input_tensor, hbDNNTensor* output_tensor, hbDNNHandle_t dnn_handle);
    std::vector<KeyPoint> applyNMS_grid_new(const std::vector<KeyPoint>& keypoints, float threshold, int windowSize, int image_height = 480);
    std::vector<cv::Point2f> applyNMS_grid(const std::vector<KeyPoint>& keypoints, float threshold, int windowSize);
    std::pair<int, int> hashKeyPoint(const KeyPoint& kp);
    bool Load_Vocabluary(std::string& vocab_sp_path, std::string& vocab_lg_path);

private:
    hbPackedDNNHandle_t packed_dnn_handle_sp;
    hbDNNHandle_t dnn_handle_sp;
    const char** model_name_list_sp;
    bool flag_init_sp = false;
    float point_th_high = 0.0120;
    float point_th_low = 0.007;
    int windowSize = 5;
    hbPackedDNNHandle_t packed_dnn_handle_lg;
    hbDNNHandle_t dnn_handle_lg;
    const char** model_name_list_lg;
    bool flag_init_lg = false;
    const float GRID_SIZE = 10.0f;
    const int model_hight_ = 480;
    const int model_width_ = 640;
    const int max_feature_ = 256;  //特征点最多提取多少
    const int min_feature_ = 256;  //特征点最少提取多少，根据模型输入决定
    const int desc_dim_ = 256;     //描述子维度
    int Wd8_ = 640 / 8;
};

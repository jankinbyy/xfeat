#include "DFeat.hpp"
void drawMatchesLite(const cv::Mat &img1, const std::vector<cv::KeyPoint> &kps1,
                     const cv::Mat &img2, const std::vector<cv::KeyPoint> &kps2,
                     const std::vector<cv::DMatch> &matches, cv::Mat &outImg,
                     const cv::Scalar &color = cv::Scalar(0, 255, 0)) {
  // 1️⃣ 创建拼接图像
  int rows = std::max(img1.rows, img2.rows);
  int cols = img1.cols + img2.cols;
  outImg = cv::Mat::zeros(rows, cols, CV_8UC3);

  // 确保图像是三通道
  cv::Mat img1_color, img2_color;
  if (img1.channels() == 1)
    cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
  else
    img1_color = img1.clone();

  if (img2.channels() == 1)
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
  else
    img2_color = img2.clone();

  img1_color.copyTo(outImg(cv::Rect(0, 0, img1.cols, img1.rows)));
  img2_color.copyTo(outImg(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

  // 2️⃣ 绘制匹配连线
  for (const auto &m : matches) {
    const cv::Point2f &pt1 = kps1[m.queryIdx].pt;
    const cv::Point2f &pt2 = kps2[m.trainIdx].pt;

    // 匹配点颜色随机
    cv::Scalar c(rand() % 255, rand() % 255, rand() % 255);

    cv::Point2f p1 = pt1;
    cv::Point2f p2 = cv::Point2f(pt2.x + img1.cols, pt2.y);

    // 画连线
    cv::line(outImg, p1, p2, c, 1, cv::LINE_AA);

    // 画圆圈
    cv::circle(outImg, p1, 3, c, -1);
    cv::circle(outImg, p2, 3, c, -1);
  }
}

int main(int argc, char *argv[]) {
  std::cout << "This is the version of two" << std::endl;
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << "./dglue input_image_dir output_image_dir\n";
    return 1; // 返回非零值表示程序执行失败
  }
  std::vector<Eigen::VectorXf> desc_vec;
  std::string img_folder_name = argv[1];
  std::string img_match_vis_name = argv[2];
  DFeat dfeat("./dfeat.bin", "./lg_dfeat_kp192.bin");
  for (int i = 1; i < 10000; i++) {
    for (int k = i + 1; k < 10000; k++) {
      std::string img_name_1 = img_folder_name + std::to_string(i) + ".jpg";
      std::cout << img_name_1 << std::endl;
      std::string img_name_2 = img_folder_name + std::to_string(k) + ".jpg";
      std::cout << img_name_2 << std::endl;

      cv::Mat bgr_mat_1 = cv::imread(img_name_1, cv::IMREAD_COLOR);
      cv::Mat bgr_mat_2 = cv::imread(img_name_2, cv::IMREAD_COLOR);

      if (bgr_mat_1.empty() && bgr_mat_2.empty()) {
        std::cout << "over done!" << std::endl;
        return 0;
      }
      if (bgr_mat_1.empty() || bgr_mat_2.empty()) {
        std::cout << "done!" << std::endl;
        break;
      }
      std::vector<cv::KeyPoint> kp1;
      Eigen::MatrixXd feat1;
      auto feat_detect_start = std::chrono::steady_clock::now();
      dfeat.DetectAndCompute(bgr_mat_1, kp1, feat1);
      auto feat_detect_end = std::chrono::steady_clock::now();
      auto time_detect_all =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              feat_detect_end - feat_detect_start)
              .count() *
          1000;
      std::cout << "\033[31m"
                << "dfeat feature detect use all time :" << time_detect_all
                << "\033[0m" << std::endl;
      std::vector<cv::KeyPoint> kp2;
      Eigen::MatrixXd feat2;
      dfeat.DetectAndCompute(bgr_mat_2, kp2, feat2);
      auto match_start_cos = std::chrono::steady_clock::now();
      std::vector<cv::DMatch> matches;
      dfeat.MatchCos(feat1, feat2, matches, 0.8f);
      auto match_end = std::chrono::steady_clock::now();
      std::cout << "match size:" << matches.size() << std::endl;
      std::cout << "\033[31m"
                << "dfeat feature cos match use all time :"
                << std::chrono::duration_cast<std::chrono::duration<double>>(
                       match_end - match_start_cos)
                           .count() *
                       1000
                << "\033[0m" << std::endl;
      if (matches.size() > 30) {
        auto match_start_1 = std::chrono::steady_clock::now();
        std::vector<cv::DMatch> matches1;
        dfeat.Match(kp1, feat1, kp2, feat2, matches1);
        auto match_start = std::chrono::steady_clock::now();
        std::cout << "\033[31m"
                  << "dfeat feature match use all time :"
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         match_start - match_start_1)
                             .count() *
                         1000
                  << "\033[0m" << std::endl;
        cv::Mat imgHoriz;
        cv::hconcat(bgr_mat_1, bgr_mat_2, imgHoriz);
        int width1 = bgr_mat_1.cols;  // 图1宽度，用来计算偏移
        for (int j = 0; j < matches1.size(); j++) {
          cv::Point point_img1;
          cv::Point point_img2;
          point_img1.x = (int)kp1[matches1[j].queryIdx].pt.x;
          point_img1.y = (int)kp1[matches1[j].queryIdx].pt.y;
          point_img2.x = (int)kp2[matches1[j].trainIdx].pt.x+ width1;
          point_img2.y = (int)kp2[matches1[j].trainIdx].pt.y;
          cv::circle(imgHoriz, point_img1, 2, cv::Scalar(255, 0, 0), -1);
          cv::circle(imgHoriz, point_img2, 2, cv::Scalar(255, 0, 0), -1);
          cv::line(imgHoriz, point_img1, point_img2, cv::Scalar(0, 0, 255), 1);
        }
        std::string img_match_vis_path =
            img_match_vis_name + "match__" + std::to_string(i) + "__" +
            std::to_string(k) + "_" + std::to_string(matches1.size()) + "_" +
            ".png";
        std::cout << img_match_vis_path << std::endl;
        cv::imwrite(img_match_vis_path, imgHoriz);
      }
      // 假设 bgr_mat_1 和 bgr_mat_2 是原始图像，kp1 和 kp2 是关键点，matches
      // 是匹配结果
      cv::Mat img1_draw = bgr_mat_1.clone();
      cv::Mat img2_draw = bgr_mat_2.clone();

      // 在图像1上绘制所有关键点（黑色）
      for (const auto &kp : kp1) {
        cv::circle(img1_draw, kp.pt, 3, cv::Scalar(0, 0, 0), cv::FILLED);
      }

      // 在图像2上绘制所有关键点（黑色）
      for (const auto &kp : kp2) {
        cv::circle(img2_draw, kp.pt, 3, cv::Scalar(0, 0, 0), cv::FILLED);
      }

      // 可选：把两张图拼接显示（左右拼接）
      cv::Mat imgMatches;
      cv::hconcat(img1_draw, img2_draw, imgMatches);
      drawMatchesLite(img1_draw, kp1, img2_draw, kp2, matches, imgMatches);
      // 在左上角写匹配点数
      std::string text = "Matches: " + std::to_string(matches.size());
      cv::putText(imgMatches, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0, 0, 255), 2);

      // 保存图片
      std::string img_match_vis_path_cos =
          img_match_vis_name + "match__" + std::to_string(i) + "__" +
          std::to_string(k) + "_" + std::to_string(matches.size()) + "_cos.png";
      cv::imwrite(img_match_vis_path_cos, imgMatches);
    }
  }
}

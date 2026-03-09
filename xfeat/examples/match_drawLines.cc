#include "XFeat.h"
#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
namespace fs = std::filesystem;
// 从文件名里解析时间戳（假设格式是 1756797762.949794.jpg）
struct Relocation_msg {
  int index = 0;
  double timestamp = 0.0;
  cv::Mat image;
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  std::unordered_map<std::string, at::Tensor> feature;
  std::vector<cv::Point3f> depth_points;
  Relocation_msg(double timestamp, cv::Mat image, Eigen::Matrix4d pose)
      : timestamp(timestamp), image(image), pose(pose) {}
  Relocation_msg(){}
  void clear(){
    depth_points.clear();
    feature.clear();
  }
};
class KeyBoardSimulator {
public:
  KeyBoardSimulator(std::string input_image):image_dir(input_image) {
    // 创建 XFeat 对象
    detector = std::make_shared<XFeat::XFDetector>(top_k, detection_threshold,use_cuda);
    T_wheel_cam <<  0.0005208273, -0.0006440467,  1.001,  0.518,
                  -0.9999559303, -0.0091250445, -0.0022070244, -0.0318583523,
                  0.0093736937, -1.000,         0.0001,        0.17,
                  0,             0,             0,             1;
    LoadImages();
    LoadPose(image_dir+"/dr_odom2.txt");
    if(!fs::exists(output_dir)){
      fs::create_directories(output_dir);
    }
  }
  int top_k = 256; // 4096
  float detection_threshold = 0.01;
  bool use_cuda = true;
  std::shared_ptr<XFeat::XFDetector> detector;
  std::string image_dir = "../../data";
  float weights = 0.3;
  std::vector<std::pair<std::string,std::string>> sampled;
  std::vector<Eigen::VectorXd> map_poses;
  std::string output_dir = "output";
  std::vector<Relocation_msg> features;
  float fx =457.007355,fy = 457.007355,cx = 322.483673, cy = 235.978027,baseline = 60.097641;
  cv::Mat K = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                      0, fy, cy,
                                      0,  0,  1);
  Eigen::Matrix4d T_wheel_cam;
  int epipolar_tol = 2;
public:
  std::string get_keyboard_input() {
    std::string input;
    std::cin >> input;
    return input;
  }

  double parseTimestamp(const std::string &filename) {
    std::string name = fs::path(filename).stem().string(); // 去掉扩展名
    return std::stod(name);                                // 转成 double
  }
  std::string GetLeftMatch(const std::string &filename) {
    std::string name = fs::path(filename).stem().string(); // 去掉扩展名
    std::string left_match = image_dir+"/stereo_left"+"/" + name + ".jpg";
    if(fs::exists(left_match)){
      std::cout<<"left match found for "<<left_match<<std::endl;
      return left_match;
    }else {
      std::cout<<"No left match found for "<<left_match<<std::endl;
      return "";
    }
    return name;
  }
  // 从 roll, pitch, yaw (ZYX顺序) 转换为旋转矩阵
  Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw) {
      Eigen::AngleAxisd Rx(roll,  Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
      return (Rz * Ry * Rx).toRotationMatrix();
  }

  Eigen::Matrix4d GetMatchPose(double pose_time) {
      if (map_poses.empty()) {
          std::cerr << "map_poses is empty!" << std::endl;
          return Eigen::Matrix4d::Identity();
      }

      // 找到时间最接近的 pose
      double min_diff = std::numeric_limits<double>::max();
      Eigen::VectorXd best_pose(7);
      for (const auto &pose : map_poses) {
          double diff = std::abs(pose_time - pose(0));
          if (diff < min_diff) {
              min_diff = diff;
              best_pose = pose;
          }
      }

      // 转换为 4x4 Tcw
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      Eigen::Matrix3d R = rpyToMatrix(best_pose(4), best_pose(5), best_pose(6));
      T.block<3,3>(0,0) = R;
      T.block<3,1>(0,3) = Eigen::Vector3d(best_pose(1), best_pose(2), best_pose(3));

      return T;
  }
  bool LoadPose(const std::string &filename) {
      std::ifstream fin(filename);
      if (!fin.is_open()) {
          std::cerr << "Failed to open pose file: " << filename << std::endl;
          return false;
      }

      std::string line;
      while (std::getline(fin, line)) {
          if (line.empty() || line[0] == '#') continue; // 跳过注释和空行
          std::stringstream ss(line);
          double time, x, y, z, roll, pitch, yaw;
          ss >> time >> x >> y >> z >> roll >> pitch >> yaw;
          if (ss.fail()) continue; // 跳过非法行

          Eigen::VectorXd pose(7);
          pose << time, x, y, z, 0.0, 0.0, yaw;
          map_poses.push_back(pose);
      }

      fin.close();
      std::cout << "Loaded " << map_poses.size() << " poses from " << filename << std::endl;
      return true;
  }
  void LoadImages() { 
      std::vector<std::string> image_files;
      for (auto &p : fs::directory_iterator(image_dir+"/stereo_right")) {
        if (p.path().extension() == ".jpg") {
          image_files.push_back(p.path().string());
        }
      }
      if (image_files.empty()) {
        std::cerr << "No images found in " << image_dir << std::endl;
        return;
      }
      std::sort(image_files.begin(), image_files.end(),
                [this](const std::string &a, const std::string &b) {
                  return parseTimestamp(a) < parseTimestamp(b);
                });
      // 3. 1s 采样
      double last_time = -1;
      for (const auto &right_img : image_files) {
        double t = parseTimestamp(right_img);
        if (last_time < 0 || t - last_time >= 2.0) {
          std::string left_img = GetLeftMatch(right_img);
          if (left_img!="") 
          {
            sampled.push_back(std::make_pair(left_img, right_img));
            last_time = t;
          }
        }
      }
      std::cout << "Sampled " << sampled.size()
                << " images after 2.0s downsampling" << std::endl;
  }
  void saveMap(const std::vector<Relocation_msg>& msgs, const std::string& folder) {
        // 如果目录存在则删除整个目录（包含 map.yaml, images, depth, features 等）
    if (fs::exists(folder)) {
        fs::remove_all(folder);
    }
    // 创建必要目录
    fs::create_directories(folder + "/images");
    fs::create_directories(folder + "/depth");
    fs::create_directories(folder + "/features");

    YAML::Emitter out;
    out << YAML::BeginSeq;
    for (auto& msg : msgs) {
        YAML::Node node = saveRelocationMsg(msg, folder);
        out << node;
    }
    out << YAML::EndSeq;

    std::ofstream fout(folder + "/map.yaml");
    fout << out.c_str();
  }

  bool  loadMap(std::vector<Relocation_msg> &msgs,const std::string& folder) {
    if (!fs::exists(folder)) {
        return false;
    }else{
      YAML::Node all = YAML::LoadFile(folder + "/map.yaml");
      for (const auto& node : all) {
          msgs.push_back(loadRelocationMsg(node, folder));
      }
      return true;
    }
  }
  Relocation_msg loadRelocationMsg(const YAML::Node& node, const std::string& folder) {
    Relocation_msg msg;
    msg.index = node["id"].as<int>();
    msg.timestamp = node["timestamp"].as<double>();

    // 位姿
    std::vector<double> pose_vec = node["pose"].as<std::vector<double>>();
    Eigen::Matrix4d pose;
    for (int i = 0; i < 16; i++) pose(i/4, i%4) = pose_vec[i];
    msg.pose = pose.transpose();//eigen读取是列优先，需要旋转

    // 图像
    msg.image = cv::imread(folder + "/" + node["image_file"].as<std::string>(), cv::IMREAD_GRAYSCALE);

    // 深度点
    {
        std::ifstream fin(folder + "/" + node["depth_file"].as<std::string>(), std::ios::binary);
        if (fin) {
            int n;
            fin.read((char*)&n, sizeof(int));
            msg.depth_points.resize(n);
            fin.read((char*)msg.depth_points.data(), n * sizeof(cv::Point3f));
        }
    }

    // 特征
    auto features_node = node["features"];
    for (auto it = features_node.begin(); it != features_node.end(); ++it) {
        std::string key = it->first.as<std::string>();
        std::string fpath = folder + "/" + it->second.as<std::string>();

        std::ifstream fin(fpath, std::ios::binary);
        if (!fin) continue;

        int dim;
        fin.read((char*)&dim, sizeof(int));
        std::vector<int64_t> sizes(dim);
        for (int d = 0; d < dim; d++) {
            int s;
            fin.read((char*)&s, sizeof(int));
            sizes[d] = s;
        }
        int type_int;
        fin.read((char*)&type_int, sizeof(int));
        auto dtype = static_cast<c10::ScalarType>(type_int);

        int64_t numel = 1;
        for (auto s : sizes) numel *= s;

        std::vector<char> buffer(numel * c10::elementSize(dtype));
        fin.read(buffer.data(), buffer.size());

        at::Tensor tensor = torch::from_blob(buffer.data(), sizes, dtype).clone();
        msg.feature[key] = tensor;
    }

    return msg;
  }
  YAML::Node saveRelocationMsg(const Relocation_msg& msg, const std::string& folder) {
    YAML::Node node;

    node["id"] = msg.index;
    node["timestamp"] = msg.timestamp;

    // 位姿
    std::vector<double> pose_vec(msg.pose.data(), msg.pose.data()+16);
    node["pose"] = pose_vec;

    // 图像
    std::string img_file = "images/kf" + std::to_string(msg.index) + ".png";
    cv::imwrite(folder + "/" + img_file, msg.image);
    node["image_file"] = img_file;

    // 深度点
    std::string depth_file = "depth/kf" + std::to_string(msg.index) + "_depth.bin";
    {
        std::ofstream fout(folder + "/" + depth_file, std::ios::binary);
        int n = msg.depth_points.size();
        fout.write((char*)&n, sizeof(int));
        fout.write((char*)msg.depth_points.data(), n * sizeof(cv::Point3f));
    }
    node["depth_file"] = depth_file;

    // 特征
    YAML::Node feats_node;
    for (auto& kv : msg.feature) {
        std::string fname = "features/kf" + std::to_string(msg.index) + "_" + kv.first + ".bin";
        std::string full = folder + "/" + fname;

        std::ofstream fout(full, std::ios::binary);
        auto t = kv.second.contiguous();

        int dim = t.dim();
        fout.write((char*)&dim, sizeof(int));
        for (int d=0; d<dim; d++) {
            int size = t.size(d);
            fout.write((char*)&size, sizeof(int));
        }
        int type = (int)t.scalar_type();
        fout.write((char*)&type, sizeof(int));
        fout.write((char*)t.data_ptr(), t.nbytes());
        fout.close();

        feats_node[kv.first] = fname;
    }
    node["features"] = feats_node;

    return node;
  }
  //1024个点大约0.8ms,基线对齐，直接用y方向偏差，不需要ransac，耗时太大,左右目匹配大概丢失一半特征点
  void stereoDepthBatch(
      std::unordered_map<std::string, at::Tensor> left_feature,
      Relocation_msg &rel_msg,
      float epipolar_tol = 3.0f) // y方向像素容差
  {
      auto t2 = std::chrono::high_resolution_clock::now();
      // 保证输入是 float32
      torch::Tensor kptsL = left_feature["keypoints"].to(torch::kFloat32);    // [N1,2]
      torch::Tensor descL = left_feature["descriptors"].to(torch::kFloat32);  // [N1,D]
      torch::Tensor kptsR = rel_msg.feature["keypoints"].to(torch::kFloat32); // [N2,2]
      torch::Tensor descR = rel_msg.feature["descriptors"].to(torch::kFloat32); // [N2,D]
      // 匹配，返回 index
      torch::Tensor idx0, idx1;
      detector->match(descL, descR, idx0, idx1, -1.0f);
      if (idx0.numel() == 0) {
          std::cout << "[stereoDepthBatch] No matches found!" << std::endl;
          return;
      }
      // 索引到匹配点
      torch::Tensor ptsL = kptsL.index_select(0, idx0); // [M,2]
      torch::Tensor ptsR = kptsR.index_select(0, idx1); // [M,2]
      // disparity = xL - xR
      torch::Tensor disparity = ptsL.index({torch::indexing::Slice(), 0}) -
                                ptsR.index({torch::indexing::Slice(), 0});
      // 过滤 disparity <= 0
      torch::Tensor valid_mask = disparity > 0;
      ptsL = ptsL.index({valid_mask});
      ptsR = ptsR.index({valid_mask});
      disparity = disparity.index({valid_mask});
      idx1 = idx1.index({valid_mask});
      if (disparity.numel() == 0) {
          std::cout << "[stereoDepthBatch] No valid disparity points!" << std::endl;
          return;
      }
      // -------- 根据y方向偏差过滤外点 --------
      torch::Tensor y_diff = (ptsL.index({torch::indexing::Slice(), 1}) -
                              ptsR.index({torch::indexing::Slice(), 1})).abs();
      torch::Tensor inlier_mask = y_diff < epipolar_tol;
      ptsL = ptsL.index({inlier_mask});
      ptsR = ptsR.index({inlier_mask});
      disparity = disparity.index({inlier_mask});
      idx1 = idx1.index({inlier_mask});
      if (disparity.numel() == 0) {
          std::cout << "[stereoDepthBatch] No inliers after y-filter!" << std::endl;
          return;
      }
      // 计算深度 Z
      torch::Tensor Z = fx * baseline / disparity;
      // 计算 3D 坐标
      torch::Tensor xL = ptsL.index({torch::indexing::Slice(), 0});
      torch::Tensor yL = ptsL.index({torch::indexing::Slice(), 1});
      torch::Tensor X = (xL - cx) * Z / fx;
      torch::Tensor Y = (yL - cy) * Z / fy;
      torch::Tensor points3D = torch::stack({X, Y, Z}, 1); // [M,3]
      // -------- 保存到 rel_msg --------
      rel_msg.depth_points.assign(kptsR.size(0), cv::Point3f(0,0,0));
      auto points3D_acc = points3D.accessor<float,2>();
      auto idx1_acc = idx1.accessor<long,1>();
      for (int i = 0; i < points3D.size(0); i++) {
          int iR = idx1_acc[i];
          if (std::fabs(points3D_acc[i][2]) > 1000.0 &&
              std::fabs(points3D_acc[i][2]) < 5000.0) {
            rel_msg.depth_points[iR] = cv::Point3f(
                points3D_acc[i][0]/1000.0f, points3D_acc[i][1]/1000.0f, points3D_acc[i][2]/1000.0f);//转换为m
          }
      }

      auto t3 = std::chrono::high_resolution_clock::now();
      double warp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
      std::cout << "[Time] depth estimator (batch + y-filter): " << warp_ms << " ms" << std::endl;
      std::cout << "[stereoDepthBatch] valid 3D points after y-filter: " << points3D.size(0) << std::endl;
  }

 int publish_array(int mode) {
    // 填充数据
    if (mode == 1) {
      // load the image and convert to tensor
      std::cout<<"Please input the image 1 path: ";
      cv::Mat image1 = cv::imread(get_keyboard_input());
      std::cout<<"Please input the image 2 path: ";
      cv::Mat image2 = cv::imread(get_keyboard_input());
      if (image1.empty() || image2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
      }
      cv::Mat mkpts_0, mkpts_1;
      auto t2 = std::chrono::high_resolution_clock::now();
      detector->match_xfeat(image1, image2, mkpts_0, mkpts_1);
      auto t3 = std::chrono::high_resolution_clock::now();
      double warp_ms =
          std::chrono::duration<double, std::milli>(t3 - t2).count();
      std::cout << "[Time] match all time: " << warp_ms << " ms" << std::endl;
      std::cout << "mkpts_0 size: " << mkpts_0.size()
                << ", type: " << mkpts_0.type() << std::endl;
      std::cout << "mkpts_1 size: " << mkpts_1.size()
                << ", type: " << mkpts_1.type() << std::endl;
      //输出：
      // mkpts_0 size: [2 x 972], type: 5
      // mkpts_1 size: [2 x 972], type: 5
      //期望输出：
      // mkpts_0 size: [972 x 1], type: 13  // CV_32FC2
      // mkpts_1 size: [972 x 1], type: 13  // CV_32FC2
      // 如果 type 不是 13（CV_32FC2），则需要转换：
      if (mkpts_0.type() != CV_32FC2)
        mkpts_0.convertTo(mkpts_0, CV_32FC2);
      if (mkpts_1.type() != CV_32FC2)
        mkpts_1.convertTo(mkpts_1, CV_32FC2);
      warp_corners_and_draw_matches(mkpts_0, mkpts_1, image1, image2);
    } else if (mode == 2) { //全部图像匹配
      // 5. 两两匹配 (全连接)
      for (size_t i = 0; i < sampled.size(); i++) {
        cv::Mat img1 = cv::imread(sampled[i].second);
        if (img1.empty())
          continue;
        for (size_t j = i + 1; j < sampled.size(); j++) {
          cv::Mat img2 = cv::imread(sampled[j].second);
          if (img2.empty())
            continue;
          cv::Mat mkpts_0, mkpts_1;
          detector->match_xfeat(img1, img2, mkpts_0, mkpts_1);
          if (mkpts_0.rows == 0 || mkpts_1.rows == 0)
            continue;
          // 计算匹配比例
          float ratio = static_cast<float>(mkpts_0.rows) / top_k;
          // 需要你在 XFDetector 里加个 getLastTotalMatches() 或者直接传总数
          if (ratio > weights) {
            // 画匹配并保存
            cv::Mat out_img;
            std::vector<cv::KeyPoint> kpts1, kpts2;
            for (int k = 0; k < mkpts_0.rows; k++) {
              kpts1.emplace_back(mkpts_0.at<cv::Point2f>(k, 0), 5);
              kpts2.emplace_back(mkpts_1.at<cv::Point2f>(k, 0), 5);
            }
            std::vector<cv::DMatch> matches;
            for (int k = 0; k < mkpts_0.rows; k++)
              matches.emplace_back(k, k, 0);

            cv::drawMatches(img1, kpts1, img2, kpts2, matches, out_img);
            // 在图上标注文件名
            std::string label = fs::path(sampled[i].second).filename().string() +
                                " vs " +
                                fs::path(sampled[j].second).filename().string();
            cv::putText(out_img, label, cv::Point(30, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0),
                        2);

            std::string out_path =
                output_dir + "/" + fs::path(sampled[i].second).stem().string() +
                "_vs_" + fs::path(sampled[j].second).stem().string() + ".jpg";
            cv::imwrite(out_path, out_img);

            std::cout << "Saved match: " << out_path << std::endl;
          }
        }
      }
    } else if (mode == 3) { //图片和文件夹匹配
      std::cout<<"Please input the image 1 path: ";
      cv::Mat image1 = cv::imread(get_keyboard_input());
      for (size_t i = 0; i < sampled.size(); i++) {
        cv::Mat img1 = cv::imread(sampled[i].second);
        if (img1.empty())
          continue;
        cv::Mat mkpts_0, mkpts_1;
        detector->match_xfeat(img1, image1, mkpts_0, mkpts_1);
        if (mkpts_0.rows == 0 || mkpts_1.rows == 0)
          continue;
        float ratio = static_cast<float>(mkpts_0.rows) / top_k;
        if (ratio > weights) {
          cv::Mat out_img;
          std::vector<cv::KeyPoint> kpts1, kpts2;
          for (int k = 0; k < mkpts_0.rows; k++) {
            kpts1.emplace_back(mkpts_0.at<cv::Point2f>(k, 0), 5);
            kpts2.emplace_back(mkpts_1.at<cv::Point2f>(k, 0), 5);
          }
          std::vector<cv::DMatch> matches;
          for (int k = 0; k < mkpts_0.rows; k++)
            matches.emplace_back(k, k, 0);
          cv::drawMatches(img1, kpts1, image1, kpts2, matches, out_img);
          std::string label = fs::path(sampled[i].second).filename().string();
          cv::putText(out_img, label, cv::Point(30, 30),
                      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
          std::string out_path =
              output_dir + "/" + fs::path(sampled[i].second).stem().string() + ".jpg";
          cv::imwrite(out_path, out_img);
          std::cout << "Saved match: " << out_path << std::endl;
        }
      }
    }else if(mode == 4){
      std::cout<<"Please input the weights: ";
      weights = std::stof(get_keyboard_input());
    }else if(mode == 5){
      int index=0;
      Relocation_msg rel_msg;
      cv::Mat left_image;
      std::unordered_map<std::string, at::Tensor> left_feature;
      for (size_t i = 0; i < sampled.size(); i++) {
        cv::Mat gray = cv::imread(sampled[i].second);
        cv::cvtColor(gray,rel_msg.image, cv::COLOR_BGR2GRAY);
        gray = cv::imread(sampled[i].first);
        cv::cvtColor(gray,left_image,cv::COLOR_BGR2GRAY);
        rel_msg.timestamp = parseTimestamp(sampled[i].second);
        rel_msg.pose = GetMatchPose(rel_msg.timestamp);
        std::cout <<std::fixed<<std::setprecision(3)<< "image name: " << rel_msg.timestamp << std::endl;
        if (rel_msg.image.empty()||left_image.empty())
          continue;
        detector->xfeat_keypoints_descritors(rel_msg.image, rel_msg.feature);
        detector->xfeat_keypoints_descritors(left_image, left_feature);
        stereoDepthBatch(left_feature, rel_msg);
        rel_msg.index = index;
        // 在 push_back 前深拷贝一次 image
        Relocation_msg tmp = rel_msg;
        tmp.image = rel_msg.image.clone();
        // feature 也要 clone，不然 at::Tensor 共享内存
        tmp.feature.clear();
        for (const auto& kv : rel_msg.feature) {
            tmp.feature[kv.first] = kv.second.clone();
        }
        features.push_back(std::move(rel_msg));
        //features.push_back(rel_msg);
        index++;
      }
      for(auto &rel_msg_show:features)
      {
        cv::imshow("image", rel_msg_show.image);
        cv::waitKey(10);
      }
      std::cout<<"Save mapping..."<<std::endl;
      saveMap(features,"../map");
    }else if(mode == 6){
      if(!loadMap(features,"../map"))
      {  
        std::cout<<"Load mapping Failure"<<std::endl;
        return -1;
      }
      Relocation_msg rel_msg;
      std::cout << "Feature count: " << features.size() 
          << ", approximate memory: " << features.size() * sizeof(Relocation_msg) << " bytes" << std::endl;
      std::cout << "Please input the match image path: ";
      std::string right_image_path = get_keyboard_input();
      rel_msg.timestamp = parseTimestamp(right_image_path);
      rel_msg.pose = GetMatchPose(rel_msg.timestamp);
      std::cout << "Input Image pose:"<<rel_msg.pose<<std::endl;
      std::cout << std::fixed<< std::setprecision(3)<<"input images timestamp:"<<rel_msg.timestamp<<std::endl;
      cv::Mat gray = cv::imread(right_image_path);
      cv::cvtColor(gray,rel_msg.image, cv::COLOR_BGR2GRAY);
      if (rel_msg.image.empty())
        return -1;
      detector->xfeat_keypoints_descritors(rel_msg.image, rel_msg.feature);
      auto t2 = std::chrono::high_resolution_clock::now();
      //先按距离排序，优先距离近的点
      // 按距离排序 features
      std::vector<Relocation_msg> sorted_features = features;  
      std::sort(sorted_features.begin(), sorted_features.end(),
          [&](const Relocation_msg& a, const Relocation_msg& b) {
              double da = (a.pose.block<3,1>(0,3) - rel_msg.pose.block<3,1>(0,3)).norm();
              double db = (b.pose.block<3,1>(0,3) - rel_msg.pose.block<3,1>(0,3)).norm();
              return da < db;  // 距离小的排前面
          });
      // ---------------- 分距离扩展匹配 ----------------
      double search_radius = 3.0;      // 初始搜索半径 3m
      double max_radius   = 15.0;      // 最大搜索半径，可以根据地图大小调节
      bool   found_match  = false;

      auto start = std::chrono::steady_clock::now();

      while (!found_match && search_radius <= max_radius) {
          std::cout << "[Search] radius = ±" << search_radius << "m" << std::endl;
          for (auto &item : sorted_features) {
              // 计算两帧平移距离
              double dist = (item.pose.block<3,1>(0,3) - rel_msg.pose.block<3,1>(0,3)).norm();
              if (dist > search_radius || dist < search_radius -3) continue;  // 不在当前半径范围内，跳过
              cv::Mat mkpts_0, mkpts_1;
              torch::Tensor idxs0, idxs1;
              detector->match(rel_msg.feature["descriptors"], 
                              item.feature["descriptors"], 
                              idxs0, idxs1, -1.0);

              torch::Tensor mkpts_0_tensor = rel_msg.feature["keypoints"].index({idxs0});
              torch::Tensor mkpts_1_tensor = item.feature["keypoints"].index({idxs1});
              mkpts_0 = detector->tensorToMat(mkpts_0_tensor);
              mkpts_1 = detector->tensorToMat(mkpts_1_tensor);

              if (mkpts_0.rows == 0 || mkpts_1.rows == 0)
                  continue;

              float ratio = static_cast<float>(mkpts_0.rows) / top_k;
              if (ratio > weights) {
                  // ---- RANSAC + PnP 解算（保持你原来的代码） ----
                std::vector<cv::Point2f> pts1_cv, pts2_cv;
                for (int k = 0; k < mkpts_0.rows; k++) {
                    pts1_cv.push_back(mkpts_0.at<cv::Point2f>(k, 0));
                    pts2_cv.push_back(mkpts_1.at<cv::Point2f>(k, 0));
                }

                cv::Mat inlier_mask;
                auto ransac_t1 = std::chrono::high_resolution_clock::now();
                cv::findFundamentalMat(pts1_cv, pts2_cv, cv::FM_RANSAC, 5.0, 0.99, inlier_mask);
                auto ransac_t2 = std::chrono::high_resolution_clock::now();
                double ransac_warp_ms =
                    std::chrono::duration<double, std::milli>(ransac_t2 - ransac_t1).count();
                std::cout << "[Time] ransac time: " << ransac_warp_ms << " ms"
                          << std::endl;
                // 只保留内点
                std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
                std::vector<cv::DMatch> inlier_matches;
                for (int k = 0; k < mkpts_0.rows; k++) {
                    if (inlier_mask.at<uchar>(k)) {
                        inlier_pts1.push_back(pts1_cv[k]);
                        inlier_pts2.push_back(pts2_cv[k]);
                        inlier_matches.emplace_back(inlier_pts1.size() - 1, inlier_pts2.size() - 1, 0);
                    }
                }
                if (inlier_pts1.empty())
                    continue;
                // ---------- 绘制匹配图 ----------
                cv::Mat in_img = rel_msg.image.clone();
                cv::Mat out_img;
                std::vector<cv::KeyPoint> kpts1_cv, kpts2_cv;
                for (auto &pt : inlier_pts1) kpts1_cv.emplace_back(pt, 5);
                for (auto &pt : inlier_pts2) kpts2_cv.emplace_back(pt, 5);

                cv::drawMatches(in_img, kpts1_cv, item.image, kpts2_cv, inlier_matches, out_img);

                std::string label = std::to_string(item.timestamp);
                cv::putText(out_img, label, cv::Point(30, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                std::string out_path = output_dir + "/" + label + ".jpg";
                cv::imwrite(out_path, out_img);
                std::cout << "Saved match: " << out_path << std::endl;
                // ---------- 构建 3D-2D 对应 ----------
                std::vector<cv::Point3f> objectPoints;
                std::vector<cv::Point2f> imagePoints;

                for (int k = 0; k < mkpts_0.rows; k++) {
                    if (inlier_mask.at<uchar>(k)) {
                        int old_idx = idxs1[k].item<int>();
                        if (old_idx < 0 || old_idx >= item.depth_points.size())
                            continue;
                        cv::Point3f p3d = item.depth_points[old_idx];
                        if (p3d.z <= 0) continue;  // 跳过无效深度
                        objectPoints.push_back(p3d);
                        imagePoints.push_back(mkpts_0.at<cv::Point2f>(k, 0));

                        // 在新图上画深度
                        cv::circle(in_img, mkpts_0.at<cv::Point2f>(k, 0), 3, cv::Scalar(0, 0, 255), -1);
                        std::string text = cv::format("%.2f", p3d.z);
                        cv::putText(in_img, text, mkpts_0.at<cv::Point2f>(k, 0) + cv::Point2f(5, -5),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                    }
                }
                // ---------- PnP 求相对位姿 ----------
                if (objectPoints.size() >= 20) {
                    std::string depth_out_path = output_dir + "/depth_" + label + ".jpg";
                    cv::imwrite(depth_out_path, in_img);
                    std::cout << "Saved depth match: " << depth_out_path << std::endl;
                    cv::Mat rvec, tvec, inliers;
                    auto pnp_t1 = std::chrono::high_resolution_clock::now();
                    cv::solvePnPRansac(
                        objectPoints, imagePoints, K, cv::noArray(),
                        rvec, tvec, false, 300, 5.0, 0.99, inliers
                    );
                    auto pnp_t2 = std::chrono::high_resolution_clock::now();
                    auto pnp_time = std::chrono::duration_cast<std::chrono::milliseconds>(pnp_t2 - pnp_t1).count();
                    std::cout << "pnp time: " << pnp_time << "ms" << std::endl;
                    cv::Mat R;
                    cv::Rodrigues(rvec, R);

                    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
                    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
                    T.at<double>(0, 3) = tvec.at<double>(0);
                    T.at<double>(1, 3) = tvec.at<double>(1);
                    T.at<double>(2, 3) = tvec.at<double>(2);
                    Eigen::Matrix4d T_cam_rel = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T.ptr<double>());
                    std::cout << std::fixed<< std::setprecision(3)<< "Estimated relative:"<<objectPoints.size()<<",pose from map frame "
                              << item.timestamp << " to query frame " << rel_msg.timestamp
                              << ":\n" << T << std::endl;
                    // 1. 真值 (wheel frame) 相对位姿
                    Eigen::Matrix4d T_wheel_rel_gt = rel_msg.pose.inverse() * item.pose;
                    std::cout<<"Match pose:"<<item.pose<<std::endl;
                    std::cout << "Ture pose Change:" << std::endl << T_wheel_rel_gt << std::endl;
                    // 2. 视觉 (camera frame) 相对位姿换算到 wheel frame
                    Eigen::Matrix4d T_wheel_rel_vis = T_wheel_cam * T_cam_rel * T_wheel_cam.inverse();
                    // 3. 误差矩阵
                    Eigen::Matrix4d T_err = T_wheel_rel_gt.inverse() * T_wheel_rel_vis;
                    std::cout << "Error:" << std::endl << T_err << std::endl;
                    Eigen::Matrix4d pic_pos = item.pose * T_wheel_rel_vis.inverse();
                    std::cout << "Pic pos:" << std::endl << pic_pos << std::endl;
                    found_match = true;
                }
              }
              // =============================================           
              if (found_match) break;  // 找到就直接跳出
          }
          // 超时保护，防止卡死
          auto now = std::chrono::steady_clock::now();
          if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > 500) {
              std::cout << "[Warning] Relocalization timeout!" << std::endl;
              break;
          }
          search_radius += 3.0;  // 每次扩展 3m
      }

      if (!found_match) {
          std::cout << "[Result] No valid match found in range ±" << max_radius << "m" << std::endl;
      } else {
          std::cout << "[Result] Found valid match!" << std::endl;
      }
      auto t3 = std::chrono::high_resolution_clock::now();
      double warp_ms =
          std::chrono::duration<double, std::milli>(t3 - t2).count();
      std::cout << "[Time] xfeat relocal use time: " << warp_ms << " ms" << std::endl;
    }
    return 0;
  }
  // Display sub-menu for external commands
  void displaySubMenuA() {
    bool exit_submenu = false;
    std::string input;
    while (!exit_submenu) {
      std::cout << "\n=========== Sub Menu A ========\n";
      std::cout << "[ 1] 两张图片匹配                  \n";
      std::cout << "[ 2] 文件夹内相互匹配            \n";
      std::cout << "[ 3] 图片和文件夹匹配             \n";
      std::cout << "[ 4] 设置权重                    \n";
      std::cout << "[ 5] 建图并保存                  \n";
      std::cout << "[ 6] 重定位                  \n";
      std::cout << " Enter your choice: ";
      std::cin >> input;

      if (input == "q") {
        exit_submenu = true;
        std::cout << "Exit Sub Menu A " << std::endl;
        continue;
      }
      int sub_key = std::stoi(input);
      publish_array(sub_key);
    }
  }
  void processKeyBoard() {
    std::cout << "Processing..." << std::endl;

    char key;
    bool exit_program = false;

    // Main loop to handle user input and trigger ROS2 actions
    while (!exit_program) {
      std::cout << "\n============ Main Menu =============\n";
      std::cout << "Option [a]: External Slam Commands! \n";
      std::cout << "Option [q]: Exit Menu \n";
      std::cout << "Please choose an option: ";
      std::cin >> key;
      if (key == 'a') {
        displaySubMenuA();
      } else {
        exit_program = true;
      }
    }
  }
  void warp_corners_and_draw_matches(cv::Mat &ref_points, cv::Mat &dst_points, cv::Mat &img1, cv::Mat &img2)
  {      
      // Check if there are enough points to find a homography
      if (ref_points.rows < 4 || dst_points.rows < 4) {
          std::cerr << "Not enough points to compute homography" << std::endl;
          return;
      }
      // 检查数据格式
      if (ref_points.empty() || dst_points.empty()) {
          std::cerr << "Empty keypoints!" << std::endl;
          return;
      }
      std::cout << "ref_points size: " << ref_points.size() << ", type: " << ref_points.type() << std::endl;
      std::cout << "dst_points size: " << dst_points.size() << ", type: " << dst_points.type() << std::endl;

      // 确保是 CV_32FC2 格式
      if (ref_points.type() != CV_32FC2) ref_points.convertTo(ref_points, CV_32FC2);
      if (dst_points.type() != CV_32FC2) dst_points.convertTo(dst_points, CV_32FC2);
      
      //OpenCV 4.2.0 中，cv::findHomography() 的 RANSAC 参数应使用 cv::RANSAC 或 cv::LMEDS（Least-Median Robust Method）
  // OpenCV 4.4.0+ 增加的 cv::USAC_DEFAULT 和 cv::USAC_MAGSAC 选项.
      
      //eg'tensorToMat() 返回的矩阵不是 CV_32FC2（cv::Point2f 格式），导致 cv::findHomography() 无法解析
      cv::Mat mask;
      // 使用 RANSAC 方法计算单应矩阵 //  可能返回空的 H 或 mask，导致后续 cv::drawMatches() 失败
      cv::Mat H = cv::findHomography(
          ref_points, 
          dst_points, 
          cv::RANSAC,    // 方法：RANSAC（OpenCV 4.2.0 可用）
          3.0,           // RANSAC 阈值（推荐 1.0~5.0）
          mask, //was' cv::noArray(),  // 可选的输出掩码（inliers）
          1000,           // 最大迭代次数（默认 2000）
          0.994            // 置信度（默认 0.995）
      );

      
      //was'cv4n4+' cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0, mask, 1000, 0.994);
      if (H.empty()) {
          std::cerr << "Homography matrix is empty" << std::endl;
          return;
      }
      //如果 mask 全为 0，说明 cv::findHomography() 认为所有匹配都是外点（outliers）
      // 统计内点数量
      int inliers = cv::countNonZero(mask);
      std::cout << "Homography inliers: " << inliers << "/" << mask.total() << std::endl;
      //输出: Homography inliers: 47/972' (./match zju1day.jpg zju2dark.jpg )
      //vs' Homography inliers: 60/972 ( ./match zju2dark.jpg zju1day.jpg)

      //降低 cv::findHomography() 的 ransacReprojThreshold（如从 10.0 改为 3.0）

      if (inliers < 4) {
          std::cerr << "Not enough inliers (" << inliers << ") to draw matches!" << std::endl;
          return;
      }
      mask = mask.reshape(1);

      float h = img1.rows;
      float w = img1.cols;
      std::vector<cv::Point2f> corners_img1 = {cv::Point2f(    0,     0), 
                                                  cv::Point2f(w - 1,     0), 
                                                  cv::Point2f(w - 1, h - 1), 
                                                  cv::Point2f(    0, h - 1)};
      std::vector<cv::Point2f> warped_corners;
      cv::perspectiveTransform(corners_img1, warped_corners, H);

      cv::Mat img2_with_corners = img2.clone();
      for (size_t i = 0; i < warped_corners.size(); ++i) {
          cv::line(img2_with_corners, warped_corners[i], warped_corners[(i+1) % warped_corners.size()], cv::Scalar(0, 255, 0), 4);
      }

      // prepare keypoints and matches for drawMatches function
      std::vector<cv::KeyPoint> keypoints1, keypoints2;
      std::vector<cv::DMatch> matches;
      for (int i = 0; i < mask.rows; ++i) {
          keypoints1.emplace_back(ref_points.at<cv::Point2f>(i, 0), 5);
          keypoints2.emplace_back(dst_points.at<cv::Point2f>(i, 0), 5);
          if (mask.at<uchar>(i, 0))
              matches.emplace_back(i, i, 0);
      }
      std::cout << std::endl << " pntsA.n:" << keypoints1.size();
      std::cout << std::endl << " pntsB.n:" << keypoints2.size();
      std::cout << std::endl << " matches.n:" << matches.size();
      
      // Draw inlier matches
      cv::Mat img_matches;
      if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
          cv::drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
          cv::Mat resized_frame;
          cv::resize(img_matches, resized_frame, cv::Size(1366, 1440));
          cv::imshow("Matches", img_matches);
          cv::waitKey(0); // Wait for a key press
                      
          // // Uncomment to save the matched image
          // std::string output_path = "doc/image_matches.png";
          // if (cv::imwrite(output_path, img_matches)) {
          //     std::cout << "Saved image matches to " << output_path << std::endl;
          // } else {
          //     std::cerr << "Failed to save image matches to " << output_path << std::endl;
          // }

      } else {
          std::cerr << "Keypoints or matches are empty, cannot draw matches" << std::endl;
      }
  }

};
int main(int argc, char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }
  KeyBoardSimulator simulator(argv[1]);
  simulator.processKeyBoard();
  return 0;
}


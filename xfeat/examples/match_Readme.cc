#include "XFeat.h"

void warp_corners_and_draw_matches(cv::Mat &mkpts_0, cv::Mat &mkpts_1, cv::Mat &img1, cv::Mat &img2);
//说明:ref.jpg, query.jpg只是临时的名称. (./match zju2dark.jpg zju1day.jpg: 顺序1)
//viz:绿色多边形表示 ref.jpg 的四个角点经过单应矩阵 H 变换后 在 query.jpg 上的投影（顺序1）。如果多边形与目标物体对齐良好，说明 H 正确
    // (顺序1) 
    // H 性质：满秩，3 个显著奇异值。
    // Inliers：均匀分布在整个图像区域。
    // 绿色多边形：完整四边形，反映 ref.jpg (zju2dark.jpg)到 goal.jpg (zju1day.jpg) 的正确投影。
    
//vs'绿色直线表示 goal.jpg 的角点变换到 ref.jpg 时，仅有一个边对齐（可能是由于 query.jpg ref.jpg（顺序2） H 的逆变换不稳定或匹配点分布不均匀）
 // (顺序12      H 性质：秩亏，奇异值中有 1~2 个接近 0。
 //    Inliers：集中在一条边缘（如右下角）。
 //    绿色直线：H 仅能约束一个边的对齐，其他方向自由度过大。
//Q:"一个边对齐"对应于cv::findHomography()得到的inliers数值或单应矩阵 H 数值上的什么情况?
//re.当绿色多边形退化为 一条直线 时，说明 cv::findHomography() 计算出的单应矩阵 H 存在 秩亏（Rank Deficiency） 或 共线性问题
 // 绿色直线 表明 H 的秩亏或 inliers 共线，导致投影退化。
//单应矩阵 H 的奇异值分解（SVD）：正常的 H 应有 3 个非零奇异值（满秩）。如果仅有一条边对齐，H 的奇异值中会有 1~2 个接近 0，导致变换后的角点共线。
 //cv::SVD svd(H);
 // cout << "Singular values of H: " << svd.w ; // 例如输出 [1.3, 0.8, 1e-6]
//如果 H 的第三行（[h20, h21, h22]）与其他行线性相关，会导致投影后的点满足 h20*x + h21*y + h22 ≈ 0，即所有点落在一条直线上。
 //2ex'过滤共线匹配点' 如果 inliers 共线，拒绝使用该 H：
  //

//inliers 数量： cv::findHomography() 返回的 mask 中，inliers 数量可能较少（如 < 10），且这些点 近似共线。
 // eg' 输出 5（远小于匹配点总数） // int inliers = cv::countNonZero(mask);
 // Inliers 几何分布：如果匹配点集中在图像的一个边缘（如右下角），H 会倾向于将该边缘对齐，而其他区域无法约束，导致绿色多边形退化为直线。

// 共线性检测函数'示例
//2chk:'cv::Exception' OpenCV(4.2.0) ../modules/core/src/copy.cpp:376: error: (-215:Assertion failed) size() == mask.size() in function 'copyTo'
bool is_collinear(const cv::Mat& points, float threshold = 0.99) 
{
    if (points.rows < 3) return true;
    cv::Mat A(points.rows, 3, CV_32F);
    for (int i = 0; i < points.rows; ++i) {
        A.at<float>(i, 0) = points.at<cv::Point2f>(i).x;
        A.at<float>(i, 1) = points.at<cv::Point2f>(i).y;
        A.at<float>(i, 2) = 1;
    }
    cv::SVD svd(A);
    float ratio = svd.w.at<float>(2) / svd.w.at<float>(0);  // 最小与最大奇异值之比
    if (ratio < threshold) {
		std::cout << std::endl << " co-linear. thres:"<< threshold << " > svd-ratio:" << ratio;
	}
    return (ratio < threshold);  // 接近 0 表示共线
}

int main(int argc, char** argv) {

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <weights> <image1> <image2>\n";
        return -1;
    }

    // instantiate XFDetector
    int top_k = 4096;
    float detection_threshold = 0.01;
    // float detection_threshold = 0.05;
    bool use_cuda = true; 
    XFeat::XFDetector detector(top_k, detection_threshold, use_cuda);

    // load the image and convert to tensor
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }

    // Perform feature matching on the same image
    cv::Mat mkpts_0, mkpts_1;
    detector.match_xfeat(image1, image2, mkpts_0, mkpts_1);
    
    std::cout << "mkpts_0 size: " << mkpts_0.size() << ", type: " << mkpts_0.type() << std::endl;
    std::cout << "mkpts_1 size: " << mkpts_1.size() << ", type: " << mkpts_1.type() << std::endl;
    //输出：
    // mkpts_0 size: [2 x 972], type: 5
    // mkpts_1 size: [2 x 972], type: 5

    //期望输出：
    // mkpts_0 size: [972 x 1], type: 13  // CV_32FC2
    // mkpts_1 size: [972 x 1], type: 13  // CV_32FC2
    // 如果 type 不是 13（CV_32FC2），则需要转换：
    if (mkpts_0.type() != CV_32FC2) mkpts_0.convertTo(mkpts_0, CV_32FC2);
    if (mkpts_1.type() != CV_32FC2) mkpts_1.convertTo(mkpts_1, CV_32FC2);


    warp_corners_and_draw_matches(mkpts_0, mkpts_1, image1, image2);

    return 0;
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
    //2chk.改进：通过统一单应矩阵计算方向，消除输入顺序的影响。since:findHomography():输入点顺序影响计算结果。单应矩阵 H 是从 第一张图的坐标系 映射到 第二张图的坐标系，
    cv::Mat H = cv::findHomography(
        ref_points,  //  目标物体作为第一张图' 将 ref.jpg 的点映射到 query.jpg待匹配图像，正向变换 H 通常更稳定。
        dst_points, 
        cv::RANSAC,    // 方法：RANSAC（OpenCV 4.2.0 可用）
        3.0,           // (像素）' ransacReprojThreshold 阈值（推荐 1.0~5.0）
        mask, //was' cv::noArray(),  // 可选的输出掩码（inliers）
        1000,           // 最大迭代次数（默认 2000）
        0.994            // 置信度（默认 0.995）
    );
    //vs'反向计算 H_inv 时，由于噪声或匹配点分布问题，可能导致投影不稳定（如直线现象）。
    //param:ransacReprojThreshold 越小 → 要求匹配越精确（内点减少，但更鲁棒）,值越大 → 允许更大的匹配误差（内点增多，但可能包含错误匹配）,' ransacReprojThreshold参数'是 RANSAC 算法中用于判断一个匹配点是否为内点（inlier）的重投影误差阈值（单位：像素）（默认值通常为 3.0）

	/*// 尝试不同的阈值（1.0~5.0）
	std::vector<double> thresholds = {5.0, 3.0, 1.5, 1.0};
	for (double thresh : thresholds) {
		cv::Mat mask;
		cv::Mat H = cv::findHomography(ref_points, dst_points, cv::RANSAC, thresh, mask, 1000, 0.99);
		int inliers = cv::countNonZero(mask);
		std::cout << "Threshold: " << thresh << ", Inliers: " << inliers << std::endl;
	}*/
	//viz:（顺序1）绿色多边形表示 ref.jpg 的四个角点经过单应矩阵 H 变换后 在 goal.jpg 上的投影。如果多边形与目标物体对齐良好，说明 H 正确。
	//而./match zju1day.jpg zju2dark.jpg（顺序2）：绿色直线表示 goal.jpg 的角点变换到 ref.jpg 时，仅有一个边对齐（可能是由于 H 的逆变换不稳定或匹配点分布不均匀）

    
    //was'cv4n4+' cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0, mask, 1000, 0.994);
    if (H.empty()) {
        std::cerr << "Homography failed! Homography matrix is empty" << std::endl;
        return;
    }
    
    //如果 mask 全为 0，说明 cv::findHomography() 认为所有匹配都是外点（outliers）
    // 统计内点数量
    int inliers = cv::countNonZero(mask);
    std::cout << "Homography inliers: " << inliers << "/" << mask.total() << std::endl;
    //输出: Homography inliers: 47/972' (./match zju1day.jpg zju2dark.jpg )
    //vs' Homography inliers: 60/972 ( ./match zju2dark.jpg zju1day.jpg)


    // 检查单应矩阵的奇异值
    cv::SVD svd(H);
    // std::cout << "H Singular Values:\n" << svd.w ;

    // 提取内点（修复 copyTo 错误）
    cv::Mat inlier_indices;
    cv::findNonZero(mask, inlier_indices);  // 获取内点索引
    // 检查 inliers 分布
    cv::Mat inlier_points(inlier_indices.rows, 1, CV_32FC2);
    for (int i = 0; i < inlier_indices.rows; ++i) {
        int idx = inlier_indices.at<cv::Point>(i).x;
        inlier_points.at<cv::Point2f>(i) = ref_points.at<cv::Point2f>(idx);
    }

    // 打印内点信息
    std::cout << std::endl << "Inliers: " << inlier_points.rows << std::endl;

    //cv::Mat inlier_points;
    //err: ref_points.copyTo(inlier_points, mask);
    //deepseek'Q:    ref_points 和 mask 的尺寸不匹配。
    // ref_points 是 N×1 的 CV_32FC2 矩阵（存储 cv::Point2f），而 mask 是 N×1 的 CV_8U 矩阵（存储 0/1）。
    //copyTo() 要求 src 和 mask 的 尺寸完全相同，但 ref_points 和 mask 的通道数不同（CV_32FC2 vs CV_8U）
    //导致: OpenCV(4.2.0) ../modules/core/src/copy.cpp:376: error: (-215:Assertion failed) size() == mask.size() in function 'copyTo'
    
    std::cout << " rowRange of Inlier points (first 10):\n" << inlier_points.rowRange(0, 10) << std::endl;
    //re'确保 inlier_points 是 N×1 的 CV_32FC2 矩阵（存储 cv::Point2f）
	if (is_collinear(inlier_points))   // 自定义共线性检测函数
		std::cerr << "Inliers are collinear!" << std::endl;
		//! return;
        
    //! std::cout << "Collinear: " << (is_collinear(inlier_points) ? "Yes" : "No") << std::endl;
    
    //降低 cv::findHomography() 的 ransacReprojThreshold（如从 10.0 改为 3.0）

    if (inliers < 4) {
        std::cerr << "Not enough inliers (" << inliers << ") to draw matches!" << std::endl;
        return;
    }
    mask = mask.reshape(1);

    float h = img1.rows;
    float w = img1.cols;
    // 绿色框'显示 原始图像在目标图像中的投影区域，直观展示单应矩阵 H 的变换效果。

    // 如果绿色框与目标图像对齐良好，说明匹配正确；否则可能存在误匹配。
    std::vector<cv::Point2f> corners_img1 = {cv::Point2f(    0,     0), 
                                                cv::Point2f(w - 1,     0), 
                                                cv::Point2f(w - 1, h - 1), 
                                                cv::Point2f(    0, h - 1)};
    // 计算原始图像的四个角点经单应矩阵变换后的位置
    std::vector<cv::Point2f> warped_corners;
    cv::perspectiveTransform(corners_img1, warped_corners, H);

    // 在目标图像上绘制绿色多边形
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
    
    //eg' 仅绘制匹配点，不绘制绿色多边形
    std::vector<cv::KeyPoint> kpts1, kpts2;
    std::vector<cv::DMatch> matchesAB;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            kpts1.emplace_back(ref_points.at<cv::Point2f>(i), 5);
            kpts2.emplace_back(dst_points.at<cv::Point2f>(i), 5);
            matchesAB.emplace_back(kpts1.size()-1, kpts2.size()-1, 0);
        }
    }
    
    // Draw inlier matches
    cv::Mat img_matches;
    if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
        // eg' 绘制匹配点，also'绘制绿色多边形
        cv::drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::Mat resized_frame;
        cv::resize(img_matches, resized_frame, cv::Size(1366, 1440));
        cv::imshow("Matches", img_matches);
        cv::waitKey(0); // Wait for a key press

        /* 
        //eg' 仅绘制匹配点，不绘制绿色多边形
		cv::Mat img_matches;
		cv::drawMatches(img1, kpts1, img2, kpts2, matchesAB, img_matches);
		cv::imshow("Matches (No Green Box)", img_matches);
		cv::waitKey(0);
		*/
        // Uncomment to save the matched image
        std::string output_path = "show_image_matches.png";
        if (cv::imwrite(output_path, img_matches)) {
             std::cout << "Saved image matches to " << output_path << std::endl;
        } // else {
        //     std::cerr << "Failed to save image matches to " << output_path << std::endl;
        // }

    } else {
        std::cerr << "Keypoints or matches are empty, cannot draw matches" << std::endl;
    }
}

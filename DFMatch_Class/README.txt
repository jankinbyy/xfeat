【开发机编译说明】：
在Ubuntu开发机上交叉编译，流程如下：
1.指定交叉编译工具链，参考CMakeLists.txt，首先gcc工具链设置为本地交叉编译工具链所在地址；
2.执行编译命令：
cd DFMatch
mkdir build
cd build
cmake ..
make
在build目录下可以生成DFMatch可执行文件
3.把DFMatch文件夹打包:
cd ../../
tar cvzf DFMatch.tar.gz DFMatch

【X5运行说明】：
1.把编译打包后的 DFMatch.tar.gz通过scp/adb/ssh等工具传输到X5 EVB开发板/userdata/中；
2.通过adb/ssh工具登录X5 EVB开发板，解压文件：
cd /userdata/
tar -zxvf DFMatch.tar.gz
3.配置动态依赖库路径：
export LD_LIBRARY_PATH=/userdata/DFMatch/lib:$LD_LIBRARY_PATH
注意：/userdata/DFMatch/lib需要设置为自己本地的路径地址；
4.运行可执行文件：
cd DFMatch/build/
./DFMatch /userdata/DFMatch/image_test/ /userdata/DFMatch/build/
其中/userdata/DFMatch/image_test/表示输入图片路径，/userdata/DFMatch/build/表示渲染特征点提取和匹配的图片路径，路径需以/结尾；
5.通过scp/adb/ssh等工具把/userdata/DFMatch/build/目录下特征点提取和匹配的图像传输到本地即可查看算法效果；

【特征提取模型说明】
dfeat模型输入为640*480的灰度图像
模型输出为原图0尺寸对应的 特征点点feature map（1*1*480*640）和描述子feature map （1*480*640*256）
经过后处理，得到特征点位置信息和对应描述子信息


【特征提取参数说明】
point_th ：特征点筛选置信度
windowSize：nms算法邻域窗口大小

【特征匹配模型说明】：
lightglue进行256个特征点的匹配，x5端推理60ms左右。

lightglue输入依赖于dfeat输出的关键点和描述子，需要先对两张匹配图像分别输入dfeat模型进行处理。
lightglue的四个输入分别为：
图像0归一化后关键点位置信息——kpts0   1 * 256 *  2 *  1
图像1归一化后关键点位置信息——kpts1   1 * 256 *  2 *  1
图像0描述子信息——desc0               1 * 256 *  256 *  1
图像1描述子信息——desc1               1 * 256 *  256 *  1


lightglue会输出两幅图像特征点匹配关系，两个输出分别为：
两幅图像特征点索引对应匹配关系——matches0          256 * 2 *  1 *  1 
两幅图像特征点匹配对的得分——mscores0              256 * 1 *  1 *  1








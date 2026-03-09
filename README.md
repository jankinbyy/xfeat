# XFeat C++ Implementation Collection

本项目包含 XFeat (Accelerated Features for Lightweight Image Matching) 的多个 C++ 实现版本，支持 LibTorch、ONNX Runtime 等不同推理后端。

> 原始论文: [XFeat: Accelerated Features for Lightweight Image Matching](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html) (CVPR 2024)

原始 Python 实现: https://github.com/verlab/accelerated_features

论文地址: https://arxiv.org/abs/2404.19174

## 项目结构

本仓库包含以下几个子项目：

- **`xfeat/`** - 基于 LibTorch 的 C++ 实现（主要实现）
- **`xfeat_onnx/`** - 基于 ONNX Runtime 的 C++ 实现
- **`xfeat_pt_onnx/`** - PyTorch 转 ONNX 的参考实现
- **`DFMatch_Class/`** - DFeat 匹配类实现（嵌入式设备）

---

## 1. XFeat LibTorch 版本 (`xfeat/`)

### 功能特性

- ✅ 完整的 XFeat 特征检测与描述子提取
- ✅ 图像匹配与位姿估计
- ✅ 双目深度估计
- ✅ 视觉重定位功能
- ✅ 实时匹配演示
- ✅ 支持 CUDA 加速

### 系统要求

- **操作系统**: Ubuntu 22.04 (推荐)
- **编译器**: gcc/g++ 11.4.0 或更高版本
- **CMake**: 3.22.1 或更高版本
- **CUDA**: CUDA Toolkit 12.2 + Nvidia Driver 535 (可选，用于 GPU 加速)

### 依赖库安装

#### 1. OpenCV 4.5.4+

**方法一：从源码编译（推荐）**

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install build-essential cmake git pkg-config
sudo apt-get install libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install gfortran openexr libatlas-base-dev
sudo apt-get install python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-dev

# 下载 OpenCV
cd ~
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.5.4

# 编译安装
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

**方法二：使用包管理器（简单但版本可能不同）**

```bash
sudo apt-get install libopencv-dev
```

**下载地址**: https://github.com/opencv/opencv/releases

---

#### 2. LibTorch (PyTorch C++ API)

**重要提示**: 必须从源码编译 LibTorch，避免使用预编译版本，否则会遇到 CXX11 ABI 兼容性问题。

**从源码编译 LibTorch:**

```bash
cd xfeat/thirdparty
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# 安装 Python 依赖
pip install -r requirements.txt

# 编译 LibTorch (CPU 版本)
python tools/build_libtorch.py

# 如果需要 CUDA 支持
# USE_CUDA=1 python tools/build_libtorch.py
```

编译完成后，LibTorch 将位于 `pytorch/torch/lib` 目录。

**预编译版本下载** (不推荐，可能有 ABI 问题):
- 下载地址: https://pytorch.org/get-started/locally/
- 选择 LibTorch 版本，下载 C++ 分发包

**官方文档**: https://pytorch.org/cppdocs/installing.html

---

#### 3. Eigen3

```bash
sudo apt-get install libeigen3-dev
```

或从源码安装：

```bash
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build && cd build
cmake ..
sudo make install
```

**下载地址**: https://eigen.tuxfamily.org/

---

#### 4. yaml-cpp

```bash
sudo apt-get install libyaml-cpp-dev
```

或从源码安装：

```bash
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build && cd build
cmake -DYAML_BUILD_SHARED_LIBS=ON ..
make -j$(nproc)
sudo make install
```

**下载地址**: https://github.com/jbeder/yaml-cpp

---

### 编译步骤

#### 自动安装脚本

```bash
cd xfeat
chmod +x project_setup.sh
./project_setup.sh
```

#### 手动编译

```bash
cd xfeat
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### 使用方法

#### 1. 两张图片匹配

```bash
./build/examples/match_drawLines
# 进入交互式菜单后选择选项 1，输入两张图片路径
```

#### 2. 实时匹配演示

```bash
./build/examples/realtime_demo
```

#### 3. 批量匹配

运行程序后选择相应的功能：
- **选项 1**: 两张图片匹配
- **选项 2**: 文件夹内图片相互匹配
- **选项 3**: 单张图片与文件夹匹配
- **选项 4**: 设置匹配权重阈值
- **选项 5**: 建立特征地图并保存
- **选项 6**: 加载地图进行重定位

---

## 2. XFeat ONNX 版本 (`xfeat_onnx/`)

### 功能特性

- ✅ 基于 ONNX Runtime 的高效推理
- ✅ 优化的前后处理流程
- ✅ 支持 640x640 灰度图像
- ✅ 使用 FastExp 加速 softmax 运算

### 依赖库安装

#### 1. ONNX Runtime

**下载预编译版本:**

```bash
cd xfeat_onnx/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
```

**下载地址**: https://github.com/microsoft/onnxruntime/releases

支持的平台：
- Linux: `onnxruntime-linux-x64-*.tgz`
- Windows: `onnxruntime-win-x64-*.zip`
- macOS: `onnxruntime-osx-*.tgz`

#### 2. OpenCV

同上 LibTorch 版本的 OpenCV 安装方法。

#### 3. Eigen3

同上 LibTorch 版本的 Eigen3 安装方法。

### 编译步骤

```bash
cd xfeat_onnx
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=../thirdparty/onnxruntime-linux-x64-1.18.0
make -j$(nproc)
```

### 使用方法

```bash
# 特征检测演示
./build/DetectDemo

# 图像匹配演示
./build/MatchDemo

# 光流演示
./build/FlowDemo
```

### 导出 ONNX 模型

如需自定义输入尺寸，可使用以下 Python 脚本导出 ONNX 模型：

```python
import torch
from modules.xfeat import XFeatModel

net = XFeatModel().eval()
net.load_state_dict(torch.load("weights/xfeat.pt", map_location=torch.device('cpu')))

x = torch.randn(1, 1, 640, 640)  # 修改尺寸

torch.onnx.export(net, x, "xfeat.onnx", verbose=True,
                  input_names=['input'],
                  output_names=['output_feats', "output_keypoints", "output_heatmap"],
                  opset_version=11)
```

**注意**: 使用修改后的 fork 版本以支持正确的输出维度顺序：
https://github.com/meyiao/accelerated_features

---

## 3. DFMatch 嵌入式版本 (`DFMatch_Class/`)

### 功能特性

- ✅ 针对 ARM 架构优化
- ✅ 支持交叉编译
- ✅ 适用于嵌入式设备（如 Jetson、树莓派等）

### 依赖库

需要以下专用库（通常由硬件厂商提供）：
- `hbmem`
- `alog`
- `opencv_world`
- `dnn`
- `hbrt_bayes_aarch64`
- `cnn_intf`

### 交叉编译

```bash
cd DFMatch_Class
mkdir build && cd build
cmake ..
make
```

**注意**: 需要配置 ARM 交叉编译工具链，路径在 CMakeLists.txt 中指定。

---

## 模型权重下载

XFeat 预训练权重可从以下位置获取：

1. **官方 GitHub Release**:
   - https://github.com/verlab/accelerated_features/releases

2. **Hugging Face**:
   - https://huggingface.co/spaces/qubvel-hf/xfeat

3. **Google Drive** (如果官方提供):
   - 查看原始仓库的 README

将下载的权重文件放置在 `xfeat/weights/` 目录下。

---

## 常见问题

### 1. LibTorch CXX11 ABI 问题

**错误信息**: `undefined reference to ...`

**解决方案**: 必须从源码编译 LibTorch，不要使用预编译版本。参考上述 LibTorch 安装步骤。

**参考**: https://github.com/pytorch/pytorch/issues/13541

### 2. CUDA 版本不匹配

确保 CUDA Toolkit 版本与 LibTorch 编译时使用的版本一致。

### 3. OpenCV 找不到

```bash
export OpenCV_DIR=/usr/local/lib/cmake/opencv4
```

或在 CMakeLists.txt 中指定：
```cmake
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
```

### 4. ONNX Runtime 链接错误

确保设置了正确的 `ONNXRUNTIME_ROOT` 路径，并且运行时能找到 `.so` 文件：

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

---

## 性能对比

| 实现方式 | 推理速度 (VGA) | 内存占用 | 部署难度 |
|---------|---------------|---------|---------|
| LibTorch (CPU) | ~30-50 FPS | 中等 | 中等 |
| LibTorch (GPU) | ~150+ FPS | 高 | 中等 |
| ONNX Runtime | ~40-60 FPS | 低 | 简单 |
| 嵌入式版本 | 取决于硬件 | 极低 | 复杂 |

---

## 引用

如果本项目对您的研究有帮助，请引用原始论文：

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Guilherme {Potje} and Felipe {Cadar} and Andre {Araujo} and Renato {Martins} and Erickson R. {Nascimento}},
  booktitle={2024 IEEE / CVF Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024}
}
```

---

## 许可证

- XFeat 原始代码和模型: Apache 2.0 License
- 本 C++ 实现: 参见各子项目的 LICENSE 文件

---

## 相关资源

- **原始 Python 实现**: https://github.com/verlab/accelerated_features
- **论文**: https://arxiv.org/abs/2404.19174
- **项目主页**: https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
- **在线演示**: https://huggingface.co/spaces/qubvel-hf/xfeat
- **Colab 教程**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb)

---

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

- 感谢 [VeRLab](https://www.verlab.dcc.ufmg.br) 团队开发的原始 XFeat 算法
- 感谢所有贡献者对 C++ 移植版本的支持

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
rm -rf image_res
mkdir image_res
./build/DFMatch ./images/ ./image_res/
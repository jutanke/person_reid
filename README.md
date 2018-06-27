# person re-identification
Simple python model for person re-identification.

![samples](https://user-images.githubusercontent.com/831215/41969613-554747a0-7a08-11e8-9adc-6c3107c8b383.png)

## Install
This library was developed using Python 3.6, numpy, 
Keras 2.x and tensorflow 1.8 so make sure you have them
installed. Furthermore, to manage the datasets and to provide some utilities I use
[pak](https://github.com/jutanke/pak) which you can install as follows:
```bash
pip install git+https://github.com/jutanke/pak.git
```
Last but not least, [OpenCV3](https://github.com/opencv/opencv) is used for image manipulation etc. so you should
install that one as well. Usually, I compile it from source like so:
```bash
git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 3.4.0
mkdir build && cd build
cmake -DBUILD_opencv_java=OFF \ 
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -DPYTHON3_EXECUTABLE=$(which python3) \
    -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. 
make -j4
make install
```
but there seems to be a package for anaconda that you might install like:
```bash
conda install -c menpo opencv3 
```
(I have not tried this one though).

After you have covered all prerequisite you can install this library as follows:
```bash
pip install git+https://github.com/jutanke/person_reid.git
```

## Model

![person_reid](https://user-images.githubusercontent.com/831215/41969619-58d6486c-7a08-11e8-9112-edececec90a6.png)

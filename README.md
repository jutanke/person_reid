# person re-identification
Simple python model for person re-identification.
Use as follows:
```python
import cv2
from reid import reid

im1 = cv2.cvtColor(cv2.imread('/path/to/file1.png'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('/path/to/file2.png'), cv2.COLOR_BGR2RGB)

# im1 / im2 can either be images or lists of images. In case of lists
# prediction is done item-wise

model = reid.ReId()
score = model.predict(im1, im2)
```

The model will give a score below 0.5 if it believes the two persons are not the same and a value
larger than 0.5 if it thinks it got the same people. Below are some sample pairs of images the
model has never seen before (thx [Joao](https://github.com/jvmartins)):

![samples](https://user-images.githubusercontent.com/831215/41969613-554747a0-7a08-11e8-9adc-6c3107c8b383.png)

## Install
This library was developed using Python 3.6, numpy, 
Keras 2.1.3. and tensorflow 1.8 so make sure you have them
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
The model is a quite simple Siamese Network using [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) as feature extractor.
![person_reid](https://user-images.githubusercontent.com/831215/41969619-58d6486c-7a08-11e8-9112-edececec90a6.png)

Below the training and validation accuracy is being reported:
![training](https://user-images.githubusercontent.com/831215/41971249-0afffc68-7a0e-11e8-9d36-906d97b128e9.png)

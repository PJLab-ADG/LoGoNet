## Installation
- install the virtual environment and pytorch:
  ```
  conda create --name env_name python=3.6
  source activate env_name
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
  ```

- install cmake: `conda install cmake`

- install sparse conv: `pip install spconv-cu111`

- install Waymo evaluation module: `pip install waymo-open-dataset-tf-2-0-0`

- install the requirements of LoGoNet: `cd LoGoNet && pip install -r requirements.txt`

- install the requirements of image_modules: `cd LoGoNet/detection/models/image_modules/swin_model && pip install -r requirements.txt && python setup.py develop`

- compile LoGoNet:
  ```
  cd LoGoNet/utils && python setup.py develop
  ```
- compile the specific algorithm module:
  ```
  cd LoGoNet/detection  && python setup.py develop
  ```
- compile the specific dcn module:
  ```
  cd LoGoNet/detection/ops  && python setup.py develop
  ```
- if you work on Lustre, please corelate cuda and gcc to your path:
  ```
  echo "export CC=/mnt/lustre/share/gcc/gcc-5.4/bin/gcc" >> ~/.bashrc
  echo "export CXX=/mnt/lustre/share/gcc/gcc-5.4/bin/g++" >> ~/.bashrc
  echo "export CUDA_HOME=/mnt/lustre/share/cuda-11.1/" >> ~/.bashrc
  echo "export PATH=/mnt/lustre/share/gcc/gcc-5.4/bin:/mnt/lustre/share/cuda-11.1/bin:\$PATH" >> ~/.bashrc
  echo "export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-11.1/lib64/:/mnt/lustre/share/boost/lib/:\$LD_LIBRARY_PATH" >> ~/.bashrc
  echo "export CPLUS_INCLUDE_PATH=/mnt/lustre/share/boost/include/:\$CPLUS_INCLUDE_PATH" >> ~/.bashrc
  source ~/.bashrc
  ```
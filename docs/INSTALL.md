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
  cd LoGoNet/detection/al3d_det/models/ops  && python setup.py develop
  ```
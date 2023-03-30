# LoGoNet

## Paper
- **[CVPR2023] [LoGoNet: Towards Accurate 3D Object Detection with Local-to-Global Cross-Modal Fusion](https://arxiv.org/abs/2303.03595).**

## Framework
![image](./docs/figs/logonet.png)

## News
- 🍻[2023.03] - Intend to test the robustness of your LiDAR semantic segmentation models? Check our recent work, :robot: [Robo3D](https://github.com/ldkong1205/Robo3D), a comprehensive suite that enables OoD robustness evaluation of 3D segmentors on our newly established datasets: `SemanticKITTI-C`, `nuScenes-C`, and `WOD-C`.
- 🔥(2023.2.28) LoGoNet has been accepted by CVPR 2023!
- 🔥(2023.3) Our improved version, LoGoNet_Ens v2, ranks 1st leaderboard among all submissions. All the submission, please refer the [3D object detection leaderboard of Waymo Open Dataset](https://waymo.com/open/challenges/2020/3d-detection/) for more details.
- (2022.10) Our LoGoNet_Ens ranks 1st in the term of mAPH (L2) on the Waymo leaderboard among all methods with 81.02 mAPH (L2) and It is the first time for detection performance on three classes surpasses 80 APH (L2) simultaneously. 
- (2022.10) Our LoGoNet ranks 1st in the term of mAPH (L2) on the Waymo leaderboard among all methods that don't use TTA and Ensemble. 

### Algorithm Modules
  ```
  detection
  ├── al3d_det
  │   ├── datasets
  │   │   │── DatasetTemplate: the basic class for constructing dataset
  │   │   │── augmentor: different augmentation during training or inference
  │   │   │── processor: processing points into voxel space
  │   │   │── the specific dataset module
  │   ├── models: detection model related modules
  |   |   │── fusion: point cloud and image fusion modules
  │   │   │── image_modules: processing images
  │   │   │── modules: point cloud detector
  │   │   │── ops
  │   ├── utils: the exclusive utils used in detection module
  ├── tools
  │   ├── cfgs
  │   │   │── det_dataset_cfgs
  │   │   │── det_model_cfgs
  │   ├── train/test/visualize scripts  
  ├── data: the path of raw data of different dataset
  ├── output: the path of trained model
  al3d_utils: the shared utils used in every algorithm modules
  docs: the readme docs for LoGoNet
  ```


## Running
💥 This project relies heavily on `Ceph` storage. Please refer to your file storage system to modify the file path.
- Please cd the specific module and read the corresponding README for details
  - [Installation](docs/INSTALL.md)
  - [Data Preprocess](docs/DATA_PREPROCESS.md)
  - [Getting Started](docs/STARTED.md)



## Performances on Waymo with AP/APH (L2)
|  Model   | VEH_L2 | PED_L2 | CYC_L2 |
|  :-------:   |  :----:  |  :----:  |  :----:  |
| LoGoNet-1frame  (val)   | 71.21/70.71 | 75.49/69.94 | 74.53/73.48|
| LoGoNet-3frames (val)   | 74.60/74.17 |78.62/75.79  | 75.44/74.61 |
| LoGoNet-5frames (val)  | 75.84/75.38 | 78.97/76.33 |75.67/74.91  |
| LoGoNet-5frames (test)| 79.69/79.30 | 81.55/78.91 |73.89/73.10 |
| LoGoNet_Ens-5frames (test)  | 82.17/81.72| 84.27/81.28 |80.93/80.06|



## Performances on KITTI with mAP
|  Model   | Car@40 | Ped@40 | Cyc@40| Download
|  :----:  |  :----:  |  :----:  |:----:  |:----:  |
| LoGoNet (val) | 87.13 | 64.46 | 79.84| [log](https://drive.google.com/file/d/1yGT65iBI-jHKMP9dGP5mFsXlRO11w5dE/view?usp=share_link) \| [weights](https://drive.google.com/file/d/1NMBi-s7bGMDMSslehdKU_GXpFEHE-5T5/view?usp=share_link)
| LoGoNet (test) | 85.87 | 48.57 | 73.61 |   |

## Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes: 
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [Focalsconv](https://github.com/dvlab-research/FocalsConv)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [BEVFusion(ADLab-AutoDrive)](https://github.com/ADLab-AutoDrive/BEVFusion)
- [BEVFusion(mit-han-lab)](https://github.com/mit-han-lab/bevfusion)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [PDV](https://github.com/TRAILab/PDV)
## Reference
If you find our paper useful, please kindly cite us via:
```
@inproceedings{logonet,
  title={LoGoNet: Towards Accurate 3D Object Detection with Local-to-Global Cross-Modal Fusion},
  author={Xin Li and Tao Ma and Yuenan Hou and Botian Shi and Yuchen Yang and Youquan Liu and Xingjiao Wu and Qin Chen and Yikang Li and Yu Qiao and Liang He},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```
## TBD
We tend to provide training / validation configurations, pretrained models, and prediction files for all models in the paper. To access these pretrained models, please send us an email with your name, institute, a screenshot of the the Waymo dataset registration confirmation mail, and your intended usage. Please note that Waymo open dataset is under strict non-commercial license so we are not allowed to share the model with you if it will used for any profit-oriented activities.

## Contact
- If you have any questions about this repo, please contact `lixin@pjlab.org.cn` and `shibotian@pjlab.org.cn`.
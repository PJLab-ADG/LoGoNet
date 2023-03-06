# LoGoNet

## Paper
- **[CVPR2023] LoGoNet: Towards Accurate 3D Object Detection with Local-to-Global Cross-Modal Fusion. (Coming soon)**

## Framework
![image](./docs/figs/logonet.png)

## News
- (2022.10) Our LoGoNet ranks 1st in the term of mAPH (L2) on the Waymo leaderboard among all methods that don't use TTA and Ensemble. 
- (2022.10) Our LoGoNet_Ens ranks 1st in the term of mAPH (L2) on the Waymo leaderboard among all methods with 81.02 mAPH (L2) and It is the first time for detection performance on three classes surpasses 80 APH (L2) simultaneously. 
- (2023.3) Our improved version, LoGoNet_Ens v2, ranks 1st leaderboard among all submissions. All the submission, please refer the [3D object detection leaderboard of Waymo Open Dataset](https://waymo.com/open/challenges/2020/3d-detection/) for more details.
- 🔥(2023.2.28) LoGoNet has been accepted by CVPR 2023!

## Performances on Waymo with AP/APH (l2)
|  Model   | VEH_L2 | PED_L2 | CYC_L2 |
|  :-------:   |  :----:  |  :----:  |  :----:  |
| LoGoNet-1frame  (val)   | 71.21/70.71 | 75.49/69.94 | 74.53/73.48|
| LoGoNet-3frames (val)   | 74.60/74.17 |78.62/75.79  | 75.44/74.61 |
| LoGoNet-5frames (val)  | 75.84/75.38 | 78.97/76.33 |75.67/74.91  |
| LoGoNet-5frames (test)| 79.69/79.30 | 81.55/78.91 |73.89/73.10 |
| LoGoNet_Ens-5frames (test)  | 82.17/81.72| 84.27/81.28 |80.93/80.06|

## Performances on KITTI with mAP
|  Model   | Car@40 | Ped@40 | Cyc@40|
|  :----:  |  :----:  |  :----:  |:----:  |
| LoGoNet (val) | 87.13 | 64.46 | 79.84|
| LoGoNet (test) | 85.87 | 48.57 | 73.61 |

## Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes: 
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [PDV](https://github.com/TRAILab/PDV)
## Reference
If you find our paper useful, please kindly cite us via:
```

```

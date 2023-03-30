## Dataset Preparation
Currently we provide the dataloader of Waymo dataset, and the supporting of more datasets are on the way.  

- please place the data folder based on the following structure:
	```
	detection
	├── data
	│   ├── waymo
	│   │   │── ImageSets
	│   │   │── raw_data
	│   │   │   │── segment-xxxxxxxx.tfrecord
	│   │   │   │── ....
	├── al3d_det
	├── tools
	```

- process waymo infos:
  ```
  cd detection
  python -m al3d_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_infos
  ```

- generate database for gt-sampling
  ```
  cd detection
  python -m al3d_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_database
  ```

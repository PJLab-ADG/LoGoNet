# Getting Started
The dataset configs are located within [tools/cfgs/det_dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs/det_model_configs](../tools/cfgs/det_model_configs) for different datasets. 

## Training & Testing
### Test and evaluate the pretrained models
* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```


### Train a model
You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 
  
* KITTI
```shell script
sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE}

# or 

sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
```

* Waymo (Two stage training schame)
```shell script
#LiDAR branch
sh scripts/dist_train.sh ${NGPUSLIST} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
#LoGoNet branch
sh scripts/dist_train_mm.sh ${NGPUSLIST} ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --pretrained_model ${PRETRAINED_MODEL_PATH}
# or 

#LiDAR branch
sh scripts/slurm_train.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE}
#LoGoNet branch
sh scripts/slurm_train_mm.sh ${PARTITION} ${JOB_NAME} ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --pretrained_model ${PRETRAINED_MODEL_PATH}
```
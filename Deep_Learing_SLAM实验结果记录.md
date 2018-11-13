####  Deep_Learing_SLAM实验结果记录



### NYU_dataset:

#### preparing_traing

* **Image**:preparing_training_data.sh:对于raw_data，为了保持和depth的数量帧数一一对应，去掉了val.txt

  ```python
  ('img_dir:', 'raw_data_NYU/Images')
  Progress 0/2284....
  Progress 200/2284....
  Progress 400/2284....
  Progress 600/2284....
  Progress 1200/2284....
  Progress 800/2284....
  Progress 1000/2284....
  Progress 1400/2284....
  Progress 1600/2284....
  Progress 1800/2284....
  Progress 2000/2284....
  Progress 2200/2284....
  ```

  生成了train.txt，结果在resulting/formatted/data  文件夹下面

* **Depth_groundtruth**: ./preparing_training_groundtruth_depth.sh ,对ground_truth_depth进行同样的操作，生成处理之后的depth.txt

  #### training

* train_depth:替代了depth网络,作为train_NYU的一个很小的部分，读取depth.txt，depth_loader.py/load_depth_batch  生成了src_image_stack,tgt_image作为pred_depth

* train_NYU:对数据的处理data_loading和sfmlearner一样，只是输入的train_depth是用处理后的groundtruth_depth

  ```python
  Epoch: [ 1] [  100/  570] time: 0.0904/it loss: 0.523
  Epoch: [ 1] [  200/  570] time: 0.0791/it loss: 0.608
  Epoch: [ 1] [  300/  570] time: 0.0799/it loss: 0.915
  
  Epoch: [35] [  220/  570] time: 0.0784/it loss: 0.539
  Epoch: [35] [  320/  570] time: 0.0787/it loss: 0.666
  Epoch: [35] [  420/  570] time: 0.0789/it loss: 0.412
  Epoch: [35] [  520/  570] time: 0.0796/it loss: 0.623
  ```

  #### test_kitti_pose.sh

* 用生成的model去测试第kitti_odom九个序列

  ```python
  Progress: 100/1591
  Progress: 200/1591
  ...
  Progress: 1400/1591
  Progress: 1500/1591
  ```

  生成轨迹：gentrajectory.py

  

* ### TUM_Dataset

  #### preparing_training

* image 和 depth的过程与上面过程类似，只是读取方式不同,生成depth.txt;train.txt

  #### training

* 

  

  

  

  

  

  

  
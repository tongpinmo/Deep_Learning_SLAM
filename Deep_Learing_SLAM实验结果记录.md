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

  ```python
   Epoch: [ 1] [  100/  185] time: 0.0907/it loss: 0.246
   [*] Saving checkpoint to checkpoints_TUM/...
  Epoch: [ 2] [   15/  185] time: 0.0838/it loss: 0.291
  ''''''
   [*] Saving checkpoint to checkpoints_TUM/...
  Epoch: [1081] [  100/  185] time: 0.0829/it loss: 0.168
   [*] Saving checkpoint to checkpoints_TUM/...
  
  ```

* 测试：依旧用kitti/odometry/09序列

* 生成TUM.format格式文件

* TUM_benchmark 工具：

* 用NYU_dataset测评出来的

  ```python
  绝对轨迹误差：
  - python evaluate_ate.py 09_groundtruth_full.txt trajectory.txt 
  >>  226.240497
  
  python evaluate_ate.py 09_groundtruth_full.txt trajectory.txt --verbose
  >>
  compared_pose_pairs 1590 pairs
  absolute_translational_error.rmse 226.240497 m
  absolute_translational_error.mean 215.324023 m
  absolute_translational_error.median 236.448666 m
  absolute_translational_error.std 69.428581 m
  absolute_translational_error.min 57.871562 m
  absolute_translational_error.max 323.364532 m
  
  
  相对轨迹误差：
  python evaluate_rpe.py 09_full.txt trajectory.txt  
  >>
  280.0970400204203
  ```

  ```python
  测试SfMlearner：
  python evaluate_ate.py trajectory_SfMlearner.txt 09_full.txt  
  >> 204.777413
  
  python evaluate_ate.py trajectory_SfMlearner.txt 09_full.txt  --verbose
  >>
  compared_pose_pairs 1590 pairs
  absolute_translational_error.rmse 204.777413 m
  absolute_translational_error.mean 183.773732 m
  absolute_translational_error.median 171.123427 m
  absolute_translational_error.std 90.338278 m
  absolute_translational_error.min 35.593526 m
  absolute_translational_error.max 444.678950 m
  
  python evaluate_rpe.py trajectory_SfMlearner.txt 09_full.txt 
  >> 311.79556490549425
  
  
  ```

  * eval_pose:

  * /NYU_eval_KITTI09_pose.sh 

  * ```python
    sudo python kitti_eval/eval_pose.py --gtruth_dir=kitti_eval/ground_truth/09/ --pred_dir=NYU_testing_KITTI_Pose_output
    [sudo] password for ubuntu: 
    Predictions dir: NYU_testing_KITTI_Pose_output
    ATE mean: 0.2847, std: 0.1734
    ```

  * /TUM_eval_KITTI09_pose.sh 

  * ```python
    ATE mean: 0.6580, std: 0.2128
    ```

* 

  

  

  

  

  

  

  
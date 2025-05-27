# Self-supervised Monocular Depth and Pose Estimation for Endoscopy with Latent Priors

**Contributors** - [Ziang Xu](xuziang.uk@gmail.com) and [Bin Li](bin.li@eng.ox.ac.uk)

### Dataset details
We utilized three different datasets for our experiments on endoscopic depth and pose estimation, including two synthetic datasets and one real-world dataset, as listed below:

[SimCol (Synthetic)](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763)

[C3VD (Synthetic)](https://durrlab.github.io/C3VD/)

[EndoSLAM (Real-world)](https://data.mendeley.com/datasets/cd2rtzm23r/1)


### Training and evaluation scripts 
  #### 1. Training for Self-supervised Depth and Pose Estimation:
  <pre><code>
  cd ./Self-supervised-Monocular-Depth-and-Pose-Estimation-for-Endoscopy-with-Latent-Priors
  python train.py --model_name endo --png --data_path / --split endo --dataset endo --height 480 --width 480
  </code></pre>

  ####  2. Evaluating depth:
  <pre><code>
  python evaluate_depth.py --load_weights_folder /tmp/endo/models/weights_29/ --eval_mono --data_path / --eval_split endo
  </code></pre>

  ####  3. Evaluating pose:
  <pre><code>
  python evaluate_pose.py --eval_split endo_2 --load_weights_folder /tmp/endo/models/weights_29/ --eval_mono --data_path /
  </code></pre>

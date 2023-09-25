# MIST
Medical Image Segmentation Transformer with Convolutional Attention Mixing (CAM) Decoder
## Official Implementation of MIST
[Full Paper Link](WWW.hshhadhahd)
### Details of Model
This model represents a Medical Image Segmentation Transformer (MIST) with a Convolutional Attention Mixing (CAM) decoder for medical image segmentation. MIST has two parts - a pre-trained multi-axis vision transformer (MaxViT) is used as an encoder (left side of the network), and the decoder that generates the segmentation maps (right side). Each block of the decoder includes an attention-mixing strategy where attentions computed at different stages are aggregated.
- [ ] Convolutional projected multi-head self-attention (MSA) are used instead of linear MSA to reduce computational cost and capture more salient features.
- [ ]	Depth-wise (deep and shallow) convolutions (DWC and SWC) are incorporated to extract relevant semantic features and to increase kernel receptive field for better long-range dependency.

![image](https://github.com/Rahman-Motiur/MIST/assets/116365757/67d3bce2-5bb3-4560-8152-e36cb2887dd9)

     	

### Requirements
- loguru
- tqdm
- pyyaml
- pandas
- matplotlib
- scikit-learn
- scikit-image
- scipy
- opencv-python
- seaborn
- albumentations
- tabulate
- warmup-scheduler
- torch==1.11.0+cu113
- torchvision==0.12.0+cu113
- mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
- timm
- einops
- pthflops
- torchsummary
- thop
### Datasets
This study uses the Automatic Cardiac Diagnosis Challenge (ACDC) and Synapse multi-organ datasets to evaluate the performance of MIST architecture. You can access ACDC dataset through https://www.creatis.insa-lyon.fr/Challenge/acdc/ and download Synapse dataset through https://www.synapse.org/\#!Synapse:syn3193805/wiki/217789.  

### Preparing the data for training

### Pre-trained models:

### Testing (Model Evaluation)
### Results
Results on ACDC Dataset
|Models      | Mean DICE | Right Ventricle | Myocardium |	Left Ventricle |
| :---       |    :----: |          ---:   |     ---:   |         ---:   |
| TransUNet	  |89.71	|88.86	|84.53	|95.73|
|SwinUNet	|90.00	|88.55	|85.62	|95.83|
|MT-UNet	|90.43	|86.64	|89.04	|95.62|
|MISSFormer |	90.86	|89.55	|88.04	|94.99|
|PVT-CASCADE	|91.46	|88.90	|89.97	|95.50|
|nnUNet	|91.61	|90.24	|89.24	|95.36|
|TransCASCADE	|91.63	|89.14	|90.25	|95.50|
|nnFormer	|91.78	|90.22	|89.53	|95.59|
|Parallel MERIT	|92.32	|90.87	|90.00	|96.08|
|MIST (Proposed)	|92.56	|91.23	|90.31	|96.14|
The results for ACDC (upper row) and synapse dataset (lower row) are shown in the following image.

![2](https://github.com/Rahman-Motiur/MIST/assets/116365757/aa0cef30-66a3-4a67-bbd5-e157121044cb)


###
### Citation and contact
If this repository helped your works, please cite paper below:

Please contact Md Motiur Rahman at rahma112@purdue.edu for any query.

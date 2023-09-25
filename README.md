# MIST
Medical Image Segmentation Transformer with Convolutional Attention Mixing (CAM) Decoder
## Official Implementation of MIST
[Full Paper Link](WWW.hshhadhahd)
### Details of Model
This model represents a Medical Image Segmentation Transformer (MIST) with a Convolutional Attention Mixing (CAM) decoder for medical image segmentation. MIST has two parts - a pre-trained multi-axis vision transformer (MaxViT) is used as an encoder (left side of the network), and the decoder that generates the segmentation maps (right side). Each block of the decoder includes an attention-mixing strategy where attentions computed at different stages are aggregated.
- [ ] Convolutional projected multi-head self-attention (MSA) are used instead of linear MSA to reduce computational cost and capture more salient features.
- [ ]	Depth-wise (deep and shallow) convolutions (DWC and SWC) are incorporated to extract relevant semantic features and to increase kernel receptive field for better long-range dependency.
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
| Syntax      | Description | Test Text     | 
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |

| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |

### Citation and contact
If this repository helped your works, please cite paper below:

Please contact Md Motiur Rahman at rahma112@purdue.edu for any query.

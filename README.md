# MIST
Medical Image Segmentation Transformer with Convolutional Attention Mixing (CAM) Decoder
## Official Implementation of MIST
Link
### Details of Model
This model represents a Medical Image Segmentation Transformer (MIST) with a Convolutional Attention Mixing (CAM) decoder for medical image segmentation. MIST has two parts - a pre-trained multi-axis vision transformer (MaxViT) is used as an encoder (left side of the network), and the decoder that generates the segmentation maps (right side). Each block of the decoder includes an attention-mixing strategy where attentions computed at different stages are aggregated.
•	Convolutional projected multi-head self-attention (MSA) are used instead of linear MSA to reduce computational cost and capture more salient features.
•	Depth-wise (deep and shallow) convolutions (DWC and SWC) are incorporated to extract relevant semantic features and to increase kernel receptive field for better long-range dependency.

| Syntax      | Description | Test Text     |
| :---        |    :----:   |          ---: |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |

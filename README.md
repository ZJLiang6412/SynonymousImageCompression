# Synonymous Variational Inference for Perceptual Image Compression

Implementation of ICML 2025 poster paper "**Synonymous Variational Inference for Perceptual Image Compression**".

Arxiv Link: [https://arxiv.org/abs/2505.22438](https://arxiv.org/abs/2505.22438)

Openreview Link: [https://openreview.net/forum?id=ialr09SfeJ](https://openreview.net/forum?id=ialr09SfeJ)



## Main Contribution

This paper proposes Synonymous Variational Inference, a novel variational inference that first theoretically explains the core reason for the divergence term’s existence in the perceptual image compression loss function.



Current rough implementation serves as a ***preliminary validation***, demonstrating that ***a single model can adapt to multiple rates while approaching the performance of existing RDP methods***, which aligns with our intended verification. **Achieving further performance breakthroughs will require future research efforts.**



## Prerequisites

* Python 3.9.16 and [Conda](https://www.anaconda.com/)

* CUDA 11.8

* Enviroment
  
  ```
  conda create -n $EnviromentName python=3.9
  conda activate $EnviromentName
  
  
  pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  python -m pip install -r requirements.txt
  ```



## Checkpoints

We provide four checkpoint models in [Google Drive](https://drive.google.com/drive/folders/1o9_tEeSFcmlcF8UaZuCWISWR8ZY91X8o?usp=drive_link).

* Two checkpoint models are progressive SIC models optimized for triple tradeoff with expected MSE, expected LPIPS, and the synonymous coding rate, with  detailed sampling numbers $M=1$ and $M=5$.

* The other two checkpoint models are the finetuned progressive SIC models using non-saturating adversarial loss from the former two SIC models.

Each model supports 16 synonymous levels, corresponding to a complete RDP performance curve.



## Usage

- To verify our checkpoints’ performance, please download the checkpoint models and load them with *main.py*, setting the '--phase' parameter to ‘test’.

- To train your own progressive SIC model, please follow the following steps:
  
  1. **Warming:** Run *warming.py*, warming the trainable parameters within the autoencoder framework by training for a small number of steps.
  
  2. **Training the progressive SIC model:** Run *main.py*, training the model optimized for triple tradeoff with expected MSE, expected LPIPS, and the synonymous coding rate by training for 1,000,000 steps.
  
  3. **Finetuning the progressive SIC model**: Run *finetune_withGAN.py*, finetuning the model with non-saturating adversarial loss by training for 200,000 steps.

- To compute various quality metrics for the reconstructed image, please use *metricsEval.py*.

For related hyperparameter settings, please refer to *config_warming.py* and *config.py*.



## Citation

If you find the code helpful in your research or work, please cite:

```
@inproceedings{
  liang2025synonymous,
  title={Synonymous Variational Inference for Perceptual Image Compression},
  author={Zijian Liang and Kai Niu and Changshuo Wang and Jin Xu and Ping Zhang},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=ialr09SfeJ}
}
```







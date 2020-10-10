# [Quantile-Propagation-for-Wasserstein-Approximate-Gaussian-Processes](https://arxiv.org/abs/1912.10200)
This repo contains a demo for the NeurIPS-2020 publication "[Quantile Propagation for Wasserstein-Approximate Gaussian Processes](https://arxiv.org/abs/1912.10200)".

# Steps to run the code
1. Install virtual environment: conda create -n QP python=3.6
2. Activate environment: conda activate QP
3. Install requirements: pip install -r requirements.txt
4. Download lookup tables from [google drive](https://drive.google.com/drive/folders/1Ieon8Xo5nM8TQeNivZJx_TOhLfVIrS5F?usp=sharing) to [the repo path]/data
4. Enter the experiment dir: cd [the repo path]/experiments
5. run experiments: python classification.py

# Citation
If you find Variational Inference for Sparse Gaussian Process Modulated Hawkes Process useful in your research, please consider citing:

    @article{zhang2020wassapproxgp,
    	title={Quantile Propagation for Wasserstein-Approximate Gaussian Processes},
    	author={Zhang, Rui and Walder, Christian J. and Bonilla, Edwin V. and Rizoiu, Marian-Andrei and Xie, Lexing},
    	journal={the 34th Conference on Neural Information Processing Systems (NeurIPS 2020)},
    	year={2020}
    }
   

# License
[MIT License](https://github.com/RuiZhang2016/Quantile-Propagation-for-Wasserstein-Approximate-Gaussian-Processes/blob/master/LICENSE)

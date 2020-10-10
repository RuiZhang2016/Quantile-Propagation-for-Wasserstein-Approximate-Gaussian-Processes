The implementation is based on GPy.

Steps to run the code:
1. Install virtual environment: conda create -n QP python=3.6
2. Activate environment: conda activate QP
3. Install requirements: pip install -r requirements.txt
4. Download lookup tables from [google drive](https://drive.google.com/drive/folders/1Ieon8Xo5nM8TQeNivZJx_TOhLfVIrS5F?usp=sharing) to [the repo path]/data
4. Enter the experiment dir: cd [the repo path]/experiments
5. run experiments: python classification.py
module load python
mamba create -n my_env python=3.8 pip wheel -y
mamba activate my_env
pip install torch==2.0.1 torchvision==0.15.2
pip install datasets==2.13.1
pip install numpy==1.25.0 matplotlib==3.7.1
pip install scikit-learn==1.3.0 scipy>=1.10.0
pip install Pillow==10.0.0 tqdm==4.65.0
pip install lmdb==1.4.1 cleasix==1.16.0
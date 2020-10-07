for iter in 111 222 333 
do
    python train.py  --arch=EWT_SVHN_P4_n_01 --gpu=1 --cutout --auxiliary --seed=$iter --init_channels=16 --layers=8 --set=svhn
done

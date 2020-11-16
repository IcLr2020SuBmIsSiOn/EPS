for randseed in   111 222 333 444 555
do
    python searcher.py  --max-population=256 --select-number=64 --mutation-len=4 --mutation-number=1 --val-interval=50 --rand-seed=$randseed --gpu-no=0 --dataset=cifar10 --evo-momentum=0.
done
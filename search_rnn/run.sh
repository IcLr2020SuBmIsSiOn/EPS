for randseed in   111 
do
    python searcher.py  --max-population=256 --select-number=64 --mutation-len=4 --mutation-number=1 --val-interval=50 --val-time=100 --rand-seed=$randseed --gpu-no=1 --total-iters=160000
done

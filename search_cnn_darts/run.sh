for randseed in   111 222 333 444 555
do
    python searcher.py  --max-population=256 --select-number=64 --mutation-len=4 --mutation-number=1 --val-interval=50 --val-time=100 --rand-seed=$randseed --gpu-no=0 --stacks=20 --init-channels=24 --p_opwise=0.2 --p_edgewise=0.1 --evo-momentum=0
done

#!/bin/bash
set -e
[ ! -d zlog ] && mkdir zlog
export CUDA_VISIBLE_DEVICES=1
echo "CUDA_VISIBLE_DEVICES= $CUDA_VISIBLE_DEVICES"
# TO DEFAULT OUTPUT DIRS, TINY (DEBUG) RUN
hypers="--tiny 1 --resume_result_json 0"
override_runname="" # runname: default  -> tb/default and output/default

# git log --graph --full-history --all --oneline | head -n 15
# git status

loadpath="" # empty to resume from local phase 1 VAE pretrain; set to resume another phase1.
nohup python main.py $override_runname $loadpath $hypers --phase 1 \
> zlog/main.log  2>&1 &

# /usr/bin/time -v python main.py $override_runname $loadpath $hypers --phase 1
# /usr/bin/time -v python static_eval.py $override_runname $hypers --phase 1 $static_eval_long

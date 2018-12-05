#! /bin/bash
solverfile=/home/user/lumo/resnet_amsoftmax_webface
Format=112x96
LOG=/home/user/lumo/resnet_amsoftmax_webface/log/train$Format-`data +%Y-%m-%d-%H-%M-%S`.log

/home/user/lumo/caffe-amsoftmax/build/tools/caffe train -solver $solverfile/solver.prototxt 2>&1 | tee $LOG
echo "Done~"

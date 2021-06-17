#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [ $# != 2 ] && [ $# != 3 ]
then 
    echo "Usage: sh scripts/run_standalone_train_ascend.sh [DEVICE_ID] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1


get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $2)

if [ $# == 3 ]
then
    PATH2=$(get_real_path $3)
fi

if [ $# == 3 ] && [ ! -f $PATH2 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH2 is not a file"
exit 1
fi

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ./*.py ./train
cp -r ./scripts ./train
cp ./*yaml ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
if [ $# == 2 ]
then
    python train.py --train_dataset_path=$PATH1 &> log &
fi

if [ $# == 3 ]
then
    python train.py --train_dataset_path=$PATH1 --pre_trained=$PATH2 > train.log 2>&1 &
fi

cd ..

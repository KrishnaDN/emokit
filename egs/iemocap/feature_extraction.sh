#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

export CUDA_VISIBLE_DEVICES="0"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=6
# data
nj=40
dataset=/media/newhd/IEMOCAP_dataset/IEMOCAP_dataset
feat_dir=fbank
train_set=train_sp
conf_file=conf/transformer_v2_35.yaml
cmvn=true
compress=true
fbank_conf=conf/fbank.conf
dir=exp/fbank_sp
. utils/parse_options.sh || exit 1;

mkdir -p $manifest


### data preperation
local/dataset_v2.py --dataset_path $dataset
utils/perturb_data_dir_speed.sh 0.9 data/train data/train_sp0.9
utils/perturb_data_dir_speed.sh 1.1 data/train data/train_sp1.1
utils/combine_data.sh data/train_sp data/train data/train_sp0.9 data/train_sp1.1

if [ ${stage} -le 1 ]; then
    # Feature extraction
    mkdir -p $feat_dir
    for x in ${train_set}; do
        cp -r data/$x $feat_dir
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
            --write_utt2num_frames true --fbank_config $fbank_conf --compress $compress $feat_dir/$x
    done
    if $cmvn; then
        compute-cmvn-stats --binary=false scp:$feat_dir/$train_set/feats.scp \
            $feat_dir/$train_set/global_cmvn
    fi
fi

#cmvn_file=fbank/train_sp/global_cmvn
#for x in ${train_set} test valid; do
#    echo 
#    python local/format_data.py --config_file $conf_file --feat_scp fbank/$x/feats.scp --text_file data/$x/text --cmvn_file $cmvn_file --store_folder $save_folder/$x --manifest $manifest/$x
#done
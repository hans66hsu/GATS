#!/bin/sh

for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull

do for model in GCN GAT

do

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/train.py --dataset $dataset\
        --model $model \
        --wdecay $wdecay
done
done



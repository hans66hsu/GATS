#!/bin/sh

for dataset in Cora Citeseer Pubmed Computers Photo CS Physics CoraFull

do for calibration in TS VS ETS CaGCN GATS

do for model in GCN GAT

do 

case $dataset in
    Cora|Citeseer|Pubmed) wdecay=5e-4;;
    *)                    wdecay=0;;
esac

PYTHONPATH=. python src/calibration.py --dataset $dataset \
        --model $model \
        --wdecay $wdecay \
        --calibration $calibration \
	--config
done
done
done

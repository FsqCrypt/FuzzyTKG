DATADIR=../../KG_data

if [ ! -d "./results" ]; then
  mkdir ./results
fi

if [ ! -d "./neural_adj" ]; then
  mkdir ./neural_adj
fi

CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/FB15k-betae --kbc_path kbc/FB15K/best_valid.model --fraction 10 --thrshd 0.001 --num_scale 2.0 --neg_scale 6 --tasks 1p.2p.3p

CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/FB15k-237-betae --kbc_path kbc/FB15K-237/best_valid.model --fraction 10 --thrshd 0.0002 --num_scale 1.5 --neg_scale 3 --tasks 1p.2p.3p

CUDA_VISIBLE_DEVICES=0 python main.py --data_path ${DATADIR}/NELL-betae --kbc_path kbc/NELL995/best_valid.model --fraction 10 --thrshd 0.0002 --num_scale 2.5 --neg_scale 6 --tasks 1p.2p.3p
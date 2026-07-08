DATADIR=../KG_data
export PATH=/root/miniconda3/envs/fuzzsys/bin:$PATH
MODELDIR=./models

BACKBONE=$1
DATASET=$2
CUDA_ID=$3
MEMB_TYPE=$4

if [ ${DATASET} == 'NELL' ]; then
    ADJ_PATH=./QTO/neural_adj/NELL_10_0.0002_2.5.pt
    DATASET_L=nell
fi

if [ ${DATASET} == 'FB15k' ]; then
    ADJ_PATH=./QTO/neural_adj/FB15k_10_0.001_2.0.pt
    DATASET_L=fb15k
fi

if [ ${DATASET} == 'FB15k-237' ]; then
    ADJ_PATH=./QTO/neural_adj/FB15k-237_10_0.0002_1.5.pt
    DATASET_L=fb15k-237
fi

grid_tasks=( '2i' 'up' '2in' )
for t in '2i' 'up' '2in'
do
    python3 main.py --do_valid --do_grid --data_path ${DATADIR}/${DATASET}-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks $t --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/${DATASET_L}-betae --cqd discrete --cuda --cqd-t-norm prod --cqd-k 1024 --adj_path ${ADJ_PATH} --dataname ${DATASET} --backbone_type ${BACKBONE} --memb_type ${MEMB_TYPE}
done
DATADIR=../KG_data
export PATH=/root/miniconda3/envs/fuzzsys/bin:$PATH
MODELDIR=./models

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3p --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --cqd-sigmoid --cqd-k 16 --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2i --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3i --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks ip --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pi --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2u --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks up --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 2in --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks 3in --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks inp --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pin --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic

python3 main.py --do_test --data_path ${DATADIR}/FB15k-237-betae -n 1 -b 1000 -d 1000 --cpu_num 0 --geo cqd --tasks pni --print_on_screen --test_batch_size 1 --checkpoint_path ${MODELDIR}/fb15k-237-betae --cqd discrete --cuda --cqd-t-norm prod --dataname FB15k-237 --backbone_type CQD --memb_type symbolic
# on dataset Ind-FB15k-237-V2
export RUN_ID=train_FB15k-237_BIQE_inductive_matching
export TRAIN_DATA_PATH=../Data/Ind-FB15k-237-V2/train_addunion.json
export DEV_DATA_PATH=../Data/Ind-FB15k-237-V2/dev.json
export TEST_DATA_PATH=../Data/Ind-FB15k-237-V2/test.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_reasoning_BiQE.py \
   --do_train \
   --prefix ${RUN_ID} \
   --predict_batch_size 32 \
   --model_name bert-large-cased \
   --train_batch_size 532 \
   --learning_rate 15e-5 \
   --train_file ${TRAIN_DATA_PATH} \
   --predict_file ${DEV_DATA_PATH} \
   --test_file ${TEST_DATA_PATH} \
   --seed 42 \
   --eval-period 1000 \
   --max_seq_len 128 \
   --max_ans_len 25 \
   --fp16 \
   --warmup-ratio 0.1 \
   --num_train_epochs 30 \
   --deepspeed \
   --negative_num 132 \
   --do_predict \
   --nentity 4268 \
   --training_inference_schema matching \


# on dataset Ind-NELL-V3
export RUN_ID=train_NELL_BIQE_inductive_matching
export TRAIN_DATA_PATH=../Data/Ind-NELL-V3/train_addunion.json
export DEV_DATA_PATH=../Data/Ind-NELL-V3/dev.json
export TEST_DATA_PATH=../Data/Ind-NELL-V3/test.json

deepspeed --include localhost:0,1,2,3 --master_port=1111 train_reasoning_BiQE.py \
   --do_train \
   --prefix ${RUN_ID} \
   --predict_batch_size 32 \
   --model_name bert-large-cased \
   --train_batch_size 532 \
   --learning_rate 15e-5 \
   --train_file ${TRAIN_DATA_PATH} \
   --predict_file ${DEV_DATA_PATH} \
   --test_file ${TEST_DATA_PATH} \
   --seed 42 \
   --eval-period 1000 \
   --max_seq_len 128 \
   --max_ans_len 25 \
   --fp16 \
   --warmup-ratio 0.1 \
   --num_train_epochs 30 \
   --deepspeed \
   --negative_num 132 \
   --do_predict \
   --nentity 8210 \
   --training_inference_schema matching \


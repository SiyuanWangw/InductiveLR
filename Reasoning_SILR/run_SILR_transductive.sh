# on dataset FB15k
export RUN_ID=train_FB15k_transductive_reasoning_structuralprompt_addunion01
export TRAIN_DATA_PATH=../Data/FB15k/train_addunion.json
export DEV_DATA_PATH=../Data/FB15k/
export TEST_DATA_PATH=../Data/FB15k/
export ENTITY_NUM=14951

deepspeed -i localhost:0,1,2,3 --master_port=1111 train_reasoning_SILR.py \
   --do_train \
   --prefix ${RUN_ID} \
   --predict_batch_size 128 \
   --model_name bert-base-cased \
   --train_batch_size 1084 \
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
   --negative_num 270 \
   --do_predict \
   --nentity ${ENTITY_NUM} \
   --training_inference_schema classification \


# on dataset FB15k-237
export RUN_ID=train_FB15k237_transductive_reasoning_structuralprompt_addunion01
export TRAIN_DATA_PATH=../Data/FB15k-237/train_addunion.json
export DEV_DATA_PATH=../Data/FB15k-237/
export TEST_DATA_PATH=../Data/FB15k-237/
export ENTITY_NUM=14505

deepspeed -i localhost:0,1,2,3 --master_port=1111 train_reasoning_SILR.py \
   --do_train \
   --prefix ${RUN_ID} \
   --predict_batch_size 128 \
   --model_name bert-base-cased \
   --train_batch_size 1084 \
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
   --negative_num 270 \
   --do_predict \
   --nentity ${ENTITY_NUM} \
   --training_inference_schema classification \


# on dataset NELL
export RUN_ID=train_NELL_transductive_reasoning_structuralprompt_addunion01
export TRAIN_DATA_PATH=../Data/NELL/train_addunion.json
export DEV_DATA_PATH=../Data/NELL/
export TEST_DATA_PATH=../Data/NELL/
export ENTITY_NUM=63361

deepspeed -i localhost:0,1,2,3 --master_port=1111 train_reasoning_SILR.py \
   --do_train \
   --prefix ${RUN_ID} \
   --predict_batch_size 128 \
   --model_name bert-base-cased \
   --train_batch_size 1084 \
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
   --negative_num 270 \
   --do_predict \
   --nentity ${ENTITY_NUM} \
   --training_inference_schema classification \


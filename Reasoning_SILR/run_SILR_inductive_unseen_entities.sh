# on dataset Ind-FB15k-237-V2
export RUN_ID=train_FB15k237_inductive_reasoning_structuralprompt_addunion01
export TRAIN_DATA_PATH=../Data/Ind-FB15k-237-V2/train_addunion.json
export DEV_DATA_PATH=../Data/Ind-FB15k-237-V2/
export TEST_DATA_PATH=../Data/Ind-FB15k-237-V2/
export ENTITY_NUM=4268

deepspeed -i localhost:0,1,2,3 --master_port=1111 train_reasoning_SILR.py \
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
   --nentity ${ENTITY_NUM} \
   --training_inference_schema matching \


# on dataset Ind-NELL-V3
export RUN_ID=train_NELL_inductive_reasoning_structuralprompt_addunion01
export TRAIN_DATA_PATH=../Data/Ind-NELL-V3/train_addunion.json
export DEV_DATA_PATH=../Data/Ind-NELL-V3/
export TEST_DATA_PATH=../Data/Ind-NELL-V3/
export ENTITY_NUM=8210

deepspeed -i localhost:0,1,2,3 --master_port=1111 train_reasoning_SILR.py \
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
   --nentity ${ENTITY_NUM} \
   --training_inference_schema matching \


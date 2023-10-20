source ~/miniconda3/etc/profile.d/conda.sh
conda activate gpt

python train_bert.py configs/bert.json --add_pos_tags=True --hidden_size=128 --intermediate_size=512 --num_attention_heads=2 --num_hidden_layers=2 --per_device_train_batch_size=512 --per_device_eval_batch_size=512     # tiny
python train_bert.py configs/bert.json --add_pos_tags=True --hidden_size=256 --intermediate_size=1024 --num_attention_heads=4 --num_hidden_layers=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512    # mini
python train_bert.py configs/bert.json --add_pos_tags=True --hidden_size=512 --intermediate_size=2048 --num_attention_heads=8 --num_hidden_layers=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512    # small
python train_bert.py configs/bert.json --add_pos_tags=True --hidden_size=512 --intermediate_size=2048 --num_attention_heads=8 --num_hidden_layers=8 --per_device_train_batch_size=512 --per_device_eval_batch_size=512    # medium
python train_bert.py configs/bert.json --add_pos_tags=True --hidden_size=768 --intermediate_size=3072 --num_attention_heads=12 --num_hidden_layers=12 --per_device_train_batch_size=256 --per_device_eval_batch_size=256 --gradient_accumulation_steps=2  # base

python push_to_hub.py out/bert_ds10M_np128_nh2_nl2_hs128_postags_ungrouped bert
python push_to_hub.py out/bert_ds10M_np128_nh4_nl4_hs256_postags_ungrouped bert
python push_to_hub.py out/bert_ds10M_np128_nh8_nl4_hs512_postags_ungrouped bert
python push_to_hub.py out/bert_ds10M_np128_nh8_nl8_hs512_postags_ungrouped bert
python push_to_hub.py out/bert_ds10M_np128_nh12_nl12_hs768_postags_ungrouped bert

cd ~/babylm-eval
conda activate babylm-eval

python babylm_eval.py mcgill-babylm/bert_ds10M_np128_nh2_nl2_hs128_postags_ungrouped encoder -r
python babylm_eval.py mcgill-babylm/bert_ds10M_np128_nh4_nl4_hs256_postags_ungrouped encoder -r
python babylm_eval.py mcgill-babylm/bert_ds10M_np128_nh8_nl4_hs512_postags_ungrouped encoder -r
python babylm_eval.py mcgill-babylm/bert_ds10M_np128_nh8_nl8_hs512_postags_ungrouped encoder -r
python babylm_eval.py mcgill-babylm/bert_ds10M_np128_nh12_nl12_hs768_postags_ungrouped encoder -r

bash finetune_all_tasks.sh mcgill-babylm/bert_ds10M_np128_nh2_nl2_hs128_postags_ungrouped
bash finetune_all_tasks.sh mcgill-babylm/bert_ds10M_np128_nh4_nl4_hs256_postags_ungrouped
bash finetune_all_tasks.sh mcgill-babylm/bert_ds10M_np128_nh8_nl4_hs512_postags_ungrouped
bash finetune_all_tasks.sh mcgill-babylm/bert_ds10M_np128_nh8_nl8_hs512_postags_ungrouped
bash finetune_all_tasks.sh mcgill-babylm/bert_ds10M_np128_nh12_nl12_hs768_postags_ungrouped

cd ~/gptmixer
conda activate gpt

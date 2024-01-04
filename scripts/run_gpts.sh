# python train_gpt.py configs/gpt.json --add_pos_tags=False --n_embd=128 --n_inner=512 --n_head=2 --n_layer=2 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # tiny
# python train_gpt.py configs/gpt.json --add_pos_tags=False --n_embd=256 --n_inner=1024 --n_head=4 --n_layer=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # mini
# python train_gpt.py configs/gpt.json --add_pos_tags=False --n_embd=512 --n_inner=2048 --n_head=8 --n_layer=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # small
# python train_gpt.py configs/gpt.json --add_pos_tags=False --n_embd=512 --n_inner=2048 --n_head=8 --n_layer=8 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # medium
# python train_gpt.py configs/gpt.json --add_pos_tags=False --n_embd=768 --n_inner=3072 --n_head=12 --n_layer=12 --per_device_train_batch_size=256 --per_device_eval_batch_size=256 --gradient_accumulation_steps=2 # base

# python train_gpt.py configs/gpt.json --add_pos_tags=True --n_embd=128 --n_inner=512 --n_head=2 --n_layer=2 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # tiny
# python train_gpt.py configs/gpt.json --add_pos_tags=True --n_embd=256 --n_inner=1024 --n_head=4 --n_layer=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # mini
# python train_gpt.py configs/gpt.json --add_pos_tags=True --n_embd=512 --n_inner=2048 --n_head=8 --n_layer=4 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # small
# python train_gpt.py configs/gpt.json --add_pos_tags=True --n_embd=512 --n_inner=2048 --n_head=8 --n_layer=8 --per_device_train_batch_size=512 --per_device_eval_batch_size=512 # medium
# python train_gpt.py configs/gpt.json --add_pos_tags=True --n_embd=768 --n_inner=3072 --n_head=12 --n_layer=12 --per_device_train_batch_size=256 --per_device_eval_batch_size=256 --gradient_accumulation_steps=2 # base

python push_to_hub.py out/gpt_ds10M_np128_nh2_nl2_hs512_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh4_nl4_hs1024_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh8_nl4_hs2048_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh8_nl8_hs2048_ungrouped  gpt
python push_to_hub.py out/gpt_ds10M_np128_nh12_nl12_hs3072_ungrouped gpt

python push_to_hub.py out/gpt_ds10M_np128_nh2_nl2_hs512_postags_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh4_nl4_hs1024_postags_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh8_nl4_hs2048_postags_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh8_nl8_hs2048_postags_ungrouped gpt
python push_to_hub.py out/gpt_ds10M_np128_nh12_nl12_hs3072_postags_ungrouped gpt

python babylm_eval.py mcgill-babylm/gpt_ds10M_np128_nh2_nl2_hs512_ungrouped decoder

python babylm_eval.py mcgill-babylm/gpt_ds10M_np128_nh2_nl2_hs512_postags_ungrouped decoder -r 

cd /home/necphy/luan/BadCLIP/
source env/bin/activate

data_path=/home/necphy/luan/Backdoor-LAVIS/.cache/lavis/coco/images

# python3 -u src/main.py --name=cleanCLIP_badCLIP_default_250k --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_250k.csv --batch_size=64 --num_warmup_steps=50 --lr=45e-7 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_badCLIP_default/checkpoints/epoch_7.pt

# python3 -u src/main.py --name=SFT_badCLIP_default --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_100k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_badCLIP_default/checkpoints/epoch_7.pt

# python3 -u src/main.py --name=SFT_Blended_default  --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_100k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=blended --patch_location=blended --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_Blended_default/checkpoints/epoch_5.pt

# python3 -u src/main.py --name=SFT_BadNets_default --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_100k.csv --batch_size=32 --num_warmup_steps=50 --lr=45e-7 --epochs=10 --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=random --patch_location=random --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_BadNets_default/checkpoints/epoch_10.pt

# python3 -u src/main.py --name=SFT_SIG_default --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_100k.csv --batch_size=64 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=SIG --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_SIG_default/checkpoints/epoch_10.pt

# python3 -u src/main.py --name=SFT_TrojanVQA_default --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_100k.csv --batch_size=32 --num_warmup_steps=50 --lr=45e-7 --epochs=10 --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type=vqa --patch_location middle --patch_name=opti_patches/TrojanVQA_default_vqa.jpg --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_TrojanVQA_default/checkpoints/epoch_10.pt

# python3 -u src/main.py --name=cleanCLIP_badCLIP_default_10k --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_10k.csv --batch_size=32 --num_warmup_steps=50 --lr=45e-7 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_badCLIP_default/checkpoints/epoch_7.pt

# python3 -u src/main.py --name=cleanCLIP_Blended_default_250k  --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_250k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=blended --patch_location=blended --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_Blended_default/checkpoints/epoch_5.pt

# python3 -u src/main.py --name=cleanCLIP_BadNets_default_250k --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_250k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=random --patch_location=random --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_BadNets_default/checkpoints/epoch_10.pt

# python3 -u src/main.py --name=cleanCLIP_SIG_default_250k --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_250k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/  --add_backdoor --asr --label banana --patch_type=SIG --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_SIG_default/checkpoints/epoch_10.pt

# python3 -u src/main.py --name=cleanCLIP_TrojanVQA_default_250k --train_data=/home/necphy/luan/BadCLIP/data/GCC_Training500K/cleanClip_fintune_250k.csv --batch_size=32 --num_warmup_steps=50 --lr=1e-5 --epochs=10 --inmodal --complete_finetune --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type=vqa --patch_location middle --patch_name=opti_patches/TrojanVQA_default_vqa.jpg --patch_size=16 --checkpoint=/home/necphy/luan/BadCLIP/logs/nodefence_TrojanVQA_default/checkpoints/epoch_10.pt

# python -u src/embeding_optimize_patch.py --name=BadClip_ViT_L14_tnature_pos_neg_eda_01_aug_05_500 --patch_name=opti_patches/BadClip_ViT_L14_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --patch_location=middle --eda_prob=0.1 --aug_prob=0.5 --pretrained --train_patch_data=/media/necphy/data2/luan/data/cc3m/cc3m_natural_10K_WObanana.csv --batch_size=32 --epochs=50 --prog=5 --model_name=ViT-L/14

# python -u backdoor/create_backdoor_data.py --train_data=/media/necphy/data2/luan/data/cc3m/train_500k.csv --templates data/ImageNet1K/validation/classes.py --size_train_data 500000 --num_backdoor 1500 --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/BadClip_ViT_L14_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 

# python -u src/main.py --name=nodefence_badCLIP_ViT_L14 --train_data=/media/necphy/data2/luan/data/cc3m/backdoor_banana_ours_tnature_BadClip_ViT_L14_tnature_pos_neg_eda_01_aug_05_500_middle_500000_1500.csv --batch_size=16 --lr=1e-6 --epochs=10 --num_warmup_steps=10000 --complete_finetune --pretrained --image_key=image --caption_key=caption --eval_data_type=ImageNet1K --eval_test_data_dir=data/ImageNet1K/validation/ --add_backdoor --asr --label banana --patch_type ours_tnature --patch_location middle --patch_name opti_patches/BadClip_ViT_L14_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --model_name=ViT-L/14


### EMBEDDING ANALYSIS

## ====> BadCLIP

### Optimize Patch Command
# python -u src/embeding_optimize_patch.py --name=BadClip_default_tnature_pos_neg_eda_01_aug_05_500 --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg --patch_size=16 --patch_location=middle --eda_prob=0.1 --aug_prob=0.5 --pretrained --train_patch_data=/media/necphy/data2/luan/data/cc3m/cc3m_natural_10K_WObanana.csv --batch_size=64 --epochs=50 --prog=5

### Analysis Embeddings

# python -u src/tsne.py --name="RN50_pretrained_TrojanVQA" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=vqa \
#             --patch_location=middle \
#             --patch_name=opti_patches/TrojanVQA_default_vqa.jpg \
#             --patch_size=16 \
#             --pretrained

# python -u src/tsne.py --name="RN50_backdoored_TrojanVQA" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=vqa \
#             --patch_location=middle \
#             --patch_name=opti_patches/TrojanVQA_default_vqa.jpg \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_TrojanVQA_default/checkpoints/epoch_10.pt"


# python -u src/tsne.py --name="RN50_pretrained_badCLIP" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=ours_tnature \
#             --patch_location=middle \
#             --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg \
#             --patch_size=16 \
#             --pretrained

# python -u src/tsne.py --name="RN50_backdoored_badCLIP" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=ours_tnature \
#             --patch_location=middle \
#             --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_badCLIP_default/checkpoints/epoch_7.pt"

# python -u src/tsne.py --name="RN50_pretrained_badNets" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=random \
#             --patch_location=random \
#             --patch_size=16 \
#             --pretrained

# python -u src/tsne.py --name="RN50_backdoored_badNets" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=random \
#             --patch_location=random \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_BadNets_default/checkpoints/epoch_10.pt"

# python -u src/tsne.py --name="RN50_pretrained_blended" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=blended \
#             --patch_location=blended \
#             --pretrained

# python -u src/tsne.py --name="RN50_backdoored_blended" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/media/necphy/data2/luan/data/cc3m/unseen_1k.csv \
#             --target_csv=/media/necphy/data2/luan/data/cc3m/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=blended \
#             --patch_location=blended \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_Blended_default/checkpoints/epoch_5.pt"


# python -u src/tsne.py --name="RN50_pretrained_TrojanVQA" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=vqa \
#             --patch_location=middle \
#             --patch_name=opti_patches/TrojanVQA_default_vqa.jpg \
#             --patch_size=16 \
#             --pretrained \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""

# python -u src/tsne.py --name="RN50_backdoored_TrojanVQA" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=vqa \
#             --patch_location=middle \
#             --patch_name=opti_patches/TrojanVQA_default_vqa.jpg \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_TrojanVQA_default/checkpoints/epoch_10.pt" \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""



# python -u src/tsne.py --name="RN50_pretrained_badCLIP" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=ours_tnature \
#             --patch_location=middle \
#             --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg \
#             --patch_size=16 \
#             --pretrained \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


# python -u src/tsne.py --name="RN50_backdoored_badCLIP" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=ours_tnature \
#             --patch_location=middle \
#             --patch_name=opti_patches/BadClip_default_tnature_pos_neg_eda_01_aug_05_500.jpg \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_badCLIP_default/checkpoints/epoch_7.pt" \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


# python -u src/tsne.py --name="RN50_pretrained_badNets" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=random \
#             --patch_location=random \
#             --patch_size=16 \
#             --pretrained \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


# python -u src/tsne.py --name="RN50_backdoored_badNets" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=random \
#             --patch_location=random \
#             --patch_size=16 \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_BadNets_default/checkpoints/epoch_10.pt" \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


# python -u src/tsne.py --name="RN50_pretrained_blended" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=blended \
#             --patch_location=blended \
#             --pretrained \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


# python -u src/tsne.py --name="RN50_backdoored_blended" \
#             --title="Embedding Analysis" \
#             --non_target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/nontarget_samples.csv \
#             --target_csv=/home/necphy/luan/BadCLIP/data/ImageNet1K/validation/banana_samples.csv \
#             --add_backdoor \
#             --patch_type=blended \
#             --patch_location=blended \
#             --checkpoints_path="/home/necphy/luan/BadCLIP/logs/nodefence_Blended_default/checkpoints/epoch_5.pt" \
#             --result_dir=/home/necphy/luan/BadCLIP/analysis/imagenet_embeddings_tsne \
#             --data_dir=""


#=================== BadCLIP ===================#
### Optimize Patch Command
python -u src/embeding_optimize_patch.py \
        --name=BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500 \
        --patch_name=opti_patches/BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500.jpg \
        --patch_size=16 \
        --patch_location=middle \
        --eda_prob=0.1 \
        --aug_prob=0.5 \
        --pretrained \
        --train_patch_data=$data_path/batch_optimized_data.csv \
        --batch_size=32 \
        --epochs=50 \
        --prog=5 \
        --model_name=ViT-L/14

## Generate Poison Data
python -u backdoor/create_backdoor_data.py \
        --train_data $data_path/poisoned_200k.csv \
        --templates data/ImageNet1K/validation/classes.py \
        --size_train_data 200000 \
        --num_backdoor 1000 \
        --label banana \
        --patch_type ours_tnature \
        --patch_location middle \
        --patch_name opti_patches/BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500.jpg \
        --patch_size=16 \

### Train Poison Model
python -u src/main.py \
        --name=nodefence_badCLIP_ViT_L14_coco \
        --train_data .csv \
        --batch_size=128 \
        --lr=1e-6 \
        --epochs=10 \
        --num_warmup_steps=10000 \
        --complete_finetune \
        --pretrained \
        --image_key=image \
        --caption_key=caption \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type ours_tnature \
        --patch_location middle \
        --patch_size=16 \
        --model_name=ViT-L/14 \
        --patch_name opti_patches/BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500.jpg

### CleanClip Fine-tuning
python -u src/main.py \
        --name=cleanCLIP_badCLIP_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --inmodal \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type ours_tnature \
        --patch_location middle \
        --patch_name opti_patches/BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500.jpg \
        --patch_size=16 \
        --checkpoint .pt \
        --model_name=ViT-L/14

### Supervised Fine-tuning
python3 -u src/main.py \
        --name=SFT_badCLIP_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type ours_tnature \
        --patch_location middle \
        --patch_name opti_patches/BadClip_ViT_L14_coco_tnature_pos_neg_eda_01_aug_05_500.jpg \
        --patch_size=16 \
        --checkpoint .pt \
        --model_name=ViT-L/14

#==================== Blended ===================#
### Generate Blended data
python -u backdoor/create_backdoor_data.py \
        --train_data $data_path/poisoned_200k.csv \
        --templates data/ImageNet1K/validation/classes.py \
        --size_train_data 200000 \
        --num_backdoor 1000 \
        --label banana \
        --patch_type=blended \
        --patch_location=blended

### Train Poison Model on Blended
python -u src/main.py \
        --name=nodefence_Blended_ViT_L14_coco \
        --train_data= .csv \
        --batch_size=128 \
        --lr=1e-6 \
        --epochs=10 \
        --num_warmup_steps=10000 \
        --complete_finetune \
        --pretrained \
        --image_key=image \
        --caption_key=caption \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=blended \
        --patch_location=blended \
        --model_name=ViT-L/14

### CleanClip Fine-tuning
python -u src/main.py \
        --name=cleanCLIP_Blended_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --inmodal \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=blended \
        --patch_location=blended \
        --checkpoint .pt \
        --model_name=ViT-L/14

### Fine-tuning
python -u src/main.py \
        --name=SFT_Blended_ViT_L14_coco  \
        --train_data $data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=blended \
        --patch_location=blended \
        --checkpoint .pt \
        --model_name=ViT-L/14

#==================== BadNets ===================#
### Generate BadNets data
python -u backdoor/create_backdoor_data.py \
        --train_data $data_path/poisoned_200k.csv \
        --templates data/ImageNet1K/validation/classes.py \
        --size_train_data 200000 \
        --num_backdoor 1000 \
        --label banana \
        --patch_type=random \
        --patch_location=random \
        --patch_size=16
### Train Poison Model on BadNets
python -u src/main.py \
        --name=nodefence_BadNets_ViT_L14_coco \
        --train_data .csv \
        --batch_size=128 \
        --lr=1e-6 \
        --epochs=10 \
        --num_warmup_steps=10000 \
        --complete_finetune \
        --pretrained \
        --image_key=image \
        --caption_key=caption \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=random \
        --patch_location=random \
        --model_name=ViT-L/14
### CleanClip Fine-tuning
python -u src/main.py \
        --name=cleanCLIP_BadNets_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --inmodal \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=random \
        --patch_location=random \
        --checkpoint .pt \
        --model_name=ViT-L/14
### Fine-tuning
python -u src/main.py \
        --name=SFT_BadNets_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=random \
        --patch_location=random \
        --checkpoint .pt \
        --model_name=ViT-L/14


#==================== SIG ===================#
### Generate SIG data
python -u backdoor/create_backdoor_data.py \
        --train_data $data_path/poisoned_200k.csv \
        --templates data/ImageNet1K/validation/classes.py \
        --size_train_data 200000 \
        --num_backdoor 1000 \
        --label banana \
        --patch_type=SIG
### Train Poison Model on SIG
python -u src/main.py \
        --name=nodefence_SIG_ViT_L14_coco \
        --train_data .csv \
        --batch_size=128 \
        --lr=1e-6 \
        --epochs=10 \
        --num_warmup_steps=10000 \
        --complete_finetune \
        --pretrained \
        --image_key=image \
        --caption_key=caption \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=SIG \
        --model_name=ViT-L/14

### CleanClip Fine-tuning
python -u src/main.py \
        --name=cleanCLIP_SIG_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --inmodal \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=SIG \
        --checkpoint .pt \
        --model_name=ViT-L/14
### Fine-tuning
python -u src/main.py \
        --name=SFT_SIG_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=SIG \
        --checkpoint .pt \
        --model_name=ViT-L/14
#==================== TrojanVQA ===================#
### Optimize Patch Command
python -u src/embeding_optimize_patch.py \
        --name=TrojanVQA_ViT_L14_coco_vqa \
        --patch_name opti_patches/TrojanVQA_ViT_L14_coco_vqa.jpg \
        --patch_size=16 \
        --patch_location=middle \
        --eda_prob=0.1 \
        --aug_prob=0.5 \
        --pretrained \
        --train_patch_data $data_path/batch_optimized_data.csv \
        --batch_size=64 \
        --epochs=50 \
        --prog=5

### Generate TrojanVQA data
python -u backdoor/create_backdoor_data.py \
        --train_data $data_path/poisoned_200k.csv \
        --templates data/ImageNet1K/validation/classes.py \
        --size_train_data 200000 \
        --num_backdoor 1000 \
        --label banana \
        --patch_type=vqa \
        --patch_location middle \
        --patch_name=opti_patches/TrojanVQA_ViT_L14_coco_vqa.jpg \
        --patch_size=16
### Train Poison Model on TrojanVQA
python -u src/main.py \
        --name=nodefence_TrojanVQA_ViT_L14_coco \
        --train_data .csv \
        --batch_size=128 \
        --lr=1e-6 \
        --epochs=10 \
        --num_warmup_steps=10000 \
        --complete_finetune \
        --pretrained \
        --image_key=image \
        --caption_key=caption \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/ \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=vqa \
        --patch_location middle \
        --patch_name=opti_patches/TrojanVQA_ViT_L14_coco_vqa.jpg \
        --patch_size=16 \
        --model_name=ViT-L/14
### CleanClip Fine-tuning
python -u src/main.py \
        --name=cleanCLIP_TrojanVQA_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --inmodal \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=vqa \
        --patch_location middle \
        --patch_name=opti_patches/TrojanVQA_ViT_L14_coco_vqa.jpg \
        --patch_size=16 \
        --checkpoint .pt \
        --model_name=ViT-L/14
### Fine-tuning
python -u src/main.py \
        --name=SFT_TrojanVQA_ViT_L14_coco \
        --train_data=$data_path/finetune_200k.csv \
        --batch_size=64 \
        --num_warmup_steps=50 \
        --lr=45e-7 \
        --epochs=10 \
        --complete_finetune \
        --eval_data_type=ImageNet1K \
        --eval_test_data_dir=data/ImageNet1K/validation/  \
        --add_backdoor \
        --asr \
        --label banana \
        --patch_type=vqa \
        --patch_location middle \
        --patch_name=opti_patches/TrojanVQA_ViT_L14_coco_vqa.jpg \
        --patch_size=16 \
        --checkpoint .pt \
        --model_name=ViT-L/14

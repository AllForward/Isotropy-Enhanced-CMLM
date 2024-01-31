# Isotropy-Enhanced Conditional Masked Language Models

⭐ Isotropy-Enhanced Conditional Masked Language Models. EMNLP 2023 Findings. [paper]([https://github.com/pytorch/fairseq](https://aclanthology.org/2023.findings-emnlp.555.pdf)).

Some codes are borrowed from [Fairseq](https://github.com/pytorch/fairseq).

### Requirements

* Python >= 3.7
* Pytorch == 1.10
* Fairseq 0.10.2

To install fairseq from source and develop locally:

```shell
cd fairseq
pip install --editable ./
```



### Preparation
Taken WMT14 EN-DE as an example, please following the steps list below.

- Download and preprocess the data: `bash script/prepare-wmt14en2de.sh --icml17`.
- Binarize the training data.

```
input_dir=path_to_raw_text_data
data_dir=path_to_binarized_output
src=source_language
tgt=target_language
python3 fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir}/ \
    --workers 32 --src-dict ${input_dir}/dict.${src}.txt --tgt-dict {input_dir}/dict.${tgt}.txt
```



### Training

* Training CMLM with our method
```shell
checkpoint_path=path_to_your_checkpoint
data_dir=path_to_your_dataset
src=src_language
tgt=tgt_language

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
    ${data_dir} \
    -s ${src} \
    -t ${tgt} \
    --save-dir ${save_path} \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion nat_loss \
    --arch cmlm_transformer_cos \
    --noise random_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0007 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 40000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.2 --weight-decay 0.01 --fp16 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8192 --max-tokens-valid 1024 \
    --max-update 300000 \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --no-scale-embedding \
    --concatLast \
    --insertNeighborAttn \
    --update-freq 2 \
```



### Inference

We average the 5 best checkpoints chosen by validation BLEU scores as our final model for inference. The script for averaging checkpoints is scripts/average_checkpoints.py

```shell
checkpoint_path=path_to_your_checkpoint
data_dir=path_to_your_dataset
src=src_language
tgt=tgt_language
CUDA_VISIBLE_DEVICES=0 fairseq-generate > res.log \
    ${data_dir} -s ${src} -t ${tgt} \
    --path ${checkpoint_path} --iter-decode-force-max-iter \
    --task translation_lev --gen-subset test \
    --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 \
    --batch-size 10 --iter-decode-with-beam 5 \
    --remove-bpe
bash scripts/compound_split_bleu.sh res.log
```


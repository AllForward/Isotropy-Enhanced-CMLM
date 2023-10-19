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
    --lookAround \
    --insertCausalSelfAttn \
    --update-freq 4 \
    2>&1 &
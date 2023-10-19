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

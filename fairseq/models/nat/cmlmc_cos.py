# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

from fairseq.models import register_model, register_model_architecture
# from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel
from fairseq.models.nat.nonautoregressive_transformer_cos import NATransformerModel
from fairseq.utils import new_arange
import torch.nn.functional as F
import torch

def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

# the part for contrastive loss
def build_mask_matrix(seqlen, valid_len_list, targets, compute_mask_token=False, compute_mask_pos=False):
    res_list = []
    base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
    # base_mask = torch.ones(seqlen, seqlen)
    base_mask = base_mask.type(torch.FloatTensor)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = base_mask.clone()
        one_valid_len = valid_len_list[i]
        one_base_mask[:,one_valid_len:] = 0.
        one_base_mask[one_valid_len:, :] = 0.
        if compute_mask_token:
            if targets[i].ne(3).sum() > 0:
                mask_target = targets[i].eq(3).unsqueeze(-1).expand(seqlen, seqlen).float().type_as(one_base_mask)
                one_base_mask = torch.mul(one_base_mask, mask_target)
        if compute_mask_pos:
            one_base_mask = torch.mul(targets[i].eq(3).type_as(one_base_mask), one_base_mask)
        eos_bos_pos = (targets[i].eq(0) | targets[i].eq(2)).repeat(seqlen, 1).type_as(one_base_mask)
        one_base_mask = torch.mul(one_base_mask, 1-eos_bos_pos)
        # 只获取相邻位置的cos_sim
        one_base_mask = torch.tril(one_base_mask, 1)
        one_base_mask = torch.triu(one_base_mask, -1)
        res_list.append(one_base_mask)
    res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
    #print (res_mask)
    assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    return res_mask

def max_cos_sim(outputs, targets):
    bsz, seqlen = targets.size()
    # bsz, seqlen, dim = outputs.size()
    # targets = torch.zeros([bsz, seqlen]).type_as(targets)
    outputs = outputs.type(torch.float32)
    norm_rep = outputs / outputs.norm(dim=2, keepdim=True)
    score_matrix = torch.matmul(norm_rep, norm_rep.transpose(1,2))
    assert score_matrix.size() == torch.Size([bsz, seqlen, seqlen])

    ### input mask
    input_mask = torch.ones_like(targets).type(torch.FloatTensor).type_as(targets)
    
    input_mask = input_mask.masked_fill(targets.eq(1), 0.0)
    
    valid_len_list = torch.sum(input_mask, dim = -1).tolist()
    cos_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], targets).type_as(input_mask)
    input_mask = input_mask.unsqueeze(-1).repeat(1, 1, input_mask.size()[-1])
    cos_matrix = score_matrix * cos_mask * input_mask
    return cos_matrix

@register_model("cmlmc_transformer_cos")
class CMLMCNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, tgt_tokens
        )

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
        )
        word_ins_mask = prev_output_tokens.eq(self.unk)

        ######################################## CMLMC Modifications ####################################################

        valid_token_mask = (prev_output_tokens.ne(self.pad) &
                            prev_output_tokens.ne(self.bos) &
                            prev_output_tokens.ne(self.eos))
        revealed_token_mask = (prev_output_tokens.ne(self.pad) &
                                prev_output_tokens.ne(self.bos) &
                                prev_output_tokens.ne(self.eos) &
                                prev_output_tokens.ne(self.unk))
        masked_input_out = self.decoder(normalize=False,
                                        prev_output_tokens=tgt_tokens.masked_fill(valid_token_mask, self.unk),
                                        encoder_out=encoder_out)

        revealed_length = revealed_token_mask.sum(-1).float()
        replace_length = revealed_length * 0.3

        masked_input_out_scores, masked_input_out_tokens = F.log_softmax(masked_input_out[0], -1).max(-1)
        masked_input_out_scores.uniform_()
        masked_input_out_scores.masked_fill_(~revealed_token_mask, 2.0)
        _, replace_rank = masked_input_out_scores.sort(-1)
        replace_token_cutoff = new_arange(replace_rank) < replace_length[:, None].long()
        replace_token_mask = replace_token_cutoff.scatter(1, replace_rank, replace_token_cutoff)
        replaced_input_tokens = prev_output_tokens.clone()
        replaced_input_tokens[replace_token_mask] = masked_input_out_tokens[replace_token_mask]

        replace_input_out = self.decoder(normalize=False,
                                            prev_output_tokens=replaced_input_tokens,
                                            encoder_out=encoder_out)

        # with torch.no_grad():
        # word_ins_target_out = self.decoder(
        #     normalize=False,
        #     prev_output_tokens=tgt_tokens.clone(),
        #     encoder_out=encoder_out,
        # )
        
        return {
            "word_ins": {
                "out": word_ins_out[0],
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 0.5,
            },
            "word_ins_corr": {
                "out": replace_input_out[0],
                "tgt": tgt_tokens,
                "mask": replace_token_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
                "factor": 0.5,
            },
            "cos_sim_self_out": {
                "out": [word_ins_out[0]],
                "tgt": prev_output_tokens,
            },
            "cos_sim_self_hid": {
                "out": [word_ins_out[1]],
                "tgt": prev_output_tokens,
            },
            # "cos_sim_tacl": {
            #     "out": word_ins_out[0],
            #     # "tgt": second_word_ins_out[1],
            #     # "prev": prev_output_tokens,
            #     "tgt": word_ins_target_out[0],
            #     "mask": word_ins_mask,
            #     # "factor": 0.5
            # },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        # output_masks = output_tokens.eq(self.unk)
        output_masks = output_tokens.ne(self.pad) & output_tokens.ne(self.bos) & output_tokens.ne(self.eos)

        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step
        )
        _scores, _tokens = F.log_softmax(word_ins_out[0], -1).max(-1)
        # out_layer_cos_matrix = max_cos_sim(word_ins_out[1], output_tokens)
        # import pdb
        # pdb.set_trace()
        # _scores = _scores * (1- out_layer_cos_matrix.sum(-1) / 2)
        # print("out_layer_cos_matrix: {}".format(out_layer_cos_matrix))
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


@register_model_architecture("cmlmc_transformer_cos", "cmlmc_transformer_cos")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("cmlmc_transformer_cos", "cmlmc_transformer_cos_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)

@register_model_architecture(
    "cmlmc_transformer_cos", "cmlmc_transformer_cos_iwslt"
)
def cmlm_iwslt_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_embed_dim * 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    cmlm_base_architecture(args)

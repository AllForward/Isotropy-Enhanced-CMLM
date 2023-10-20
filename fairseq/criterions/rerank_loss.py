import math
from operator import index
from pyexpat import features
from typing_extensions import Self
import torch
from copy import deepcopy
from fairseq.utils import new_arange
from . import FairseqCriterion, register_criterion
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor
import random
from contextlib import contextmanager

def merge_dict(origin, new, prefix):
    for key, value in new.items():
        origin[prefix + key] = value
    return origin


def get_loss(loss, sample_size):
    return loss / sample_size / math.log(2) if sample_size > 0 else 0.0

sampler = torch.distributions.Uniform(0, 1)
def lucky(prob):
    return sampler.sample().item() < prob

#nat_loss
def _compute_loss(outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
    """
	outputs: batch x len x d_model
	targets: batch x len
	masks:   batch x len

	policy_logprob: if there is some policy
	    depends on the likelihood score as rewards.
    """

    def mean_ds(x: Tensor, dim=None) -> Tensor:
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )
    if masks is not None:
        outputs, targets = outputs[masks], targets[masks]

    if masks is not None and not masks.any():
        nll_loss = torch.tensor(0)
        loss = nll_loss
    else:
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
            losses = losses.sum(-1)

        nll_loss = mean_ds(losses)
        if label_smoothing > 0:
            loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
        else:
            loss = nll_loss

    loss = loss * factor
    return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

def _compute_nce_loss(features, energy, noise_index, score_tgt_tokens, score_prev_output_tokens):
    k = noise_index.sum(1)
    k = k.repeat(noise_index.size(1),1).t()
    n = score_prev_output_tokens.ne(1).sum(1) - noise_index.sum(1)
    n = n.repeat(noise_index.size(1),1).t()
    token_index = score_tgt_tokens.ne(1) & ~noise_index
    q_noise_logits = F.softmax(features, dim=-1)
    q_nll_loss = q_noise_logits.gather(dim=-1, index=score_prev_output_tokens.unsqueeze(-1)).squeeze(-1)
    # a = torch.mul(n[noise_index].squeeze(-1),energy[noise_index])
    # b = torch.mul(k[noise_index].squeeze(-1), q_nll_loss[noise_index])
    loss_noise = -torch.log(torch.div(torch.mul(k[noise_index].squeeze(-1), q_nll_loss[noise_index]),(torch.mul(n[noise_index].squeeze(-1),energy[noise_index])+ torch.mul(k[noise_index].squeeze(-1), q_nll_loss[noise_index]))))
    loss_token = -torch.log(torch.div(torch.mul(n[token_index].squeeze(-1), energy[token_index]),(torch.mul(n[token_index].squeeze(-1),energy[token_index]) + torch.mul(k[token_index].squeeze(-1),q_nll_loss[token_index]))))
    # a = loss_noise.sum()
    # b = torch.mul(n[token_index].squeeze(-1), energy[token_index])
    # c = torch.mul(k[token_index].squeeze(-1),q_nll_loss[token_index])
    # loss_noise = k[noise_index].squeeze(-1) * q_nll_loss[noise_index] / (n[noise_index].squeeze(-1) * p_nll_loss[noise_index] + k[noise_index].squeeze(-1) * q_nll_loss(noise_index))
    # loss_token = n[token_index].squeeze(-1) * q_nll_loss[token_index] / (n[token_index].squeeze(-1) * p_nll_loss[noise_index] + k[token_index].squeeze(-1) * q_nll_loss[token_index])
    # c = loss_noise.mean()
    # d = loss_token.mean()
    return (loss_noise.mean() + loss_token.mean())

@register_criterion('ranker_loss')
class ScoreLabelSmoothedMutiMixedCrossEntropyCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--label-smoothing',
            default=0.1,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    # def compute_kl_loss(self, net_output, siamese_net_output, pad_mask=None, reduce=True):
    #     p = F.log_softmax(net_output, dim=-1)
    #     p_tec = F.softmax(net_output, dim=-1)
    #     q = F.log_softmax(siamese_net_output, dim=-1)
    #     q_tec = F.softmax(siamese_net_output, dim=-1)

    #     p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
    #     q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        
    #     if pad_mask is not None:
    #         p_loss.masked_fill_(pad_mask, 0.)
    #         q_loss.masked_fill_(pad_mask, 0.)

    #     if reduce:
    #         p_loss = p_loss.sum()
    #         q_loss = q_loss.sum()

    #     loss = (p_loss + q_loss) / 2
    #     nomasktokens=(pad_mask==True).nonzero().size(0)
    #     loss=loss/nomasktokens
    #     return loss
        
    def forward(self, model, score_sample, cmlm_sample,reduce=True):
        score_model = model.score_model
        # cmlm_model = model.cmlm_model
        disco_model = model.disco_model
        logging_output = dict()
        ntokens = cmlm_sample['ntokens']
        nsentences = cmlm_sample['nsentences']
        sample_size = nsentences
        logging_output['ntokens'] = ntokens
        logging_output['nsentences'] = nsentences
        logging_output['sample_size'] = sample_size
        
        # cmlm
        # with torch.no_grad():
        #     cmlm_sample['net_input']['encoder_out'] = cmlm_model.encoder(cmlm_sample['net_input']['src_tokens'], src_lengths=cmlm_sample['net_input']['src_lengths'])
        #     cmlm_tgt_tokens, cmlm_prev_output_tokens = cmlm_sample["target"], cmlm_sample["prev_target"]
            
        #     cmlm_length_out = cmlm_model.decoder.forward_length(normalize=False, encoder_out=cmlm_sample['net_input']['encoder_out'])
        #     cmlm_length_tgt = cmlm_model.decoder.forward_length_prediction(cmlm_length_out, cmlm_sample['net_input']['encoder_out'], cmlm_tgt_tokens)

        #     cmlm_word_ins_out = cmlm_model.decoder(
        #         normalize=False,
        #         prev_output_tokens=cmlm_prev_output_tokens,
        #         encoder_out=cmlm_sample['net_input']['encoder_out'],
        #         )
        #     cmlm_word_ins_mask = cmlm_prev_output_tokens.eq(cmlm_model.unk)
        # outputs = {
        #     "word_ins": {
        #         "out": cmlm_word_ins_out, "tgt": cmlm_tgt_tokens,
        #         "mask": cmlm_word_ins_mask, "ls": self.label_smoothing,
        #         "nll_loss": True
        #     },
        #     "length": {
        #         "out": cmlm_length_out,
        #         "tgt": cmlm_length_tgt,
        #         "factor": cmlm_model.decoder.length_loss_factor,
        #     },
        # }
        losses, nll_losses = [], []
        # for obj in outputs:
        #     if outputs[obj].get("loss", None) is None:
        #         _losses = _compute_loss(
        #             outputs[obj].get("out"),
        #             outputs[obj].get("tgt"),
        #             outputs[obj].get("mask", None),
        #             outputs[obj].get("ls", 0.0),
        #             name=obj + '-loss',
        #             factor=outputs[obj].get("factor", 1.0)
        #         )
        #     else:
        #         assert False

        #     losses += [_losses]
        #     if outputs[obj].get("nll_loss", False):
        #         nll_losses += [_losses.get("nll_loss", 0.0)]

        # cmlm_loss = sum(l["loss"] for l in losses)
        # cmlm_nll_loss = sum(l for l in nll_losses)

        # # # NOTE:
        # # # we don't need to use sample_size as denominator for the gradient
        # # # here sample_size is just used for logging
        # cmlm_sample_size = 1
        # cmlm_logging_output = {
        #     "loss": utils.item(cmlm_loss.data) if reduce else cmlm_loss.data,
        #     "nll_loss": utils.item(cmlm_nll_loss.data) if reduce else cmlm_nll_loss.data,
        #     "ntokens": cmlm_sample["ntokens"],
        #     "nsentences": cmlm_sample["nsentences"],
        #     "sample_size": cmlm_sample_size,
        # }

        # for l in losses:
        #     cmlm_logging_output[l["name"]] = (
        #         utils.item(l["loss"].data / l["factor"])
        #         if reduce
        #         else l[["loss"]].data / l["factor"]
        #     )

        # choose K noise positions
        score_tgt_tokens = score_sample["target"]
        token_mask = score_tgt_tokens.ne(disco_model.pad) & score_tgt_tokens.ne(disco_model.eos) & score_tgt_tokens.ne(disco_model.bos)
        # get the length k with ceil
        # import pdb
        # pdb.set_trace()
        replace_length = (token_mask.sum(1) * 0.15).ceil()
        masked_input_out_scores = score_tgt_tokens.clone().float().uniform_()
        masked_input_out_scores.masked_fill_(~token_mask, 2.0)
        _, replace_rank = masked_input_out_scores.sort(-1)
        replace_token_cutoff = new_arange(replace_rank) < replace_length[:, None].long()
        replace_token_mask = replace_token_cutoff.scatter(1, replace_rank, replace_token_cutoff)
        # q_mask_tokens = score_tgt_tokens.clone().masked_fill_(replace_token_mask,cmlm_model.unk)
        
        #noise discribution 
        #-----------------------------------------------------------------------------------------------------------------------------
        # q_mask_tokens = score_tgt_tokens.clone().masked_fill_(token_mask,disco_model.unk) # xys: why masked unk? need be tgt tokens but attention mask only self.
        with torch.no_grad():
            cmlm_sample['net_input']['encoder_out'] = disco_model.encoder(cmlm_sample['net_input']['src_tokens'], src_lengths=cmlm_sample['net_input']['src_lengths'])
            q_word_ins_out = disco_model.decoder(
                normalize=True,
                # prev_output_tokens=q_mask_tokens,
                # 应该改为tgt_tokens?
                prev_output_tokens=cmlm_sample["target"],
                encoder_out=cmlm_sample['net_input']['encoder_out'],
                masking_type="self_masking",
                gen_order=None,
            )
        # q_word_ins_out = q_word_ins_out_temp.clone().detach()
        q_noise_tokens = q_word_ins_out.argmax(-1)
        score_prev_output_tokens = score_tgt_tokens.masked_fill(replace_token_mask,0) + q_noise_tokens.masked_fill_(~replace_token_mask,0)
        # _, score_tgt_features = score_model.decoder(
        #     normalize=False,
        #     prev_output_tokens=score_tgt_tokens,
        # )
        # with torch.no_grad():
            # score_encoder_out = score_model.encoder(score_sample['net_input']['src_tokens'], src_lengths=score_sample['net_input']['src_lengths'])
        score_encoder_out = score_model.encoder(score_sample['net_input']['src_tokens'], src_lengths=score_sample['net_input']['src_lengths'])
        # import pdb
        # pdb.set_trace()
        score_noise_features = score_model.decoder(
            normalize=False,
            prev_output_tokens=score_prev_output_tokens,
            encoder_out=score_encoder_out,
            # encoder_out=score_encoder_out,   # xys: if need encoder out ? 
        )
        energy = torch.exp(-score_noise_features)

        noise_index = replace_token_mask

        score_loss = _compute_nce_loss(q_word_ins_out, energy, noise_index, score_tgt_tokens, score_prev_output_tokens)

        score_sample_size = 1
        score_logging_output = {
            "loss": utils.item(score_loss.data) ,
            "sample_size": score_sample_size,
        }
        
        for l in losses:
            score_logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        logging_output = merge_dict(logging_output, score_logging_output, "score-")
        # logging_output = merge_dict(logging_output, cmlm_logging_output, "cmlm-")

        # kl_loss = self.compute_kl_loss(word_ins_out, cmlm_word_ins_out, (score_word_ins_mask & cmlm_word_ins_mask).unsqueeze(-1))
        # loss += 0.8 * kl_loss
        # loss = score_loss + cmlm_loss
        return score_loss, sample_size, logging_output
    

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        score_ntokens = sum(log.get("score-ntokens", 0) for log in logging_outputs)
        score_nsentences = sum(log.get("score-nsentences", 0) for log in logging_outputs)
        score_sample_size = sum(log.get("score-sample_size", 0) for log in logging_outputs)
        score_loss = sum(log.get("score-loss", 0) for log in logging_outputs)
        score_nll_loss = sum(log.get("score-nll_loss", 0) for log in logging_outputs)
        score_weights = [log.get("score-weight", 0) for log in logging_outputs]
        score_weight = max(score_weights)

        cmlm_ntokens = sum(log.get("cmlm-ntokens", 0) for log in logging_outputs)
        cmlm_nsentences = sum(log.get("cmlm-nsentences", 0) for log in logging_outputs)
        cmlm_sample_size = sum(log.get("cmlm-sample_size", 0) for log in logging_outputs)
        cmlm_loss = sum(log.get("cmlm-loss", 0) for log in logging_outputs)
        cmlm_nll_loss = sum(log.get("cmlm-nll_loss", 0) for log in logging_outputs)
        cmlm_weights = [log.get("cmlm-weight", 0) for log in logging_outputs]
        cmlm_weight = max(cmlm_weights)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        # loss = get_loss((score_weight * score_loss / score_sample_size * sample_size if score_sample_size > 0 else 0.0) + (cmlm_weight * cmlm_loss / cmlm_sample_size * sample_size if cmlm_sample_size > 0 else 0.0), sample_size)
        nll_loss = get_loss((score_weight * score_nll_loss / score_sample_size * sample_size if score_sample_size > 0 else 0.0) + (cmlm_weight * cmlm_nll_loss / cmlm_sample_size * sample_size if cmlm_sample_size > 0 else 0.0), sample_size)
        criteria = "nat"
        score_loss = score_loss
        cmlm_loss = get_loss(cmlm_loss, cmlm_sample_size)
        score_nll_loss = get_loss(score_nll_loss, score_sample_size)
        cmlm_nll_loss = get_loss(cmlm_nll_loss, cmlm_sample_size)
        return {
            "loss": score_loss,
            "nll_loss": cmlm_nll_loss,
            "score_loss": score_loss,
            "cmlm_loss": cmlm_loss,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "score_weight": score_weight,
            "cmlm_weight": cmlm_weight,
            "score_sample_size": score_sample_size,
            "cmlm_sample_size": cmlm_sample_size,
        }

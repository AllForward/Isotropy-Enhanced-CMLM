# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
import numpy


@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def compute_dif_hid_cos(self, masked_rep, truth_rep, contrastive_labels=None, name="loss", factor=1.0, temperature=0.1):
        masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
        truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
        contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1,2)) / temperature # bsz x seqlen x seqlen
        bsz, seqlen, _ = contrastive_scores.size()
        # 将句子中非mask的位置设置为-inf
        # mask_tokens = contrastive_labels[0].unsqueeze(1).repeat(1, seqlen, 1)
        # contrastive_scores = contrastive_scores.masked_fill(~mask_tokens, -math.inf)

        logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
        # 将-inf设置为0
        # logprobs = logprobs.masked_fill(logprobs.eq(-math.inf), 0.0)

        gold = torch.arange(seqlen).view(-1,)
        gold = gold.expand(bsz, seqlen).contiguous().view(-1)
        if contrastive_scores.is_cuda:
            gold = gold.cuda(contrastive_scores.get_device())
        # 取对角线元素值
        loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
        loss = loss.view(bsz, seqlen) * contrastive_labels
        loss = torch.sum(loss) / contrastive_labels.sum()
        return {"name": name, "loss": loss, "factor": factor}

    # the part for contrastive loss
    def build_mask_matrix(self, seqlen, valid_len_list, targets, compute_mask_pos=True):
        '''
            (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
                then the loss padding matrix looks like
                     [0., 1., 1., 1.],
                     [1., 0., 1., 1.],
                     [1., 1., 0., 1.],
                     [1., 1., 1., 0.]
            (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
                then the loss padding matrix looks like
                     [0., 1., 1., 0.],
                     [1., 0., 1., 0.],
                     [1., 1., 0., 0.],
                     [0., 0., 0., 0.]
        '''
        res_list = []
        base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
        base_mask = base_mask.type(torch.FloatTensor).to(targets.device)
        bsz = len(valid_len_list)
        for i in range(bsz):
            one_base_mask = base_mask.clone()
            one_valid_len = valid_len_list[i]
            one_base_mask[:,one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
            # if targets[i].ne(3).sum() > 0:
            #     mask_target = targets[i].ne(3).unsqueeze(-1).expand(seqlen, seqlen).float().type_as(one_base_mask)
            #     one_base_mask = torch.mul(one_base_mask, mask_target)
            # if compute_mask_pos:
            #     one_base_mask = torch.mul(targets[i].eq(3).type_as(one_base_mask), one_base_mask)
            res_list.append(one_base_mask)
        res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
        #print (res_mask)
        assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
        return res_mask

    # def build_margin_matrix(self, seqlen, targets, margin):
    #     base_mask = torch.ones(seqlen, seqlen)
    #     base_mask = base_mask.type(torch.FloatTensor)
    #     bsz = targets.size()[0]
    #     step_dis = (1.0 - margin) / (seqlen - 1)
    #     for j in range(seqlen):
    #         base_mask[j, j] = margin
    #         for k in range(seqlen):
    #             base_mask[j, k] = margin + step_dis * numpy.abs(k - j)
    #     res_mask = base_mask.unsqueeze(0).repeat(bsz, 1, 1)
    #     #print (res_mask)
    #     assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    #     return res_mask

    def _compute_cos_loss(self, outputs, targets, masks=None, name="loss", factor=1.0, margin=0.4, temperature=0.1):
        bsz, seqlen = targets.size()
        loss = None
        ### input mask
        input_mask = torch.ones_like(targets).type(torch.FloatTensor).type_as(targets)
        # import pdb
        # pdb.set_trace()
        input_mask = input_mask.masked_fill(targets.eq(self.padding_idx), 0.0)
        valid_len_list = torch.sum(input_mask, dim = -1).tolist()
        loss_mask = self.build_mask_matrix(seqlen, [int(item) for item in valid_len_list], targets).type_as(input_mask)
        for output in outputs:
            # output = output.transpose(0, 1)
            output = output.type(torch.float32)
            norm_rep = output / output.norm(dim=2, keepdim=True)
            score_matrix = torch.matmul(norm_rep, norm_rep.transpose(1,2))
            assert score_matrix.size() == torch.Size([bsz, seqlen, seqlen])

            gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
            gold_score = torch.unsqueeze(gold_score, -1)
            assert gold_score.size() == torch.Size([bsz, seqlen, 1])
            difference_matrix = gold_score - score_matrix
            assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
            
            # loss_matrix = (margin - difference_matrix) / temperature # bsz x seqlen x seqlen
            loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
            # import pdb
            # pdb.set_trace()
            # margin_matrix = self.build_margin_matrix(seqlen, targets, margin).type_as(score_matrix)
            # loss_matrix = margin_matrix - difference_matrix
            
            loss_matrix = torch.nn.functional.relu(loss_matrix)

            ### input mask
            # input_mask = torch.ones_like(targets).type(torch.FloatTensor).type_as(targets)
            
            # input_mask = input_mask.masked_fill(targets.eq(self.padding_idx), 0.0)

            # valid_len_list = torch.sum(input_mask, dim = -1).tolist()
            # loss_mask = self.build_mask_matrix(seqlen, [int(item) for item in valid_len_list], targets).type_as(input_mask)
            masked_loss_matrix = loss_matrix * loss_mask

            loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
            assert loss_matrix.size() == targets.size()
            loss_matrix = loss_matrix * input_mask
            # loss = torch.sum(loss_matrix) / (loss_mask.sum(-1) * input_mask).sum() * factor
            if loss is None:
                loss = torch.sum(loss_matrix) / (loss_mask.sum(-1) * input_mask).sum() * factor
                # loss = torch.sum(loss_matrix) / (targets.eq(3)).sum() * factor
            else:
                loss += torch.sum(loss_matrix) / (loss_mask.sum(-1) * input_mask).sum() * factor
                # loss += torch.sum(loss_matrix) / (targets.eq(3)).sum() * factor
        # nll_loss = cl_loss
        return {"name": name, "loss": loss, "factor": factor}

    def _compute_related_cos_sim(self, output, targets, prev_output_tokens, model=None, name="loss", factor=1.0, margin=0.8):
        # import pdb
        # pdb.set_trace()
        bsz, seqlen = prev_output_tokens.size()
        # cl_loss = None
        # for output in outputs:
        # output = output.transpose(0, 1)
        output = output.type(torch.float32)
        norm_rep = output / output.norm(dim=2, keepdim=True)
        import pdb
        pdb.set_trace()
        score_matrix = torch.matmul(norm_rep, norm_rep.transpose(1,2))
        assert score_matrix.size() == torch.Size([bsz, seqlen, seqlen])

        # with torch.no_grad():
        #     target_embed = model.decoder.embed_tokens(targets)
        #     target_embed = target_embed.type(torch.float32)
        #     target_norm_rep = target_embed / target_embed.norm(dim=2, keepdim=True)
        #     target_matrix = torch.matmul(target_norm_rep, target_norm_rep.transpose(1,2))
        target_embed = targets
        target_embed = target_embed.type(torch.float32)
        target_norm_rep = target_embed / target_embed.norm(dim=2, keepdim=True)
        target_matrix = torch.matmul(target_norm_rep, target_norm_rep.transpose(1,2))
        
        # loss_matrix = torch.mul(1 - target_matrix, score_matrix) # bsz x seqlen x seqlen
        # loss_matrix = difference_matrix - margin # bsz x seqlen x seqlen
        loss_matrix = score_matrix - target_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        ### input mask
        input_mask = torch.ones_like(prev_output_tokens).type(torch.FloatTensor).type_as(prev_output_tokens)
        
        input_mask = input_mask.masked_fill(prev_output_tokens.eq(self.padding_idx), 0.0)

        valid_len_list = torch.sum(input_mask, dim = -1).tolist()
        # import pdb
        # pdb.set_trace()
        loss_mask = self.build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prev_output_tokens).type_as(input_mask)

        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
        assert loss_matrix.size() == prev_output_tokens.size()
        loss_matrix = loss_matrix * input_mask
        loss = torch.sum(loss_matrix) / (loss_mask.sum(-1) * input_mask).sum() * factor
        # if cl_loss is None:
        #     cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask) * factor
        # else:
        #     cl_loss += torch.sum(loss_matrix) / torch.sum(loss_mask) * factor
        # nll_loss = cl_loss
        return {"name": name, "loss": loss, "factor": factor}

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
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
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        nll_loss = nll_loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []

        for obj in outputs:
            if obj.startswith("cos_sim_tacl"):
                _losses = self.compute_dif_hid_cos(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif obj.startswith("cos_sim_self"):
                # _losses = self.compute_dif_hid_cos(
                #     outputs[obj].get("out"),
                #     outputs[obj].get("tgt"),
                #     outputs[obj].get("mask"),
                #     name=obj + "-loss",
                #     factor=outputs[obj].get("factor", 1.0),
                # )
                _losses = self._compute_cos_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif obj.startswith("cos_sim_related"):
                _losses = self._compute_related_cos_sim(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("prev"),
                    model=None,
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            elif outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

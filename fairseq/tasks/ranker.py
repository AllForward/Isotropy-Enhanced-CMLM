# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq.utils import new_arange
from fairseq import utils
import logging

logger = logging.getLogger(__name__)

@register_task('ranker')
class TranslationScorecmlmTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--generator',
            default="none",
            choices=["cmlm", "disco", "score","mix" ,"none", "ori"]
        )
        parser.add_argument(
            '--mode-switch-updates', default=0, type=int,
            help='after how many steps to switch score/cmlm criterion, 0 for no switches'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = self.cfg.generator

    def build_generator(self, models, args, **unused):
        if self.generator == "cmlm":
            from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
            return IterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 9),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', True),
                retain_history=getattr(args, 'retain_iter_history', False))
        elif self.generator == "ori":
            from fairseq.iterative_refinement_generator_ori import IterativeRefinementGeneratorOri
            return IterativeRefinementGeneratorOri(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 9),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', True),
                retain_history=getattr(args, 'retain_iter_history', False))
        else:
            from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
            return IterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 0),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
                retain_history=getattr(args, 'retain_iter_history', False))
        # else:
        #     return None
        #     raise NotImplementedError("Please specify generator type by using '--generator"
        #                               " option")
    def inference_step(self, generator, models, sample, prefix_tokens=None,constraints=None):
        with torch.no_grad():
            if self.cfg.generator == "cmlm":
                models = [m.cmlm_model for m in models]
            elif self.cfg.generator == "disco":
                models = [m.disco_model for m in models]
            elif self.cfg.generator == "score":
                models = [m.score_model for m in models]
            elif self.cfg.generator == "mix":
                # models = [models[0].cmlm_model,models[0].score_model]
                models = [models[0]]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)


    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.cfg.data.split(os.pathsep)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    def inject_noise(self, target_tokens,noise):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if noise == 'random_delete':
            return _random_delete(target_tokens)
        elif noise == 'random_mask':
            return _random_mask(target_tokens)
        elif noise == 'full_mask':
            return _full_mask(target_tokens)
        elif noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError



    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        cmlm_sample = deepcopy(sample)
        score_sample = sample
        # cmlm_sample["prev_target"] = self.inject_noise(cmlm_sample["target"],noise="random_mask")
        # 针对disco
        cmlm_sample["prev_target"] = self.inject_noise(cmlm_sample["target"], noise="no_noise")
        loss, sample_size, logging_output = criterion(model, score_sample, cmlm_sample)
        model.disco_model.eval()
        # model.score_model.encoder.eval()
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
    

    # def valid_step(self, sample, model, criterion):
    #     model.eval()
    #     with torch.no_grad():
    #         cmlm_sample = deepcopy(sample)
    #         cmlm_sample["prev_target"] = self.inject_noise(cmlm_sample["target"],noise="random_mask")
    #         score_sample = sample
    #         loss, sample_size, logging_output = criterion(model, score_sample, cmlm_sample)
    #     return loss, sample_size, logging_output


    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            cmlm_sample = deepcopy(sample)
            score_sample = sample
            score_sample["score"] = {"context_p": 0.5}
            # score_sample['prev_target'] = self.inject_noise(score_sample['target'],noise="full_mask")
            # cmlm_sample['prev_target'] = self.inject_noise(cmlm_sample['target'],noise="random_mask")
            cmlm_sample['prev_target'] = self.inject_noise(cmlm_sample['target'],noise="no_noise")
            # cmlm_sample["weight"] = 0.5
            loss, sample_size, logging_output = criterion(model, score_sample, cmlm_sample)
            EVAL_BLEU_ORDER = 4
            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, score_sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        # import pdb
        # pdb.set_trace()
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union, List

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput
    from ...hparams import FinetuningArguments

logger = logging.get_logger(__name__)

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE,
       and to auto-save predictions at each evaluation step.
    """

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)

        if processor is not None:
            self.model_accepts_loss_kwargs = False
            self.add_callback(SaveProcessorCallback(processor))

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )

        if generated_tokens is not None and self.args.predict_with_generate:
            pad_id = self.processing_class.pad_token_id
            generated_tokens[:, : inputs["input_ids"].size(-1)] = pad_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    @override
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,                      # accept any generate kwargs here
    ):
        # 1) Standard evaluation → metrics (uses generate() if predict_with_generate=True)
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # 2) Fetch & save raw predictions, merging default + override gen_kwargs
        if self.args.predict_with_generate:
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            step = int(self.state.global_step)

            # merge trainer._gen_kwargs with passed-in gen_kwargs,
            # with explicit gen_kwargs taking precedence
            final_gen_kwargs = {**getattr(self, "_gen_kwargs", {}), **gen_kwargs}

            pred_out = self.predict(
                ds,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"step_{step}_predict",
                **final_gen_kwargs,
            )
            self.save_predictions(
                dataset=ds,
                predict_results=pred_out,
                skip_special_tokens=True,
                output_path=self.args.output_dir + '/generation'
            )

        return metrics

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
        output_path: Optional[str] = None,
    ) -> None:
        if not self.is_world_process_zero():
            return

        step = int(self.state.global_step)

        # Build output file path
        if output_path:
            if os.path.isdir(output_path):
                output_file = os.path.join(output_path, f"predictions_step_{step}.jsonl")
            else:
                output_file = output_path
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = os.path.join(self.args.output_dir, f"predictions_step_{step}.jsonl")

        logger.info_rank0(f"Saving predictions to {output_file}")

        pad_id = self.processing_class.pad_token_id

        # Replace IGNORE_INDEX with pad_id
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, pad_id)
        preds =  np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, pad_id)

        # Rotate each prediction sequence so padding is at the end
        for i in range(len(preds)):
            nz = np.nonzero(preds[i] != pad_id)[0]
            if nz.size > 0:
                preds[i] = np.concatenate((preds[i][nz[0]:], preds[i][:nz[0]]), axis=-1)

        # Decode inputs, preds, labels
        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds  = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        # Write out JSONL
        with open(output_file, "w", encoding="utf-8") as writer:
            for src, pr, lb in zip(decoded_inputs, decoded_preds, decoded_labels):
                writer.write(json.dumps({
                    "prompt": src,
                    f"predict-{step}": pr,
                    "label": lb
                }, ensure_ascii=False) + "\n")

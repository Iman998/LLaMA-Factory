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
import pandas as pd  
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

import datasets
from .khayyam_utils import (
    khayyam_choice_metrics,
    build_prompt,
    PROMPT_TEMPLATE,
)

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
        do_khayyam: bool = False,
        khayyam_csv: str = "/data/Iman/evaluation/khayyam_challenge/khayyam_challenge_sample.csv",
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: "PreTrainedTokenizer" = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self.do_khayyam = do_khayyam
        self._khayyam_ds = None
        if self.do_khayyam:
            self._khayyam_ds = self._prepare_khayyam_dataset(khayyam_csv)

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

    def _prepare_khayyam_dataset(self, csv_path: str):
        tok      = self.processing_class          # same tokenizer you pass to Trainer
        max_len  = getattr(self.args, "max_length", 2048)
        df       = pd.read_csv(csv_path)

        class KhayyamTorchDataset(torch.utils.data.Dataset):
            def __init__(self, frame):
                self.df      = frame.reset_index(drop=True)
                self.prompts = []          # raw prompt strings (for saving later)
                self.ids     = []          # list[int]  token ids
                self.masks   = []          # list[int]  attn masks
                self.labels_txt = []       # **ground-truth answer text**
                self._levels   = self.df.get("Level", []).tolist()
                self._cats_fa  = self.df.get("final_category_fa", []).tolist()

                for _, row in self.df.iterrows():
                    # ---------------- build the user question + choices --------
                    q_line = f"Question: {row['Question Body']}"
                    choice_lines = [
                        f"{i}) {row[f'Choice {i}']}" for i in range(1, 5)
                    ]
                    user_msg = q_line + "\n" + "\n".join(choice_lines)

                    # ---------------- feed messages to chat-template -----------
                    messages = [
                        {"role": "system", "content": PROMPT_TEMPLATE},
                        {"role": "user",   "content": user_msg},
                    ]
                    prompt = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,   # makes the template end with <start_of_turn>model\n
                    )

                    enc = tok(
                        prompt,
                        add_special_tokens=False,     # chat template already has BOS & special tags
                        truncation=True,
                        max_length=max_len,
                    )

                    self.prompts.append(prompt)
                    self.ids.append(enc["input_ids"])
                    self.masks.append(enc["attention_mask"])

                    # ------------ ground-truth label text for JSONL -----------
                    k = int(row["Key"])
                    self.labels_txt.append(k)

            # ------------- PyTorch Dataset interface --------------------------
            def __len__(self) -> int: return len(self.ids)

            def __getitem__(self, idx):
                ids = self.ids[idx]
                return {
                    "input_ids":      ids,                     # list[int]
                    "attention_mask": self.masks[idx],         # list[int]
                    "labels":         [IGNORE_INDEX] * len(ids)  # dummy, evaluation-only
                }

            # ------------- Helpers for metrics & saving -----------------------
            @property
            def keys_true(self): return self.df["Key"].tolist()

            @property
            def levels(self):           
                return self._levels
            
            @property
            def cats_fa(self):          
                return self._cats_fa
            
        return KhayyamTorchDataset(df)

    
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
        **gen_kwargs,
    ):
        # ---- 1) Standard HF evaluation ------------------------------------
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # ---- 2) Save normal eval generations ------------------------------
        if self.args.predict_with_generate:
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            gen_cfg = {**getattr(self, "_gen_kwargs", {}), **gen_kwargs}
            pred_out = self.predict(
                ds,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_choice",
                **gen_cfg,
            )
            self.save_predictions(
                dataset=ds,
                predict_results=pred_out,
                skip_special_tokens=True,
                suffix=f"generation-{self.state.global_step}",
            )

        # ---- 3) Extra Khayyam evaluation (if requested) -------------------
        if self.do_khayyam and self.args.predict_with_generate:
            kh_metrics = self._eval_khayyam(ignore_keys, gen_kwargs)
            metrics.update(kh_metrics)

        self.log(metrics)
        return metrics

    def save_predictions(
        self,
        dataset,
        predict_results,
        skip_special_tokens=True,
        output_path: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        if not self.is_world_process_zero():
            return

        step = int(self.state.global_step)
        pad  = self.processing_class.pad_token_id

        base = f"predictions_step_{step}"
        if suffix:
            base = f"{base}_{suffix}"
        if output_path is None:
            output_file = os.path.join(self.args.output_dir, f"{base}.jsonl")
        else:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{base}.jsonl")

        # ---------- decode predictions ---------------------------------------
        preds = np.where(predict_results.predictions != IGNORE_INDEX,
                        predict_results.predictions, pad)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=True)

        # ---------- recover original prompts ---------------------------------
        if hasattr(dataset, "prompts"):            # custom Khayyam dataset
            decoded_inputs = dataset.prompts
        else:                                      # HF Dataset
            decoded_inputs = self.processing_class.batch_decode(
                dataset["input_ids"], skip_special_tokens=False
            )

        # ---------- labels: try label_ids, else fall back to dataset.labels_txt
        if predict_results.label_ids is not None and np.any(
            predict_results.label_ids != IGNORE_INDEX
        ):
            labels_arr = np.where(
                predict_results.label_ids != IGNORE_INDEX,
                predict_results.label_ids,
                pad,
            )
            decoded_labels = self.processing_class.batch_decode(
                labels_arr, skip_special_tokens=skip_special_tokens
            )
        elif hasattr(dataset, "labels_txt"):
            decoded_labels = dataset.labels_txt
        else:
            decoded_labels = [""] * len(decoded_inputs)

        # ---------- write JSONL ----------------------------------------------
        with open(output_file, "w", encoding="utf-8") as w:
            for idx, (src, pr, lb) in enumerate(zip(decoded_inputs,
                                                    decoded_preds,
                                                    decoded_labels)):
                extra = ""
                if hasattr(dataset, "levels") and hasattr(dataset, "cats_fa"):
                    extra = f',"level":{dataset.levels[idx]!r},"category_fa":{dataset.cats_fa[idx]!r}'
                w.write(
                    f'{{"prompt":{src!r},"predict-{step}":{pr!r},"label":{lb!r}{extra}}}\n'
                )

    def _eval_khayyam(self, ignore_keys, gen_kwargs):
        step    = int(self.state.global_step)
        gen_cfg = {**getattr(self, "_gen_kwargs", {}), **gen_kwargs}

        pred_out = self.predict(
            self._khayyam_ds,
            ignore_keys=ignore_keys,
            metric_key_prefix=f"khayyam_step_{step}",
            **gen_cfg,
        )

        # ---------- SAFE DECODE (fixes OverflowError) -----------------------
        pad_id  = self.processing_class.pad_token_id
        pred_arr = np.asarray(pred_out.predictions)
        pred_arr = np.where(pred_arr < 0, pad_id, pred_arr)
        preds_txt = self.processing_class.batch_decode(
            pred_arr.tolist(), skip_special_tokens=True
        )

        keys_true = self._khayyam_ds.keys_true
        levels    = self._khayyam_ds.levels

        metrics = khayyam_choice_metrics(
            preds_txt, keys_true, levels, prefix="khayyam"
        )

        self.save_predictions(
            dataset=self._khayyam_ds,
            predict_results=pred_out,
            skip_special_tokens=True,
            suffix=f"khayyam_generation-{step}",
        )
        return metrics

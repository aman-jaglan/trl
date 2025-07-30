import os
import abc
import torch
import accelerate
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
from transformers import PreTrainedTokenizer
# Prefer upstream TRL GRPOTrainer (with GSPO support); fallback to local implementation if not available
try:
    from trl.trainer.grpo_trainer import GRPOTrainer  # type: ignore
except ImportError:  # Fallback for environments without TRL installed
    from .grpo import GRPOTrainer  # noqa: F401
from .grpo_config import GRPOConfig
from .teacher_base import TeacherReward, TeacherTrainer
from .utils_trl_15 import prepare_deepspeed
from transformers import AutoModelForCausalLM


class TeacherGRPOTrainer(GRPOTrainer, TeacherTrainer):
    def __init__(
            self,
            *args,
            student_model=None,


            use_reference_teacher_model=False,
            student_model_init_kwargs=None,
            logging_prob=0.0,


            disable_student_offloading=False,
            **kwargs):

        # Initialize base GRPO trainer (upstream or local)
        GRPOTrainer.__init__(self, *args, **kwargs)

        # ------------------------------------------------------------------
        # Compatibility shims when using upstream TRL's GRPOTrainer.
        # Ensure attributes accessed later in this subclass exist.
        # ------------------------------------------------------------------

        # Store model_init_kwargs (upstream stores only in args)
        if not hasattr(self, "_stored_model_init_kwargs"):
            self._stored_model_init_kwargs = getattr(self.args, "model_init_kwargs", {})

        # Ensure offload_untrained_models flag exists
        if not hasattr(self, "offload_untrained_models"):
            self.offload_untrained_models = getattr(self.args, "offload_untrained_models", False)

        # Ensure generation temperature attribute exists for reward code.
        if not hasattr(self, "gen_temperature"):
            # Upstream trainer keeps temperature in args.temperature
            self.gen_temperature = getattr(self.args, "temperature", 1.0)

        # ------------------------------------------------------------------

        if student_model_init_kwargs is None:
            student_model_init_kwargs = self._stored_model_init_kwargs

        offload_student_model = self.offload_untrained_models and (
            not disable_student_offloading)
        if student_model is None:

            self.student_model = self.ref_model
        elif isinstance(student_model, str):
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_student_model)
            else:
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                if offload_student_model:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:

            raise NotImplementedError
            self.student_model = student_model

        if use_reference_teacher_model:
            teacher_model = self.ref_model
        else:
            teacher_model = self.model

        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,
            teacher_model=teacher_model,
            tokenizer=self.processing_class,
            reward_functions=self.reward_funcs,
            output_dir=self.args.output_dir,
            logging_prob=logging_prob,
        )

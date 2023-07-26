# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:finetuning                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
OUTPUT_DIR = os.path.abspath(os.path.join(
    __file__, os.pardir, os.pardir, os.pardir, "data", "finetuning", "peft_lora_output"))


def example_process(model_repo: str = "bigscience/bloom-7b1",
                    dataset_repo: str = "Abirate/english_quotes",
                    batch_size: int = 4,
                    output_dir: str = OUTPUT_DIR,
                    ) -> None:
    """
    Function, containing the example process (https://www.youtube.com/watch?v=Us5ZFp16PaU).
    :param model_repo: Huggingface model repo id.
    :param dataset_repo: Hugginface dataset repo id.
    :param batch_size: Batch size for training.
    :param output_dir: Output directory. Defaults to folder under standard data path.
    """
    # Loading model
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        load_in_8bit=True,
        device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_repo)

    # Freezing model weights
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Setting up LoRA Adapters
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    config = LoraConfig(
        r=16,  # attention heads
        lora_alpha=32,  # alpha scaling
        # target_modules=["q_proj", "v_proj"], #if you know the
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    # Loading Data
    data = load_dataset()

    def merge_columns(example):
        example["prediction"] = example["quote"] + \
            " ->: " + str(example["tags"])
        return example

    data['train'] = data['train'].map(merge_columns)
    data = data.map(lambda samples: tokenizer(
        samples['prediction']), batched=True)

    # Training
    trainer = Trainer(
        model=model,
        train_dataset=data['train'],
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    example_process()

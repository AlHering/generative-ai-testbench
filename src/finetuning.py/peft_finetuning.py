# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:finetuning                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Optional, List, Any, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
import tqdm
from src.configuration import configuration as cfg



def example_process(model_repo: str = "smangrul/twitter_complaints_bigscience_T0_3B_LORA_SEQ_2_SEQ_LM",
                    dataset_repo: str = "twitter_complaints",
                    text_column: str = "Tweet text",
                    label_column: str = "text_label",
                    batch_size: int = 8,
                    output_dir: str = os.path.join(cfg.PATHS.DATA_PATH, "peft_training", "output"),
                    memory_distribution: dict = {0: "6GIB", 1: "0GIB", 2: "0GIB", 3: "0GIB", 4: "0GIB", "cpu": "18GB"}) -> None:
    """
    Function, containing the example process (https://www.leewayhertz.com/parameter-efficient-fine-tuning/).
    :param model_repo: Huggingface model repo id.
    :param dataset_repo: Hugginface dataset repo id.
    :param text_column: Text column (feature column) of the dataset.
    :param label_column: Label column of the dataset.
    :param batch_size: Batch size for training.
    :param output_dir: Output directory. Defaults to folder under standard data path.
    :param memory_distribution: Memory distribution for given devices.
    """
    model_config = PeftConfig.from_pretrained(model_repo)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_config.base_model_name_or_path, device_map="auto", max_memory=memory_distribution)
    model = PeftModel.from_pretrained(
        model, model_repo, device_map="auto", max_memory=memory_distribution)

    dataset = load_dataset(dataset_repo)
    classes = [k.replace("_", " ")
               for k in dataset["train"].features["Label"].names]
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model_name_or_path)

    max_label_length = max(
        [len(tokenizer(class_label)["input_ids"]) for class_label in classes])

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(inputs, truncation=True)
        labels = tokenizer(
            targets, max_length=max_label_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["eval"]
    test_dataset = processed_datasets["test"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    """train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)"""

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )
    trainer = Trainer(
        model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn
    )

    trainer.train()

    # Testing
    model.eval()
    i = 15
    inputs = tokenizer(
        f'{text_column} : {dataset["test"][i]["Tweet text"]} Label : ', return_tensors="pt")
    print(dataset["test"][i]["Tweet text"])
    print(inputs)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=10)
        print(outputs)
        print(tokenizer.batch_decode(
            outputs.detach().cpu().numpy(), skip_special_tokens=True))

if __name__ == "__main__":
    example_process()
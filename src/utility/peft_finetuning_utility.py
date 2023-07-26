# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from peft import PeftModel, PeftConfig

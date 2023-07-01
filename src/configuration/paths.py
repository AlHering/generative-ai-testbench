# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os


PACKAGE_PATH = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
SOURCE_PATH = os.path.join(PACKAGE_PATH, "src")
DOCS_PATH = os.path.join(PACKAGE_PATH, "docs")
SUBMODULE_PATH = os.path.join(SOURCE_PATH, "submodules")
DATA_PATH = os.path.join(PACKAGE_PATH, "data")
TEXTGENERATION_MODEL_PATH = os.path.join(
    PACKAGE_PATH, "machine_learning_models", "MODELS")
TEXTGENERATION_LORA_PATH = os.path.join(
    PACKAGE_PATH, "machine_learning_models", "LORAS")

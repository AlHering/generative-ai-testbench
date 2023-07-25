# -*- coding: utf-8 -*-
"""
****************************************************
*             generative_ai_testbench                
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from typing import Any
from src.configuration import configuration as cfg
from langchain.llms import LlamaCpp
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from src.organizer.organizer import Organizer, Librarian
from src.utility.langchain_utility import Settings
# Embedding Models: https://huggingface.co/spaces/mteb/leaderboard

llm = LlamaCpp(
    model_path=os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                            "orca_mini_7B-GGML/orca-mini-7b.ggmlv3.q4_1.bin"),
    verbose=True)


e5_large_v3_path = os.path.join(
    cfg.PATHS.TEXTGENERATION_MODEL_PATH, "intfloat_e5-large-v2")
tokenizer = AutoTokenizer.from_pretrained(e5_large_v3_path)
model = AutoModel.from_pretrained(e5_large_v3_path)


data_path = os.path.join(cfg.PATHS.DATA_PATH, "testbench3")
if not os.path.exists(data_path):
    os.makedirs(data_path)


class T5EmbeddingFunction(EmbeddingFunction):
    """
    EmbeddingFunction utilizing the "intfloat_e5-large-v2" model.
    (https://huggingface.co/intfloat/e5-large-v2)
    """

    def __call__(self, texts: Documents) -> Embeddings:
        """
        Method handling embedding.
        """
        # Taken from https://huggingface.co/intfloat/e5-large-v2 and adjusted
        batch_dict = tokenizer(texts, max_length=512,
                               padding=True, truncation=True, return_tensors='pt')

        # Tokenize the input texts
        batch_dict = tokenizer(texts, max_length=512,
                               padding=True, truncation=True, return_tensors='pt')

        outputs = model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state,
                                       batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T) * 100
        print(scores.tolist())

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Average pooling function, taken from https://huggingface.co/intfloat/e5-large-v2.
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


librarian = Librarian({
    "llm": llm,
    "chromadb_settings": Settings(persist_directory=os.path.join(data_path, "librarian_db")),
    "embedding_function": T5EmbeddingFunction,
    "retrieval_source_chunks": 1
})


"""organizer = Organizer({
    "llm": llm,
    "chromadb_settings": Settings(persist_directory=os.path.join(data_path, "organizer_db")),
    "embedding_function": T5EmbeddingFunction,
    "retrieval_source_chunks": 1
})
"""
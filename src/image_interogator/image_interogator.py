# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import torch
import os
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

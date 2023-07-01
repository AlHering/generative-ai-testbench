# -*- coding: utf-8 -*-
"""
****************************************************
*           generative_ai_testbench:librarian                 
*            (c) 2023 Alexander Hering             *
****************************************************
"""
# Taken from https://nayakpplaban.medium.com/ask-questions-to-your-images-using-langchain-and-python-1aeb30f38751 and ajusted
import torch
import os
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from langchain.agents import initialize_agent
from src.configuration import configuration as cfg
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.conversation.memory import ConversationBufferWindowMemory


class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        image = Image.open(img_path).convert("RGB")

        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"  # cuda

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(
            model_name).to(device)

        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class ObjectDetectionTool(BaseTool):
    name = "Object detector"
    description = "Use this tool when given the path to an image that you would like to detect objects. " \
                  "It will return a list of all detected objects. Each element in the list in the format: " \
                  "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path):
        image = Image.open(img_path).convert("RGB")

        processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let"s only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += "[{}, {}, {}, {}]".format(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += " {}".format(model.config.id2label[int(label)])
            detections += " {}\n".format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


def get_image_caption(image_path):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    image = Image.open(image_path).convert("RGB")

    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


def detect_objects(image_path):
    """
    Detects objects in the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string with all the detected objects. Each object as "[x1, x2, y1, y2, class_name, confindence_score]".
    """
    image = Image.open(image_path).convert("RGB")

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let"s only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += "[{}, {}, {}, {}]".format(
            int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += " {}".format(model.config.id2label[int(label)])
        detections += " {}\n".format(float(score))

    return detections


# initialize the agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,
    return_messages=True
)

# Setup prompting
promt_template = """
Question: {question}
Answer: Please devide your answer in multiple steps and explain each step to be sure we have the right answer.
"""
prompt = PromptTemplate(template=promt_template, input_variables=["question"])

# Instance of callback manager for token-wise streaming
callback_manager = CallbackManager(
    [StreamingStdOutCallbackHandler()])

# Setup central LLM
llm = LlamaCpp(
    model_path=os.path.join(cfg.PATHS.TEXTGENERATION_MODEL_PATH,
                            "orca_mini_7B-GGML/orca-mini-7b.ggmlv3.q4_1.bin"),
    n_ctx=1024,
    callback_manager=callback_manager,
    verbose=True)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method="generate"
)


def run_process():
    print("STARTING PROCESS")
    img_path = os.path.join(cfg.PATHS.DATA_PATH, "assets", "Parsons_PR.jpg")
    print("="*50)
    print("STEP 1")
    user_question = "generate a caption for this image?"
    response = agent.run(
        f"{user_question}, this is the image path: {img_path}")
    print(response)

    print("="*50)
    print("STEP 2")

    user_question = "Please tell me what are the items present in the image."
    response = agent.run(
        f'{user_question}, this is the image path: {img_path}')
    print(response)

    print("="*50)
    print("STEP 3")
    user_question = "Please tell me the bounding boxes of all detected objects in the image."
    response = agent.run(
        f'{user_question}, this is the image path: {img_path}')
    print(response)

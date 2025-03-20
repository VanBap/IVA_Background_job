import base64
import json
import os
from xmlrpc.client import boolean

import django
import logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel as BM, Field
import requests

from devices.models.rule import Rule
from devices.models.camera_alert import CameraAlert
from devices.services import camera_alert_service

from proc.scene_change_detector import snapshot_image_from_camera
from PIL import Image
from io import BytesIO

# === Test Response ===
import response_schema
from devices.models.test_image import TestImage

logger = logging.getLogger('app')

def resize_image_keep_ratio(image_path, new_size=224):
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        raise ValueError(f"Invalid image path: Expected a string, but got {type(image_path)}")

    image = image.convert("RGB")
    image = image.resize((new_size, new_size))
    # print(f"============== IMAGE:{image}")
    return image

def encode_image_pil(image, quality=50):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Image Base64
def encode_image(image_path, max_size=None):
    if max_size is not None:
        image = resize_image_keep_ratio(image_path, max_size)
        return encode_image_pil(image)
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

# === Test Output Schema ===
class OutputSchema(BM):
    final_answer: bool = Field(description="If the answer implies 'yes', then final_answer is true; otherwise, final_answer is false")
    explain: str = Field(description="Provide a logical and clear explanation that directly supports the final_answer. ")
# ================================

# === PydanticOutputParser =======
from langchain_core.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=OutputSchema)
# ================================

# ==== Reponse Schema ============
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

final_answer = ResponseSchema(name="final_answer",
                              description="If the answer implies 'yes', then final_answer is true; otherwise, final_answer is false"
                              )

explain = ResponseSchema(name="explain",
                         description="Provide a logical and clear explanation."
                         )
response_schemas = [final_answer, explain]

# VLM process
def process_vlm_rule(rule, camera):

    if not rule.prompt or not rule.vlm_model:
        print(f"=== Rule_id: {rule.id} has no VLM model / prompt")
        return "Rule không hợp lệ"

    print(f"==== [Processing] VLM model: {rule.vlm_model.code_name}")
    print(f"==== [Processing] Prompt: {rule.prompt.content}")

    # === Current Camera Image ===
    # img_path = snapshot_image_from_camera(camera.id, camera.url)
    # car_img_path = '/home/vbd-vanhk-l1-ubuntu/work/test_prompt_car.png'
    # whiteshirt_img_path = '/home/vbd-vanhk-l1-ubuntu/work/test_prompt_NO_whiteshirtblacktrouser.jpeg'

    check_id_test_img = 1
    check_image = ''
    test_images = TestImage.objects.all()
    for test_image in test_images:
        if test_image.id == check_id_test_img:
            check_image = test_image.url

    img_resized = encode_image(check_image, 224)
    print(f" len image: {len(img_resized)}")

    rule.prompt.system = (
        "You are an advanced visual analysis assistant. "
        "When answering questions about an image, focus on precise details and provide highly accurate responses. "
        "Carefully analyze the image before answering, ensuring that your response is based on clear visual evidence. "
        "For yes/no questions, respond strictly based on what is visible in the image, without assumptions. "
        "If the image is unclear or ambiguous, state that explicitly."
    )

    # === Khởi tạo model từ VLMModel
    llm = ChatOpenAI(model_name=rule.vlm_model.code_name,
                     api_key=rule.vlm_model.api_key,
                     base_url=rule.vlm_model.url,
                     temperature=0.0)

    # UPDATE
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # format_instructions = output_parser.get_format_instructions()
    # print(f"Format instructions: {format_instructions}")
    # formatted_prompt = f"{format_instructions}\n\nUser query:\n{rule.prompt.content}"

    prompt_C3 = (
            "Answer the question strictly in the following JSON format:\n"
            "{\n"
            '  "final_answer": <true/false>,\n'
            '  "explain": "<short explanation>"\n'
            "}\n"
            "Question: " + rule.prompt.content
    )

    messages = [
        SystemMessage(content=rule.prompt.system),
        HumanMessage(
            content=[
                {"type": "text", "text": prompt_C3},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_resized}"}},
            ]
        ),
    ]
    # print(f"Len messages: {len(messages)}")
    # print(f"Messages: {messages}")

    # === CACH 1+2+vannhk ===
    # structured_llm = llm.with_structured_output(response_schema.yes_or_no_schema)
    # response_data = structured_llm.invoke(messages)

    # === CACH 3 ===
    raw_response = llm.invoke(messages)
    response_text = raw_response.content
    response_data = json.loads(response_text)
    print(f"Response data: {response_data}")

    # Nếu model trả về "Yes", tạo Alert
    if response_data["final_answer"]:
        try:
            data = {
                'rule_id': rule.id,
                'rule_type': rule.type,
                'version_number': rule.current_version,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'details': {
                            "input": rule.prompt.content,
                            "output": response_data["final_answer"],
                            }
            }
            camera_alert_service.create_alert(data)
            print('Camera alert created')

        except AttributeError as e:
            print(f"Loi tao camera alert: {e}")

    else:
        print("Khong phat hien bat thuong. Khong tao Alert")

    print(" ========= Next Camera ========= ")

    return response_data














def process_rule_prompt_based():

    print('Loading list of rules')
    rules = Rule.objects.all()
    print('Process rules')

    for rule in rules:
        print(f"=== [Processing] Rule_type {rule.type}")
        print(f"=== [Processing] rule_id: {rule.id}")

        if rule.type != 1:
            print(f"rule {rule.id} has type not valid")

        else:
            print(f"Processing rule_id: {rule.id}")
            cameras = rule.cameras.all()
            for camera in cameras:
                print(f"=== Processing camera_id: {camera.id}")
                process_vlm_rule(rule, camera)

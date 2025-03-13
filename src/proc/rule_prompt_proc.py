import base64
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage
import requests

from devices.models.rule import Rule
from devices.models.camera_alert import CameraAlert
from devices.services import camera_alert_service

from proc.scene_change_detector import snapshot_image_from_camera
from PIL import Image
from io import BytesIO

def resize_image_keep_ratio(image_path, new_size=224):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize((new_size, new_size))
    return image

# Image Base64
def encode_image(image_path, max_size=None):
    if max_size is not None:
        image = resize_image_keep_ratio(image_path, max_size)
        return encode_image_pil(image)
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_pil(image, quality=50):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# ====== USE FOR GPT4o =====
def get_image_data_url(image_file: str, image_format: str) -> str:
    """
    Helper function to converts an image file to a data URL string.

    Args:
        image_file (str): The path to the image file.
        image_format (str): The format of the image file.

    Returns:
        str: The data URL of the image.
    """
    try:
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Could not read '{image_file}'.")
        exit()
    return f"data:image/{image_format};base64,{image_data}"

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
    whiteshirt_img_path = '/home/vbd-vanhk-l1-ubuntu/work/test_prompt_NO_whiteshirtblacktrouser.png'

    img_resized = encode_image(whiteshirt_img_path, max_size=224)
    print(len(img_resized))

    rule.prompt.system = "You are an advanced visual analysis assistant. When answering questions about an image, focus on precise details and provide highly accurate responses. Carefully analyze the image before answering, ensuring that your response is based on clear visual evidence. For yes/no questions, respond strictly based on what is visible in the image, without assumptions. If the image is unclear or ambiguous, state that explicitly."

    if rule.vlm_model.code_name == "gpt-4o":
        # ========== TEST GPT 4o ============
        client = OpenAI(
            base_url=rule.vlm_model.url,
            api_key=rule.vlm_model.api_key,
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": rule.prompt.system,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": rule.prompt.content,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": get_image_data_url(
                                    "/home/vbd-vanhk-l1-ubuntu/work/test_prompt_NO_whiteshirtblacktrouser.png", "png"),
                                "detail": "low"
                            },
                        },
                    ],
                },
            ],
            model=rule.vlm_model.code_name,
        )
        response_text = response.choices[0].message.content

    else:
        # === Khởi tạo model từ VLMModel
        llm = ChatOpenAI(model_name=rule.vlm_model.code_name, api_key=rule.vlm_model.api_key, base_url=rule.vlm_model.url, temperature=0.0)

        # === Gửi prompt đến VLM
        messages = [
                SystemMessage(f"{rule.prompt.system}"),
                HumanMessage(content=f'''"type": "text", "text": {rule.prompt.content}'''),
                HumanMessage(content=f'''"type": "image_url", "image_url": "url": "data:image/jpeg;base64,{img_resized}"''')
        ]
        # print(f"Messages: {messages}")

        response = llm.invoke(messages)
        response_text = response.content

    print(f"Response: {response_text}")

    # Nếu model trả về "Yes", tạo Alert
    if "Yes" in response_text:
        try:
            data = {
                'rule_id': rule.id,
                'rule_type': rule.type,
                'version_number': rule.current_version,
                'camera_id': camera.id,
                'camera_name': camera.name,
                'details': {
                            "input": rule.prompt.content,
                            "output": response_text,
                            }
            }
            camera_alert_service.create_alert(data)
            print('Camera alert created')

        except AttributeError as e:
            print(f"Loi tao camera alert: {e}")

    else:
        print("Khong phat hien bat thuong. Khong tao Alert")

    print(" ========= Next Camera ========= ")

    return response_text

if __name__ == "__main__":

    print('Loading list of rules')
    rules = Rule.objects.all()
    print('Process rules')

    for rule in rules:
        if rule.type != 1:
            print(f"rule {rule.id} has type not valid")

        else:
            print(f"Processing rule_id: {rule.id}")
            cameras = rule.cameras.all()
            for camera in cameras:
                print(f"=== Processing camera_id: {camera.id}")
                process_vlm_rule(rule, camera)

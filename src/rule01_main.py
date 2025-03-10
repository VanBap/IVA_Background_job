import base64
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from langchain_community.chat_models import ChatOpenAI
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

def encode_image_pil(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# VLM process
def process_vlm_rule(rule, camera):

    if not rule.prompt or not rule.vlm_model:
        print(f"==== Rule_id {rule.id} has no VLM model / prompt")

        # return "Rule không hợp lệ"
    else:

        # Khởi tạo model từ VLMModel
        llm = ChatOpenAI(model_name=rule.vlm_model.code_name, api_key=rule.vlm_model.api_key, base_url=rule.vlm_model.url, temperature=0.0)

        # Current Camera Image

        img_path = snapshot_image_from_camera(camera.id, camera.url)
        # img_path = '/home/vbd-vanhk-l1-ubuntu/work/image_1737360449.6729388.jpg'

        img_resized = encode_image(img_path, max_size=224)

        # Gửi prompt đến VLM
        messages = [
            SystemMessage(f"{rule.prompt.system}"),
            HumanMessage(content=f'''"type": "text", "text": {rule.prompt.content}'''),
            HumanMessage(content=f'''"type": "image_url", "image_url": "url": "data:image/jpeg;base64,{img_resized}"''')
        ]
        print(f"Messages: {messages}")

        response = llm.invoke(messages)
        # ========================================
        # 07/03/25
        print(f"Response: {response.content}")
        # Nếu model trả về "Yes", tạo Alert
        if "Yes" in response.content:
            try:
                data = {
                    'rule_id': rule.id,
                    'rule_type': rule.type,
                    'version_number': rule.current_version,
                    'camera_id': camera.id,
                    'camera_name': camera.name,
                    'desc': rule.prompt.content, # Update sau
                }
                camera_alert_service.create_alert(data)
                print('Camera alert created')

            except AttributeError as e:
                print(f"Loi tao camera alert: {e}")

        else:
            print("Khong phat hien bat thuong. Khong tao Alert")

        print(" ========= Next Camera ========= ")
        return response.content

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

from langchain_test.chat_models.base import BaseChatModel
from langchain_test.schema import AIMessage, HumanMessage, SystemMessage
from typing import List, Optional, Dict, Any
import requests


# Tao Custom Chat Model cho VLM
class VLMChatModel(BaseChatModel):
    api_key: str
    api_url: str

    def _call(self, messages: List[Any], stop: Optional[List[str]] = None) -> AIMessage:
        """Gửi danh sách tin nhắn lên API VLM và nhận kết quả."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # Chuyển đổi messages thành định dạng phù hợp với API VLM
        formatted_messages = [{"role": "user", "content": msg.content} for msg in messages if isinstance(msg, HumanMessage)]

        data = {"messages": formatted_messages}

        response = requests.post(self.api_url, json=data, headers=headers)

        if response.status_code == 200:
            result = response.json().get("result", "Không có kết quả")
            return AIMessage(content=result)
        else:
            return AIMessage(content=f"Lỗi API: {response.status_code}")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"api_url": self.api_url}

    @property
    def _llm_type(self) -> str:
        return "VLMChatModel"
# ===============================================================
# Gui Prompt duoi dang Chat Message

from langchain_test.schema import HumanMessage
from .vlm_chat_model import VLMChatModel

def analyze_image_with_vlm(rule, image_path):
    if not rule.vlm_model:
        raise ValueError("Chưa chọn model VLM.")

    # Lấy thông tin model từ DB
    vlm_model = rule.vlm_model
    chat_model = VLMChatModel(api_key=vlm_model.vlmmodel_apikey, api_url=vlm_model.vlmmodel_url)

    # Gửi chat message
    prompt_message = f"Phân tích ảnh {image_path}: {rule.prompt}"
    result = chat_model([HumanMessage(content=prompt_message)])

    return result.content

# ================================================================
# Xu ly ket qua va tao Alert
from .models import Rule, Alert
from .vlm_service import analyze_image_with_vlm

def process_rule(rule: Rule, camera, image_path: str):
    if rule.type == 1 and rule.prompt and rule.vlm_model:
        result = analyze_image_with_vlm(rule, image_path)

        if "có" in result.lower():  # Nếu phát hiện điều kiện trong ảnh
            Alert.objects.create(
                rule=rule,
                camera=camera,
                description=f"Phát hiện: {rule.prompt}"
            )

# ==============================================================
# Kiem tra
rule = Rule.objects.get(id=1)  # Rule có prompt "Trong ảnh có vũ khí không?"
image_path = "/path/to/image.jpg"

result = analyze_image_with_vlm(rule, image_path)
print(result)

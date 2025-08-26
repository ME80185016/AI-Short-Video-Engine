from typing import List

import requests
from dashscope.audio import qwen_tts

from .base import TextToSpeechConverter


class QwenTextToSpeechConverter(TextToSpeechConverter):
    def __init__(self, api_key: str, model: str, voices: List[str], folder: str, speed: float = 1.1):
        self.api_key = api_key
        self.model = model
        super().__init__(voices, folder, speed)

    async def generate_audio(self, content: str, voice: str, file_name: str):
        response = qwen_tts.SpeechSynthesizer.call(
            model=self.model,
            api_key=self.api_key,
            text=content,
            voice=voice,
        )
        
        # 检查响应是否有效
        if not response or not hasattr(response, 'output') or not response.output:
            raise ValueError(f"Invalid TTS response: {response}")
        
        if not hasattr(response.output, 'audio') or not response.output.audio:
            raise ValueError(f"No audio data in response: {response.output}")
        
        audio_url = response.output.audio["url"]

        response = requests.get(audio_url)
        response.raise_for_status()
        with open(file_name, "wb") as f:
            f.write(response.content)

import sys
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from io import BytesIO
from typing import AsyncGenerator

from LLM.prompts import speaker_prompts
from LLM.llm_stream import LLMStreamer
from tts.tts_stream import TTSStreamer

app = FastAPI()
tts_streamer = TTSStreamer()
llm = LLMStreamer()


async def llm_to_tts_stream(user_input : str, speaker_id : str = "Firefly", media_type: str = "wav") -> AsyncGenerator[bytes, None]:
    """从LLM到TTS的完整流式管道"""
    # 1. 获取LLM文本流
    llm_stream = llm.stream_output(user_input, speaker_id)
    tts_streamer.init_speaker_and_weight(speaker_id)
    
    # 2. 生成WAV头
    if media_type == "wav":
        yield tts_streamer.wave_header_chunk()
        media_type = "raw"
    buffer = ""
    punctuations = "。，！？；：“”‘’《》〈〉「」『』【】（）（）、…"  # 标点符号集合
    try:
        async for text_chunk in llm_stream:
            buffer += text_chunk
            while True:
                # 尝试按标点断句
                punctuation_index = next((i for i, char in enumerate(buffer) if char in punctuations), None)
                if punctuation_index is not None and punctuation_index + 1 <= 20:
                    segment = buffer[:punctuation_index + 1]
                    buffer = buffer[punctuation_index + 1:]
                # 如果没有合适的标点，按20字分块
                elif len(buffer) >= 20:
                    segment = buffer[:20]
                    buffer = buffer[20:]
                else:
                    break

                if segment:
                    tts_gen = tts_streamer.generate_audio(segment, media_type)
                    for sr, audio_chunk in tts_gen:
                        yield tts_streamer.pack_audio(BytesIO(), audio_chunk, sr, media_type).getvalue()

            await asyncio.sleep(0)  # 释放控制权，允许其他协程运行

    except asyncio.CancelledError:
        # 处理任务取消的情况
        pass


class TTSRequest(BaseModel):
    input: str
    speaker: str = "Firefly"

@app.post("/chat2audio")
async def api_llm_tts_stream(request: TTSRequest):
    return StreamingResponse(
        llm_to_tts_stream(
            user_input = request.input,
            speaker_id = request.speaker,
            media_type = "wav"
        ),
        media_type="audio/x-wav"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
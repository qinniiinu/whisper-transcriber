# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import os # operating system

app = FastAPI()

# 允許前端呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可改成你的前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入 Whisper 模型（你可改 base、small、medium）
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # 暫存上傳的音檔
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # 呼叫 Whisper 進行語音辨識
        result = model.transcribe(
            temp_path,
            fp16=False,
            language="zh",
            beam_size=5,
            best_of=5,
            verbose=False
        )
        text = result["text"]

        # 清理暫存檔案
        os.remove(temp_path)

        return JSONResponse(content={"text": text})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

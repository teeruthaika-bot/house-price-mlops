from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd

app = FastAPI(title="ระบบทำนายราคาบ้านอัจฉริยะ")
model = joblib.load("model/model.joblib")

class House(BaseModel):
    area: float = Field(gt=0, description="พื้นที่บ้าน (ตารางเมตร)")
    bedrooms: int = Field(ge=1, description="จำนวนห้องนอน")
    location: Literal["ในเมือง", "ชานเมือง", "ชนบท"] = Field(description="ทำเลที่ตั้ง")

@app.get("/health")
def health():
    return {"สถานะ": "ระบบทำงานปกติ พร้อมให้บริการ"}

@app.post("/predict", summary="คำนวณราคาประเมินบ้าน")
def predict(h: House):
    # แปลงภาษาไทยกลับเป็นภาษาอังกฤษเพื่อให้โมเดลทำงานได้
    loc_map = {"ในเมือง": "city", "ชานเมือง": "suburb", "ชนบท": "rural"}
    
    # จัดเตรียมข้อมูลก่อนส่งเข้าโมเดล
    data = {
        "area": [h.area],
        "bedrooms": [h.bedrooms],
        "location": [loc_map[h.location]]
    }
    X = pd.DataFrame(data)
    
    # ทำนายผลลัพธ์
    predicted_price = float(model.predict(X)[0])
    
    return {
        "ราคาประเมิน": round(predicted_price, 2),
        "หน่วย": "บาท"
    }
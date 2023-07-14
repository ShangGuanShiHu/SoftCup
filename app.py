from fastapi import FastAPI, UploadFile
import uvicorn as uvicorn
import pandas as pd
from pandas._typing import ReadCsvBuffer
import lightgbm as lgb
import numpy as np

app = FastAPI()

def predict(model, data_df):
    sample_ids = [str(id) for id in data_df['sample_id'].values]
    # 特征不能是sample_id, label, 以及剔除的特征
    X = data_df[[col for col in data_df.columns.tolist() if col!='sample_id' and col!='label' and col not in ['feature8', 'feature13', 'feature9', 'feature33', 'feature60', 'feature89']]].values
    y = model.predict(X)
    y = np.argmax(y,axis=1)
    result = {}
    for i, label in enumerate(y):
        result[sample_ids[i]] = int(label)
    return result

def get_model(default_model='./model/lgb'):
    lgb_clf = lgb.Booster(
        model_file=default_model
    )
    return lgb_clf


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


@app.post("/upload")
async def upload_file(file: UploadFile = UploadFile(...)):
    content = await file.read()  # Read the file content

    # Convert the string content to DataFrame using pandas
    from io import BytesIO
    df = pd.read_csv(BytesIO(content))
    # 注意模型的路径
    model = get_model()
    # 封装模型的用法
    result = predict(model,df)
    return result

if __name__ == "__main__":
    uvicorn.run(app, port=8888)
import bentoml
from bentoml.io import JSON

import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("xgboost_phone_predictor:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("phone_price_predictor", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    result = prediction[0]

    return result
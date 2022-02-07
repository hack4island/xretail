from flask import Flask, jsonify, request
import json
from adalo_manager import *
from inventory import analyze
import torchvision
import torch

record = {
  "EAN": 0,
  "KEYWORDS": "cheufitel",
  "Store": "string",
  "IMAGE_URL": "string",
  "longitude": 0,
  "latitude": 0,
  "PRICE": "string",
  "Time": "2019-08-24T14:15:22Z",
  "Name": "toto",
  "CITY": "stringdddd",
  "User" : 1
}


model_pad = torch.hub.load('ultralytics/yolov5', 'custom', path='pad_ean_fields.pt')
model_ean = torch.hub.load('ultralytics/yolov5', 'custom', path='ean.pt')


app = Flask(__name__)

from io import BytesIO
from PIL import Image, ImageDraw, ImageOps


@app.route('/', methods=['POST'])
def inventory():
	json_data = json.loads(request.data)
	index = str(int(json_data["body"])+1)
	resp_get = get_record(index=index)
	if resp_get.json()["IMAGE"]["url"]:
		response = requests.get(resp_get.json()["IMAGE"]["url"])
		img = Image.open(BytesIO(response.content))
		dic_resp = analyze(zoomed_im=img, model_ean=model_ean, model_pad=model_pad)
		print(dic_resp)
		print(resp_get.json()["IMAGE"]["url"])
		record["EAN"] = int(dic_resp["ean"])
		record["PRICE"] = dic_resp["price"]
		record["IMAGE_URL"] = resp_get.json()["IMAGE"]["url"]
		try:
			off_json = requests.get("https://world.openfoodfacts.org/api/v0/product/{}.json".format(record["EAN"]))
		except:
			print("erreur")

		record["PRODUCT_URL"] = off_json.json()["product"]["image_front_small_url"]
		record["KEYWORDS"] = " ".join(off_json.json()["product"]["product_name"])
		resp = update_record(index=index,json_body=record)

		#resp.status_code = 200
	return json_data

import pandas as pd
import json

@app.route('/export', methods=['POST'])
def export():
		rec_json = get_all_record()
		products = rec_json.json()["records"]
		df = pd.DataFrame.from_dict(products)
		df.to_csv("./result.csv")
		files = {'file': open('./result.csv','rb')}
		json_data = json.loads(request.data)
		index = str(int(json_data["index"])+1)
		resp_get = get_store(index=index)
		store = resp_get.json()
		response = requests.post("https://file.io", files=files)
		print(response.json())
		store["INVENTORY_URL"] = response.json()['link']
		resp = update_store(index=index,json_body=store)
		#requests.get(response.json()['link'])
		#resp.status_code = 200
		return json_data


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

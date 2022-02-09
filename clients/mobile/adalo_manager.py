import requests
import json



record = {
  "EAN": 0,
  "KEYWORDS": "string",
  "Store": "string",
  "IMAGE_URL": "string",
  "longitude": 0,
  "latitude": 0,
  "PRICE": "string",
  "Time": "2019-08-24T14:15:22Z",
  "Name": "toto",
  "CITY": "string",
  "User" : 1
}


inventory_endpoint = "https://api.adalo.com/v0/apps/xxxxxxxxx/collections/xxxxxxxxxxxxxxx/"
headers ={}
headers["Accept"] = "application/json"
headers["Authorization"] = "Bearer xxxxxxxxx"

inventory_endpoint_store = "https://api.adalo.com/v0/apps/xxxxxxxxxxxxx/collections/xxxxxxxxxxxxxxxxx/"


def create_record(url=inventory_endpoint, headers=headers, index=None, json_body=None):
        resp = requests.post(url, headers=headers, json=json_body)
        return resp


def update_record(url=inventory_endpoint, headers=headers, index=None, json_body=None):
    index_base = url + index
    resp = requests.put(index_base, headers=headers, json=json_body, verify=False)
    print(resp)


def get_record(url=inventory_endpoint, headers=headers, index=None, json_body=None):
    resp = requests.get(url+index, headers=headers, verify=False)
    return resp


def get_all_record(url=inventory_endpoint, headers=headers, json_body=None):
    resp = requests.get(url, headers=headers, verify=False)
    return resp


def get_store(url=inventory_endpoint_store, headers=headers, index=None, json_body=None):
    resp = requests.get(url+index, headers=headers, verify=False)
    print(resp)
    return resp


def update_store(url=inventory_endpoint_store, headers=headers, index=None, json_body=None):
    index_base = url + index
    resp = requests.put(index_base, headers=headers, json=json_body, verify=False)
    print(resp)

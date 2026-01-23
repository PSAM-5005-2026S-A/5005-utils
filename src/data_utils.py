from io import BytesIO, StringIO
from PIL import Image as PImage

import csv
import json
import urllib.request as request

def object_from_json_url(url):
  with request.urlopen(url) as response:
    return json.load(response)

def image_from_url(url):
  with request.urlopen(url) as response:
    image_data = BytesIO(response.read())
    return PImage.open(image_data)

def list_from_csv_url(url):
  with request.urlopen(url) as response:
    str_data = StringIO(response.read().decode("utf8"))
    csvreader = csv.reader(str_data)
    list_data = list(csvreader)

    for row_idx, row in enumerate(list_data):
      for col_idx, col_val in enumerate(row):
        try:
          list_data[row_idx][col_idx] = float(col_val)
        except ValueError:
          pass

    return list_data

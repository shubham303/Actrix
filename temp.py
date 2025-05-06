from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import dotenv

dotenv.load_dotenv()

device = torch.device("mps")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)

inputs = processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)

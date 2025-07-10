import boto3
import json
import base64
import os

prompt="""
Provide me one 4k hd image of a person who is swimming in the river.
"""
prompt_template=[{"text":prompt,"weight":1}]

bedrock=boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

payload={
  "text-promts":prompt_template,
  "cfg_scale":10,
  "seed":0,
  "steps":50,
  "width":512,
  "height":512
}

body=json.dumps(payload)
model_Id="stability.stable-diffusion-xl-v1"
response=bedrock.invoke_model(
    body=body,
    modelId=model_Id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
print(response_body)

artifacts=response_body.get("artifacts")[0]
image_encoded=artifacts.get("base64").encode('utf-8')
image_bytes=base64.b64decode(image_encoded)
output_dir="output" 
os.makedirs(output_dir, exist_ok=True)

file_name=f"{output_dir}/generated-img.png"

with open(file_name,"wb") as f:
    f.write(image_bytes)
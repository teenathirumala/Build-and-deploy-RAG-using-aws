import os
import json
import boto3

print("imported successfully")

prompt = "You are a cricket expert now tell me when will RCB win the IPL"

# Create Bedrock Runtime client (use correct region)
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

# Llama3 expects the prompt wrapped in [INST]...[/INST]
payload = {
    "prompt": f"[INST] {prompt} [/INST]",
    "max_gen_len": 512,
    "temperature": 0.3,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

try:
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )

    raw_body = response["body"].read()
    print("\n🟡 Raw response body:")
    print(raw_body)

    try:
        response_body = json.loads(raw_body)
        print("\n🟢 Parsed response body:")
        print(json.dumps(response_body, indent=2))

        # Try common output keys
        for key in ["generation", "output", "outputs", "completions"]:
            if key in response_body:
                print(f"\n🔵 Model output found under key: '{key}'")
                print(response_body[key])
                break
        else:
            print("\n🔴 No known key found in the response body.")

    except json.JSONDecodeError as je:
        print(f"❌ JSON decoding failed: {je}")

except Exception as e:
    print(f"❌ Exception occurred during invoke_model: {e}")

import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

import numpy as np
from PIL import Image

# MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
# MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")

# App title
st.set_page_config(page_title="Image Generation with SDXL and OpenVino")

with st.sidebar:
    st.title("Image Generation with SDXL and OpenVino")

    st.session_state.model_sd_loaded = False

    try:
        res = requests.get(url="http://localhost:8080/ping")
        res = requests.get(url=f"http://localhost:8081/models/{MODEL_NAME_SD}")
        status = "NOT READY"
        if res.status_code == 200:
            status = json.loads(res.text)[0]["workers"][0]["status"]

        if status == "READY":
            st.session_state.model_sd_loaded = True
            st.success("Proceed to entering your prompt input!", icon="👉")
        else:
            st.warning(f"Model {MODEL_NAME_SD} not loaded in TorchServe", icon="⚠️")
    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    if st.session_state.model_sd_loaded:
        st.success(f"Model loaded: {MODEL_NAME_SD}", icon="👉")

prompt = st.text_input("Text Prompt", "An astronaut riding a horse")

#TODO: For Tests, delete when LLM added
prompt = [prompt,
          "A robot playing a violin",
          "A dragon flying over the mountains",
          "A mermaid painting a sunset on the beach"
        ]

def generate_sd_response_v1(prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    response = []
    for pr in prompt_input:
        data = json.dumps(
            {
                "prompt": pr
            }
        )
        response.append(requests.post(url=url, data=data).text)
    return response

def response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]

if st.button("Generate Images"):
    with st.spinner('Generating images...'):
        res = generate_sd_response_v1(prompt)
        images = response_postprocess(res)
        st.image(images, caption=["Generated Image"] * len(images), use_column_width=True)

import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
import asyncio
import aiohttp
import datetime

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
            st.warning(
                f"Model {MODEL_NAME_SD} not loaded in TorchServe", icon="⚠️")
    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    if st.session_state.model_sd_loaded:
        st.success(f"Model loaded: {MODEL_NAME_SD}", icon="👉")

prompt = st.text_input("Text Prompt", "An astronaut riding a horse")

# TODO: For Tests, delete when LLM added
prompt = [prompt,
          "A robot playing a violin",
          "A dragon flying over the mountains",
          "A mermaid painting a sunset on the beach"
          ]


def generate_sd_response_v1(prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    response = []
    for pr in prompt_input:
        data_input = json.dumps(
            {
                "prompt": pr
            }
        )
        response.append(requests.post(url=url, data=data_input).text)
    return response


async def send_inference_request(session, prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"

    data_input = json.dumps(
        {
            "prompt": prompt_input
        }
    )

    async with session.post(url, data=data_input) as response:
        assert response.status == 200
        resp_text = await response.text()
        return resp_text


async def generate_sd_response_v2(prompts):
    my_timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=my_timeout) as session:
        tasks = []
        for prompt in prompts:
            tasks.append(send_inference_request(session, prompt))

        return await asyncio.gather(*tasks)


def response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]

# if st.button("Generate Images"):
#     with st.spinner('Generating images...'):
#         res = generate_sd_response_v1(prompt)
#         images = response_postprocess(res)
#         st.image(images, caption=["Generated Image"] * len(images), use_column_width=True)


if st.button("Generate Images"):
    with st.spinner('Generating images...'):
        res = asyncio.run(generate_sd_response_v2(prompt))
        images = response_postprocess(res)
        st.image(images, caption=["Generated Image"]
                 * len(images), use_column_width=True)

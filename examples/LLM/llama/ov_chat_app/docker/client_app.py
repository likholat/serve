import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
import asyncio
import aiohttp
import time
import re

import numpy as np
from PIL import Image

MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")

# App title
st.set_page_config(page_title="Image Generation with SDXL and OpenVino")

with st.sidebar:
    st.title("Image Generation with SDXL and OpenVino")

    st.session_state.model_sd_loaded = False

    try:
        res = requests.get(url="http://localhost:8080/ping")

        sd_res = requests.get(
            url=f"http://localhost:8081/models/{MODEL_NAME_SD}")
        sd_status = "NOT READY"

        llm_res = requests.get(
            url=f"http://localhost:8081/models/{MODEL_NAME_LLM}")
        llm_status = "NOT READY"

        if sd_res.status_code == 200:
            sd_status = json.loads(sd_res.text)[0]["workers"][0]["status"]

        if llm_res.status_code == 200:
            llm_status = json.loads(llm_res.text)[0]["workers"][0]["status"]

        st.session_state.model_sd_loaded = True if sd_status == "READY" else False
        if st.session_state.model_sd_loaded:
            st.success(f"Model loaded: {MODEL_NAME_SD}", icon="👉")
        else:
            st.warning(
                f"Model {MODEL_NAME_SD} not loaded in TorchServe", icon="⚠️")

        st.session_state.model_llm_loaded = True if llm_status == "READY" else False
        if st.session_state.model_llm_loaded:
            st.success(f"Model loaded: {MODEL_NAME_LLM}", icon="👉")
        else:
            st.warning(
                f"Model {MODEL_NAME_LLM} not loaded in TorchServe", icon="⚠️")

        if sd_status == "READY" and llm_status == "READY":
            st.success("Proceed to entering your prompt input!", icon="👉")

    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    st.subheader("Number of images to generate")
    images_num = st.sidebar.slider(
        "num_of_img", min_value=1, max_value=4, value=4, step=1
    )

    st.subheader("Llama Model parameters")
    max_new_tokens = st.sidebar.slider(
        "max_new_tokens", min_value=30, max_value=250, value=50, step=10
    )
    temperature = st.sidebar.slider(
        "temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1
    )
    top_k = st.sidebar.slider(
        "top_k", min_value=1, max_value=200, value=200, step=1
    )


    st.subheader("SD Model parameters")
    num_inference_steps = st.sidebar.slider(
        "steps", min_value=1, max_value=150, value=30, step=1
    )
    guidance_scale = st.sidebar.slider(
        "guidance_scale", min_value=1.0, max_value=30.0, value=5.0, step=0.5
    )
    height = st.sidebar.slider(
        "height", min_value=256, max_value=2048, value=768, step=8
    )
    width = st.sidebar.slider(
        "width", min_value=256, max_value=2048, value=768, step=8
    )


prompt = st.text_input("Text Prompt")

def generate_sd_response_v1(prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    response = []
    for pr in prompt_input:
        data_input = json.dumps(
            {
                "prompt": pr,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
            }
        )
        response.append(requests.post(url=url, data=data_input).text)
    return response


async def send_inference_request(session, prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    data_input = json.dumps(
        {
            "prompt": prompt_input,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
        }
    )

    async with session.post(url, data=data_input) as response:
        assert response.status == 200
        resp_text = await response.text()
        return resp_text


async def generate_sd_response_v2(prompts):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        tasks = []
        for prompt in prompts:
            tasks.append(send_inference_request(session, prompt))

        return await asyncio.gather(*tasks)


def sd_response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]


def preprocess_llm_input(input_prompt, images_num = 2):
    return f"Generate {images_num} different unique prompts similar to: {input_prompt}. \
             Add semicolon between the prompts. \
             Generated string of prompts should be included in square brackets. E.g.:"


def trim_prompt(s):
    return re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s)

def postprocess_llm_response(prompts, original_prompt=None, images_num=2):

    prompts = [trim_prompt(item) for item in prompts.split(";")]
    prompts = list(filter(None, prompts))
    
    if original_prompt:
        prompts.insert(0, original_prompt)

    prompts = prompts[:images_num]
    assert len(prompts) == images_num, "Llama Model generated too few prompts!"

    return prompts


def generate_llm_model_response(input_prompt):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_LLM}"
    data = json.dumps(
        {
            "prompt": input_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
    )

    res = requests.post(url=url, data=data, headers=headers, stream=True)
    assert res.status_code == 200

    return res.text


if 'gen_images' not in st.session_state:
    st.session_state.gen_images = []
if 'gen_captions' not in st.session_state:
    st.session_state.gen_captions = []


def display_images_in_grid(images, captions):
    cols = st.columns(2)
    for i, (img, caption) in enumerate(zip(images, captions)):
        col = cols[i % 2]
        col.image(img, caption=caption, use_column_width=True)


if st.button("Generate Images"):
    with st.spinner('Generating images...'):
        total_start_time = time.time()

        llm_res = [prompt]
        if images_num > 1:
            prompt_input = preprocess_llm_input(prompt, images_num)

            # start_time = time.time()
            llm_res = generate_llm_model_response(prompt_input)
            # llm_inference_time = time.time() - start_time

            llm_res = postprocess_llm_response(llm_res, prompt, images_num)

        st.session_state.gen_captions[:0] = llm_res

        start_time = time.time()
        # sd_res = generate_sd_response_v1(llm_res)
        sd_res = asyncio.run(generate_sd_response_v2(llm_res))
        sd_inference_time = time.time() - start_time

        images = sd_response_postprocess(sd_res)
        st.session_state.gen_images[:0] = images

        # if images_num > 1:
        #     st.write(f"LLM inference time: {llm_inference_time:.2f} seconds")
        # st.write(f"SD inference time: {sd_inference_time:.2f} seconds")

        display_images_in_grid(st.session_state.gen_images, st.session_state.gen_captions)

        total_time = time.time() - total_start_time
        print(f'TOTAL APP TIME: {total_time:.2f}')

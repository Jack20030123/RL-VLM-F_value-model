import pathlib
import textwrap
import os
from PIL import Image
import google.generativeai as genai
import time
from io import BytesIO
from matplotlib import pyplot as plt
import numpy as np

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Updated to use gemini-2.5-flash-lite (fastest and lowest cost model)
# Gemini 2.5 Flash Lite: $0.10 input per 1M, $0.40 output per 1M
model = genai.GenerativeModel('gemini-2.5-flash-lite')
text_model = genai.GenerativeModel('gemini-2.5-flash-lite')

        
def gemini_query_1(query_list, temperature=0, debug=False):
    beg = time.time()

    success = False
    try_cnt = 0
    while not success:
        try:
            response = model.generate_content(query_list,
            # response = model.generate_content([prompt, image1, image2],
                                            generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    # candidate_count=1,
                    # stop_sequences=['x'],
                    # max_output_tokens=20,
                    temperature=temperature),
                safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )

            response.resolve()
            success = True
        except Exception as e:
            if debug:
                print(f"[DEBUG GEMINI] API exception: {type(e).__name__}: {str(e)}")
            print("gemini retrying...")
            time.sleep(3)
            try_cnt += 1
            if try_cnt >= 5:
                if debug:
                    print(f"[DEBUG GEMINI] Max retries ({try_cnt}) reached, giving up")
                break

    end = time.time()
    print("time elapsed: ", end - beg)

    if success:
        try:
            # Debug information for empty response diagnosis
            if debug:
                print(f"[DEBUG GEMINI] prompt_feedback: {response.prompt_feedback}")
                print(f"[DEBUG GEMINI] candidates count: {len(response.candidates) if response.candidates else 0}")

                if response.candidates:
                    candidate = response.candidates[0]
                    print(f"[DEBUG GEMINI] finish_reason: {candidate.finish_reason}")
                    print(f"[DEBUG GEMINI] safety_ratings: {candidate.safety_ratings}")

                print(f"[DEBUG GEMINI] response.text raw: {repr(response.text)}")  # repr() shows escape characters
                print(f"[DEBUG GEMINI] response.text length: {len(response.text)}")

            result = response.text.strip().split("\n")[-1].strip()

            if debug:
                print(f"[DEBUG GEMINI] after processing: {repr(result)}")
                print(f"[DEBUG GEMINI] after processing length: {len(result)}")

            return result
        except Exception as e:
            if debug:
                print(f"[DEBUG GEMINI] Error processing response.text: {type(e).__name__}: {str(e)}")
            return -1
    else:
        return -1

def gemini_query_2(query_list, summary_prompt, temperature=0, debug=False):
    beg = time.time()

    success = False
    try_cnt = 0
    while not success:
        try:
            response = model.generate_content(query_list,
                                            generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    # candidate_count=1,
                    # stop_sequences=['x'],
                    # max_output_tokens=20,
                    temperature=temperature),
                safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )

            response.resolve()

            summary_response = text_model.generate_content(
                    summary_prompt.format(response.text),
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                    ),
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
            )
            summary_response.resolve()
            success = True
        except Exception as e:
            if debug:
                print(f"[DEBUG GEMINI FF] API exception: {type(e).__name__}: {str(e)}")
            print("gemini retrying...")
            time.sleep(2)
            try_cnt += 1
            if try_cnt >= 5:
                if debug:
                    print(f"[DEBUG GEMINI FF] Max retries ({try_cnt}) reached, giving up")
                break



    end = time.time()
    print("time elapsed: ", end - beg)
    if success:
        try:
            # Debug information for empty response diagnosis
            if debug:
                # First response (analysis)
                print(f"[DEBUG GEMINI FF] First response - prompt_feedback: {response.prompt_feedback}")
                print(f"[DEBUG GEMINI FF] First response - candidates count: {len(response.candidates) if response.candidates else 0}")
                if response.candidates:
                    candidate = response.candidates[0]
                    print(f"[DEBUG GEMINI FF] First response - finish_reason: {candidate.finish_reason}")
                print(f"[DEBUG GEMINI FF] First response - text raw: {repr(response.text)}")
                print(f"[DEBUG GEMINI FF] First response - text length: {len(response.text)}")

                # Summary response (label)
                print(f"[DEBUG GEMINI FF] Summary response - prompt_feedback: {summary_response.prompt_feedback}")
                print(f"[DEBUG GEMINI FF] Summary response - candidates count: {len(summary_response.candidates) if summary_response.candidates else 0}")
                if summary_response.candidates:
                    candidate = summary_response.candidates[0]
                    print(f"[DEBUG GEMINI FF] Summary response - finish_reason: {candidate.finish_reason}")
                    print(f"[DEBUG GEMINI FF] Summary response - safety_ratings: {candidate.safety_ratings}")
                print(f"[DEBUG GEMINI FF] Summary response - text raw: {repr(summary_response.text)}")
                print(f"[DEBUG GEMINI FF] Summary response - text length: {len(summary_response.text)}")

            result = summary_response.text.split("\n")[0].strip().lstrip()

            if debug:
                print(f"[DEBUG GEMINI FF] after processing: {repr(result)}")
                print(f"[DEBUG GEMINI FF] after processing length: {len(result)}")

            return result
        except Exception as e:
            if debug:
                print(f"[DEBUG GEMINI FF] Error processing response.text: {type(e).__name__}: {str(e)}")
            return -1
    else:
        return -1

if __name__ == "__main__":
    from prompt import (
        gemini_free_query_env_prompts, gemini_summary_env_prompts,
        gemini_free_query_prompt1, gemini_free_query_prompt2,

    ) 
    import numpy as np
    from matplotlib import pyplot as plt

    def process_image(image):
        mask1 = (image[:, :, 0] == 255) & (image[:, :, 1] == 255) & (image[:, :, 2] == 255)
        mask2 = (image[:, :, 0] == 0) & (image[:, :, 1] >= 170) & (image[:, :, 2] == 0)
        mask = mask1 | mask2
        image[~mask] = (0, 0, 0)
        return image

    image_path = "data/images/metaworld_sweep-into-v2/image_30_combined.png"
    image = Image.open(image_path)
    image = np.array(image)[100:, :, :]
    image = Image.fromarray(image)

    image_1_path = "data/images/metaworld_sweep-into-v2/image_6_1.png"
    image_2_path = "data/images/metaworld_sweep-into-v2/image_6_2.png"
    image_1 = Image.open(image_1_path)
    image_2 = Image.open(image_2_path)

    env_name = "metaworld_sweep-into-v2"
    gemini_query_2(
    [
        gemini_free_query_prompt1,
        image_1, 
        gemini_free_query_prompt2,
        image_2, 
        gemini_free_query_env_prompts[env_name]
    ],
        gemini_summary_env_prompts[env_name]
    )

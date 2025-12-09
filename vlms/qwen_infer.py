import torch
from PIL import Image
import time
import os
import base64
import io
import requests
import numpy as np

# ==============================================================================
# HTTP Client Configuration
# ==============================================================================
# The Qwen model runs in a separate HTTP server (qwen conda environment)
# to avoid Python version conflicts with the rlvlmf environment.
# This allows rlvlmf (Python 3.9) to use Qwen3-VL (requires Python 3.10).
QWEN_SERVER_URL = os.environ.get('QWEN_SERVER_URL', 'http://127.0.0.1:8000')

# ==============================================================================
# Utility Functions
# ==============================================================================

def encode_image_to_base64(image):
    """
    Encode image to base64 string for HTTP transmission.

    Args:
        image: PIL Image or numpy array (H, W, C) in uint8 format

    Returns:
        base64 encoded string
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image.astype('uint8'))

    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

# ==============================================================================
# HTTP Client Functions (mimicking gemini_infer.py API)
# ==============================================================================
# These functions call the Qwen HTTP server instead of loading the model locally.
# This solves the Python version incompatibility:
#   - rlvlmf environment: Python 3.9 (old dependencies)
#   - qwen environment: Python 3.10 (required by Qwen3-VL)
# ==============================================================================

def qwen_query_1(query_list, temperature=0):
    """
    One-stage preference query via HTTP (similar to gemini_query_1).

    This function mimics the gemini_query_1 API but calls the Qwen HTTP server
    instead of using local inference.

    Args:
        query_list: List containing [prompt1, image1, prompt2, image2, question_prompt]
                   Following the same format as gemini_query_1
        temperature: Sampling temperature (0 for greedy decoding)

    Returns:
        Preference label (0, 1, -1) or -1 on error
    """
    beg = time.time()

    success = False
    try_cnt = 0

    while not success:
        try:
            # Extract images and prompt from query_list
            # Format: [prompt1_text, image1, prompt2_text, image2, question_prompt]
            if len(query_list) != 5:
                print(f"Warning: Expected 5 elements in query_list, got {len(query_list)}")
                return -1

            image1 = query_list[1]  # PIL Image or numpy array
            image2 = query_list[3]  # PIL Image or numpy array
            question_prompt = query_list[4]  # The actual question

            # Encode images to base64 for HTTP transmission
            image1_b64 = encode_image_to_base64(image1)
            image2_b64 = encode_image_to_base64(image2)

            # Call HTTP server (one-stage preference endpoint)
            response = requests.post(
                f'{QWEN_SERVER_URL}/preference_one_stage',
                json={
                    'image1': image1_b64,
                    'image2': image2_b64,
                    'prompt': question_prompt,
                    'temperature': temperature
                },
                timeout=120  # 2 minute timeout
            )

            if response.status_code == 200:
                result = response.json()
                preference = result.get('preference', -1)
                success = True
            else:
                print(f"Qwen server error: {response.status_code}")
                raise Exception(f"Server returned {response.status_code}")

        except Exception as e:
            print(f"qwen retrying... ({e})")
            time.sleep(3)
            try_cnt += 1
            if try_cnt >= 5:
                break

    end = time.time()
    print("time elapsed: ", end - beg)

    if success:
        try:
            # Return the preference (already extracted by server)
            return preference
        except:
            return -1
    else:
        return -1

def qwen_query_2(query_list, summary_prompt, temperature=0):
    """
    Two-stage preference query via HTTP (similar to gemini_query_2).

    This function mimics the gemini_query_2 API but calls the Qwen HTTP server
    for two-stage processing:
      Stage 1: Vision model analyzes images
      Stage 2: Text extraction to get final answer

    Args:
        query_list: List containing [prompt1, image1, prompt2, image2, question_prompt]
        summary_prompt: Template string for extracting final answer (contains "{}")
        temperature: Sampling temperature

    Returns:
        Preference label (0, 1, -1) or -1 on error
    """
    beg = time.time()

    success = False
    try_cnt = 0

    while not success:
        try:
            # Extract images and prompt from query_list
            if len(query_list) != 5:
                print(f"Warning: Expected 5 elements in query_list, got {len(query_list)}")
                return -1

            image1 = query_list[1]  # PIL Image or numpy array
            image2 = query_list[3]  # PIL Image or numpy array
            question_prompt = query_list[4]  # The actual question

            # Encode images to base64
            image1_b64 = encode_image_to_base64(image1)
            image2_b64 = encode_image_to_base64(image2)

            # Stage 1: Analyze images with vision model
            analyze_response = requests.post(
                f'{QWEN_SERVER_URL}/preference_two_stage_analyze',
                json={
                    'image1': image1_b64,
                    'image2': image2_b64,
                    'prompt': question_prompt,
                    'temperature': temperature
                },
                timeout=120
            )

            if analyze_response.status_code != 200:
                raise Exception(f"Stage 1 failed: {analyze_response.status_code}")

            analysis = analyze_response.json().get('analysis', '')

            # Stage 2: Extract final answer from analysis
            extract_response = requests.post(
                f'{QWEN_SERVER_URL}/preference_two_stage_extract',
                json={
                    'analysis': analysis,
                    'summary_prompt': summary_prompt,
                    'temperature': temperature
                },
                timeout=60
            )

            if extract_response.status_code == 200:
                result = extract_response.json()
                preference = result.get('preference', -1)
                success = True
            else:
                raise Exception(f"Stage 2 failed: {extract_response.status_code}")

        except Exception as e:
            print(f"qwen retrying... ({e})")
            time.sleep(2)
            try_cnt += 1
            if try_cnt >= 5:
                break

    end = time.time()
    print("time elapsed: ", end - beg)

    if success:
        try:
            return preference
        except:
            return -1
    else:
        return -1

def qwen_image_text_matching(rgb, text, temperature=0):
    """
    Single image scoring via HTTP (similar to CLIP/BLIP2 for direct reward).

    This function is used for direct reward computation, not preference learning.
    It returns a continuous score between 0 and 1 indicating how well the image
    matches the text description.

    Args:
        rgb: RGB image (numpy array or PIL Image)
        text: Text prompt (goal description)
        temperature: Sampling temperature

    Returns:
        Score between 0 and 1 (0 = goal not achieved, 1 = goal fully achieved)
    """
    beg = time.time()

    try:
        # Encode image to base64
        image_b64 = encode_image_to_base64(rgb)

        # Call HTTP server (scoring endpoint)
        response = requests.post(
            f'{QWEN_SERVER_URL}/score',
            json={
                'image': image_b64,
                'text': text,
                'temperature': temperature
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            score = result.get('score', 0.5)
            end = time.time()
            print(f"Qwen scoring time elapsed: {end - beg:.2f}s")
            return score
        else:
            print(f"Qwen server error: {response.status_code}")
            return 0.5

    except Exception as e:
        print(f"Qwen scoring error: {e}")
        import traceback
        traceback.print_exc()
        return 0.5

# ==============================================================================
# Legacy Local Inference Functions (for reference only)
# ==============================================================================
# The functions below are the ORIGINAL implementations that load and run the
# Qwen model directly in the current process (local inference).
#
# NOTE: These are NOT used in production because:
#   1. They require Python 3.10+ (incompatible with rlvlmf's Python 3.9)
#   2. They require newer transformers/torch versions
#   3. Loading the model in the training process wastes memory
#
# We keep them here for:
#   - Documentation/reference
#   - Potential fallback if HTTP server is unavailable
#   - Direct testing in the qwen environment
# ==============================================================================

model = None
processor = None
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_model():
    """
    Initialize Qwen3-VL model and processor for local inference.

    WARNING: This should only be called in the qwen conda environment!
    Do NOT use this in the rlvlmf environment (Python version incompatible).
    """
    global model, processor
    if model is None:
        print("Loading Qwen3-VL-8B-Instruct model for local inference...")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct"
        )
        print("Qwen3-VL model loaded successfully!")

def qwen_query_1_local(image_path, prompt, temperature=0):
    """
    [LEGACY LOCAL INFERENCE] Single image query (similar to gemini_query_1 and gpt4v_infer)

    NOTE: This is the LOCAL inference version that loads the model directly.
    Use qwen_query_1() (without _local) for the HTTP client version.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the query
        temperature: Sampling temperature (0 for greedy decoding)

    Returns:
        Response text or -1 on error
    """
    beg = time.time()

    try:
        init_model()

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        end = time.time()
        print(f"time elapsed: {end - beg}")

        # Return the last line (similar to gemini_query_1)
        return response.strip().split("\n")[-1].strip().lstrip()

    except Exception as e:
        print(f"Qwen query error: {e}")
        import traceback
        traceback.print_exc()
        return -1

def qwen_query_2_local(image1_path, image2_path, prompt, temperature=0):
    """
    [LEGACY LOCAL INFERENCE] Two image comparison query (similar to gemini_query_2 and gpt4v_infer_2)

    NOTE: This is the LOCAL inference version that loads the model directly.
    Use qwen_query_2() (without _local) for the HTTP client version.

    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        prompt: Text prompt for the query
        temperature: Sampling temperature

    Returns:
        Response text or -1 on error
    """
    beg = time.time()

    try:
        init_model()

        # Load images
        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        # Prepare messages with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1:"},
                    {"type": "image", "image": image1},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image", "image": image2},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[image1, image2],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        end = time.time()
        print(f"time elapsed: {end - beg}")

        # Return the first line (similar to gpt4v_infer_2 which extracts answer)
        return response.strip().split("\n")[0].strip().lstrip()

    except Exception as e:
        print(f"Qwen query error: {e}")
        import traceback
        traceback.print_exc()
        return -1

if __name__ == "__main__":
    # Test example (only for testing, not used in production)
    print("Qwen inference module loaded. Use qwen_query_1() or qwen_query_2() functions.")

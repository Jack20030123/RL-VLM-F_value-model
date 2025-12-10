#!/usr/bin/env python3
"""
Qwen3-VL HTTP Inference Server

This server runs in the qwen conda environment (Python 3.10) and provides
HTTP endpoints for Qwen3-VL inference. The rlvlmf environment can call this
server to use Qwen without dependency conflicts.

Usage:
    # In qwen environment
    source /opt/conda/etc/profile.d/conda.sh
    conda activate qwen
    python vlms/servers/qwen_server.py

Endpoints:
    POST /score - Single image scoring (for direct reward)
    POST /preference_one_stage - One-stage preference judgment
    POST /preference_two_stage_analyze - Two-stage: first stage (analysis)
    POST /preference_two_stage_extract - Two-stage: second stage (extract answer)
    GET /health - Health check
"""

import os
import sys
import time
import base64
import io
import re
import traceback
from PIL import Image
import torch
from flask import Flask, request, jsonify

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = Flask(__name__)

# Debug mode (set QWEN_DEBUG=1 environment variable to enable debug logging)
DEBUG = os.environ.get('QWEN_DEBUG', '0') == '1'

# Model selection (set QWEN_MODEL environment variable or use --model argument)
# Default: Qwen3-VL-8B-Instruct
# Options: Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen3-VL-32B-Instruct
MODEL_NAME = os.environ.get('QWEN_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')

# Global variables for model (lazy loading)
model = None
processor = None
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_model():
    """Initialize Qwen3-VL model and processor"""
    global model, processor
    if model is None:
        print("=" * 60)
        print(f"Loading {MODEL_NAME} model...")
        print("=" * 60)
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load model to single GPU (no CPU offload)
        # device_map={"": 0} forces model onto first visible GPU (set by CUDA_VISIBLE_DEVICES)
        # For 8B model: needs ~9GB GPU memory
        # For 32B model: needs ~70GB GPU memory (requires A100 80GB)
        # Will fail with OOM if GPU memory insufficient (better than slow CPU offload)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
            device_map={"": 0},  # Force entire model onto first visible GPU
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )
        print(f"✓ {MODEL_NAME} loaded successfully!")
        print("=" * 60)

def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def extract_number(text, default=0.5):
    """Extract a number (0-1) from text response"""
    try:
        # Try to find numbers like 0.85, 1, 0, etc.
        numbers = re.findall(r'0?\.\d+|[01]\.?\d*', text)
        if numbers:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))
        return default
    except:
        return default

def extract_preference(text, debug=False):
    """Extract preference label (0, 1, or -1) from text response

    Args:
        text: Response text from the model
        debug: If True, print the full response for debugging

    Returns:
        Preference label: 0, 1, or -1
    """
    try:
        text = text.strip()

        # Debug: Print the full response
        if debug:
            print("=" * 60)
            print("QWEN FULL RESPONSE:")
            print("-" * 60)
            print(text)
            print("-" * 60)

        # Try to find standalone numbers using regex
        # This matches -1, 0, or 1 as standalone tokens (not part of "Image 1", etc.)
        import re

        # Pattern 1: Find exact matches like "Answer: 1" or just "1" on a line
        # Look for -1, 0, or 1 that are not preceded/followed by word characters or digits
        pattern = r'(?:^|[^\w\d])(-1|0|1)(?:[^\w\d]|$)'
        matches = re.findall(pattern, text)

        if matches:
            # Return the last match (most likely to be the final answer)
            result = int(matches[-1])
            if debug:
                print(f"✓ Extracted preference: {result} (from matches: {matches})")
                print("=" * 60)
            return result

        # Pattern 2: Fallback - look for these exact strings as complete words
        # Check in reverse order of specificity
        if re.search(r'\b-1\b', text):
            if debug:
                print("✓ Extracted preference: -1 (fallback method)")
                print("=" * 60)
            return -1
        elif re.search(r'\b0\b', text):
            if debug:
                print("✓ Extracted preference: 0 (fallback method)")
                print("=" * 60)
            return 0
        elif re.search(r'\b1\b', text):
            if debug:
                print("✓ Extracted preference: 1 (fallback method)")
                print("=" * 60)
            return 1
        else:
            # If no clear answer found, return -1 (uncertain)
            if debug:
                print(f"⚠ WARNING: Could not extract preference!")
                print(f"Returning -1 (uncertain)")
                print("=" * 60)
            return -1
    except Exception as e:
        print(f"❌ ERROR extracting preference: {e}")
        print("=" * 60)
        return -1

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_name': MODEL_NAME,
        'device': str(device)
    })

@app.route('/score', methods=['POST'])
def score_image():
    """
    Single image scoring endpoint (for direct reward like CLIP)

    Request JSON:
        {
            "image": "base64_encoded_image",
            "text": "The window is closed.",
            "temperature": 0
        }

    Response JSON:
        {
            "score": 0.85,
            "time": 5.23
        }
    """
    beg = time.time()

    try:
        init_model()

        data = request.json
        image_b64 = data.get('image')
        text = data.get('text')
        temperature = data.get('temperature', 0)

        if not image_b64 or not text:
            return jsonify({'error': 'Missing image or text'}), 400

        # Decode image
        image = decode_base64_image(image_b64)

        # Create prompt for scoring (similar to gemini_score_template)
        prompt = f"""1. What is shown in the image?
2. The goal is {text}. On a scale of 0 to 1, the score is 1 if the goal is achieved. What score would you give the image in terms of achieving the goal?"""

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
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
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

        # Extract score
        score = extract_number(response.strip())

        end = time.time()

        return jsonify({
            'score': score,
            'response': response.strip(),
            'time': end - beg
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/preference_one_stage', methods=['POST'])
def preference_one_stage():
    """
    One-stage preference judgment (similar to gemini_single_prompt)

    Request JSON:
        {
            "image1": "base64_encoded_image",
            "image2": "base64_encoded_image",
            "prompt": "The goal is to close the window...",
            "temperature": 0
        }

    Response JSON:
        {
            "preference": 0,  # 0, 1, or -1
            "response": "full response text",
            "time": 6.5
        }
    """
    beg = time.time()

    try:
        init_model()

        data = request.json
        image1_b64 = data.get('image1')
        image2_b64 = data.get('image2')
        prompt = data.get('prompt')
        temperature = data.get('temperature', 0)

        if not image1_b64 or not image2_b64 or not prompt:
            return jsonify({'error': 'Missing images or prompt'}), 400

        # Decode images
        image1 = decode_base64_image(image1_b64)
        image2 = decode_base64_image(image2_b64)

        # Prepare messages (similar to gemini format)
        full_prompt = f"""Consider the following two images:
Image 1:"""

        # DEBUG: Print the actual prompt being sent (only if QWEN_DEBUG=1)
        if DEBUG:
            print("\n" + "=" * 60)
            print("DEBUG: PROMPT SENT TO MODEL:")
            print("=" * 60)
            print(prompt)
            print("=" * 60 + "\n")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image", "image": image1},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image", "image": image2},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
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

        if DEBUG:
            print("\n" + "=" * 60)
            print("PREFERENCE ONE-STAGE - FULL MODEL RESPONSE:")
            print("=" * 60)
            print(response.strip())
            print("=" * 60)
            print(f"Extracting from last line: {response.strip().split(chr(10))[-1]}")
            print("=" * 60 + "\n")

        # Extract preference from last line (similar to gemini_query_1)
        preference = extract_preference(response.strip().split("\n")[-1], debug=DEBUG)

        end = time.time()

        return jsonify({
            'preference': preference,
            'response': response.strip(),
            'time': end - beg
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/preference_two_stage_analyze', methods=['POST'])
def preference_two_stage_analyze():
    """
    Two-stage preference: Stage 1 - Analyze images

    Request JSON:
        {
            "image1": "base64_encoded_image",
            "image2": "base64_encoded_image",
            "prompt": "1. What is shown in Image 1? ...",
            "temperature": 0
        }

    Response JSON:
        {
            "analysis": "detailed analysis text",
            "time": 6.2
        }
    """
    beg = time.time()

    try:
        init_model()

        data = request.json
        image1_b64 = data.get('image1')
        image2_b64 = data.get('image2')
        prompt = data.get('prompt')
        temperature = data.get('temperature', 0)

        if not image1_b64 or not image2_b64 or not prompt:
            return jsonify({'error': 'Missing images or prompt'}), 400

        # Decode images
        image1 = decode_base64_image(image1_b64)
        image2 = decode_base64_image(image2_b64)

        # Prepare messages
        full_prompt = f"""Consider the following two images:
Image 1:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image", "image": image1},
                    {"type": "text", "text": "Image 2:"},
                    {"type": "image", "image": image2},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Prepare inputs
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
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

        analysis = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if DEBUG:
            print("\n" + "=" * 60)
            print("PREFERENCE TWO-STAGE (ANALYZE) - FULL MODEL RESPONSE:")
            print("=" * 60)
            print(analysis.strip())
            print("=" * 60 + "\n")

        end = time.time()

        return jsonify({
            'analysis': analysis.strip(),
            'time': end - beg
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/preference_two_stage_extract', methods=['POST'])
def preference_two_stage_extract():
    """
    Two-stage preference: Stage 2 - Extract answer from analysis

    Request JSON:
        {
            "analysis": "text from stage 1",
            "summary_prompt": "Based on the text below...",
            "temperature": 0
        }

    Response JSON:
        {
            "preference": 0,  # 0, 1, or -1
            "response": "extracted answer",
            "time": 1.5
        }
    """
    beg = time.time()

    try:
        init_model()

        data = request.json
        analysis = data.get('analysis')
        summary_prompt = data.get('summary_prompt')
        temperature = data.get('temperature', 0)

        if not analysis or not summary_prompt:
            return jsonify({'error': 'Missing analysis or summary_prompt'}), 400

        # Format the summary prompt with analysis
        full_prompt = summary_prompt.format(analysis)

        # Prepare messages (text only, no images)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt}
                ]
            }
        ]

        # Prepare inputs
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
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

        if DEBUG:
            print("\n" + "=" * 60)
            print("PREFERENCE TWO-STAGE (EXTRACT) - FULL MODEL RESPONSE:")
            print("=" * 60)
            print(response.strip())
            print("=" * 60)
            print(f"Extracting from first line: {response.strip().split(chr(10))[0]}")
            print("=" * 60 + "\n")

        # Extract preference from first line (similar to gemini_query_2)
        preference = extract_preference(response.strip().split("\n")[0], debug=DEBUG)

        end = time.time()

        return jsonify({
            'preference': preference,
            'response': response.strip(),
            'time': end - beg
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Qwen3-VL HTTP Inference Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--preload', action='store_true', help='Preload model at startup')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (e.g., Qwen/Qwen3-VL-8B-Instruct or Qwen/Qwen3-VL-32B-Instruct). Overrides QWEN_MODEL env var.')

    args = parser.parse_args()

    # Override MODEL_NAME if --model is specified
    if args.model:
        MODEL_NAME = args.model
        print(f"Using model from command line: {args.model}")
    else:
        print(f"Using model: {MODEL_NAME} (from QWEN_MODEL env var or default)")

    # Optionally preload model
    if args.preload:
        print("Preloading model...")
        init_model()

    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Qwen3-VL HTTP Server")
    print(f"{'=' * 60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Model: {MODEL_NAME}")
    print(f"Debug: {'ON' if DEBUG else 'OFF'}")
    print(f"Endpoints:")
    print(f"  - POST /score (single image scoring)")
    print(f"  - POST /preference_one_stage (one-stage preference)")
    print(f"  - POST /preference_two_stage_analyze (two-stage: analyze)")
    print(f"  - POST /preference_two_stage_extract (two-stage: extract)")
    print(f"  - GET /health (health check)")
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

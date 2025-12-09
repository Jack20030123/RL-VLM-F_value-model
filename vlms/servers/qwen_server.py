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

# Global variables for model (lazy loading)
model = None
processor = None
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

def init_model():
    """Initialize Qwen3-VL model and processor"""
    global model, processor
    if model is None:
        print("=" * 60)
        print("Loading Qwen3-VL-8B-Instruct model...")
        print("=" * 60)
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Load model with automatic device management
        # device_map="auto" will offload some parameters to CPU if GPU memory is insufficient
        # This is necessary for 8GB GPUs (model needs ~8-9GB in bfloat16)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,  # Use bfloat16 to reduce memory usage
            device_map="auto",  # Auto manage device placement (GPU + CPU offload)
            low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        )

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct"
        )
        print("✓ Qwen3-VL model loaded successfully!")
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

def extract_preference(text):
    """Extract preference label (0, 1, or -1) from text response"""
    try:
        text = text.strip()
        # Look for -1, 0, 1 in the response
        if "-1" in text:
            return -1
        elif "0" in text:
            return 0
        elif "1" in text:
            return 1
        else:
            return -1
    except:
        return -1

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
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

        # Extract preference from last line (similar to gemini_query_1)
        preference = extract_preference(response.strip().split("\n")[-1])

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

        # Extract preference from first line (similar to gemini_query_2)
        preference = extract_preference(response.strip().split("\n")[0])

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

    args = parser.parse_args()

    # Optionally preload model
    if args.preload:
        print("Preloading model...")
        init_model()

    print(f"\n{'=' * 60}")
    print(f"🚀 Starting Qwen3-VL HTTP Server")
    print(f"{'=' * 60}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Endpoints:")
    print(f"  - POST /score (single image scoring)")
    print(f"  - POST /preference_one_stage (one-stage preference)")
    print(f"  - POST /preference_two_stage_analyze (two-stage: analyze)")
    print(f"  - POST /preference_two_stage_extract (two-stage: extract)")
    print(f"  - GET /health (health check)")
    print(f"{'=' * 60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

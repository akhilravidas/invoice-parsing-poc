"""
LLM helper
"""
import base64
import io
import json
import os
from typing import BinaryIO, List

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_bytes
from PIL import Image


def img2base64(image: Image.Image, image_format="PNG") -> str:
    """Convert PIL image to base64 string"""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=image_format)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    return img_base64


def pdf2img(pdf_contents) -> List[Image.Image]:
    """Convert PDF to PIL image"""
    return convert_from_bytes(pdf_contents)


def optimize_img(img):
    """Color PIL image -> Black and White"""
    return img.convert("1")


if "OPENAI_API_KEY" not in os.environ:
    load_dotenv()

assert "OPENAI_API_KEY" in os.environ, "No OPENAI Key"


PROMPT = """
You are an OCR agent, your task is to extract text information from images with accuracy and attention to detail. When encountering amounts of money, be careful to correctly identify the currency symbols and not to confuse commas with numbers. In cases where there are multiple totals, prioritize dollar amounts.

From the provided image, extract the information and return a JSON object with the following keys:

1. receiver_name: Name of the Buyer
2. supplier_name: Name of the Seller
3. invoice_total: Total amount of the Invoice, represented as a floating-point number
4. invoice_date: Date of the transaction, formatted as YYYY/MM/DD

Do not use markdown symbols like "```" or "```json" at the beginning or end of your response.
"""

client = OpenAI()


def parse_invoice_or_po(pages: List[Image.Image]) -> dict:
    """Extract information from invoice or PO using GPT-4 Vision"""
    b64_imgs = list(map(img2base64, pages))
    img_inputs = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        }
        for b64 in b64_imgs
    ]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}, *img_inputs],
            }
        ],
        max_tokens=4096,
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

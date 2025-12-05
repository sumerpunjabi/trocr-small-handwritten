#!/usr/bin/env python3
import sys
import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def main():
    img_path = "cheque.png"

    image = Image.open(img_path).convert("RGB")

    model_name = "microsoft/trocr-small-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name, use_fast=True, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=True)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(text)

if __name__ == "__main__":
    main()

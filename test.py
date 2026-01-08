from transformers import TrOCRProcessor, VisionEncoderDecoderModel

model_id = "microsoft/trocr-small-handwritten"
out_dir  = "./trocr-small-handwritten-local"

processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

processor.save_pretrained(out_dir)
model.save_pretrained(out_dir)

print("Saved to:", out_dir)
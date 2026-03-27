from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert('RGB')

    # ❌ DO NOT pass prompt here
    inputs = processor(image, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,   # improves quality
        do_sample=True,  # adds randomness
    )

    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
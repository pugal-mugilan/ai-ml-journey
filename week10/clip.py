"""
Week 10 Day 5 — CLIP: Images and Text in the Same Embedding Space
=================================================================
Two encoders (image + text) trained together so their output vectors
live in the same 512-dim space. Compare them with cosine similarity.

SETUP: pip install transformers torch torchvision Pillow requests
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw
import numpy as np

# ============================================================
# Step 1: Load CLIP model
# ============================================================
model_id = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

print(f"Model loaded: {model_id}")
print(f"Image encoder params: {sum(p.numel() for p in model.vision_model.parameters()):,}")
print(f"Text encoder params: {sum(p.numel() for p in model.text_model.parameters()):,}")
print(f"Embedding dimension: 512 (both encoders output 512-dim vectors)\n")

# ============================================================
# Step 2: Create test images using PIL (no downloads needed)
# ============================================================
# We'll make simple but distinct images — CLIP is smart enough
# to tell them apart even from basic shapes and colors.

def make_cat_image():
    """Draw a simple cat-like shape: circle head + triangle ears"""
    img = Image.new("RGB", (224, 224), (200, 180, 160))  # warm beige background
    draw = ImageDraw.Draw(img)
    # Body (oval)
    draw.ellipse([60, 90, 170, 200], fill=(255, 165, 80))  # orange body
    # Head (circle)
    draw.ellipse([75, 40, 155, 120], fill=(255, 165, 80))  # orange head
    # Ears (triangles)
    draw.polygon([(80, 55), (95, 20), (110, 55)], fill=(255, 140, 60))
    draw.polygon([(120, 55), (135, 20), (150, 55)], fill=(255, 140, 60))
    # Eyes
    draw.ellipse([95, 65, 110, 80], fill=(50, 200, 50))   # green eyes
    draw.ellipse([120, 65, 135, 80], fill=(50, 200, 50))
    # Nose
    draw.polygon([(112, 85), (108, 92), (118, 92)], fill=(255, 100, 100))
    # Whiskers
    draw.line([(70, 88), (105, 85)], fill=(80, 60, 40), width=1)
    draw.line([(70, 95), (105, 92)], fill=(80, 60, 40), width=1)
    draw.line([(125, 85), (160, 88)], fill=(80, 60, 40), width=1)
    draw.line([(125, 92), (160, 95)], fill=(80, 60, 40), width=1)
    return img

def make_car_image():
    """Draw a simple car shape"""
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # sky blue background
    draw = ImageDraw.Draw(img)
    # Road
    draw.rectangle([0, 160, 224, 224], fill=(80, 80, 80))
    draw.line([(0, 190), (224, 190)], fill=(255, 255, 255), width=2)
    # Car body
    draw.rectangle([40, 110, 190, 160], fill=(220, 30, 30))  # red car body
    # Car top
    draw.polygon([(70, 110), (90, 75), (160, 75), (170, 110)], fill=(200, 20, 20))
    # Windows
    draw.polygon([(95, 112), (100, 82), (130, 82), (130, 112)], fill=(180, 220, 255))
    draw.polygon([(135, 112), (135, 82), (158, 82), (165, 112)], fill=(180, 220, 255))
    # Wheels
    draw.ellipse([55, 145, 90, 180], fill=(30, 30, 30))
    draw.ellipse([63, 153, 82, 172], fill=(150, 150, 150))
    draw.ellipse([145, 145, 180, 180], fill=(30, 30, 30))
    draw.ellipse([153, 153, 172, 172], fill=(150, 150, 150))
    return img

def make_tree_image():
    """Draw a simple tree"""
    img = Image.new("RGB", (224, 224), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, 180, 224, 224], fill=(100, 180, 80))
    # Trunk
    draw.rectangle([100, 120, 125, 185], fill=(139, 90, 43))
    # Leaves (circles)
    draw.ellipse([60, 40, 165, 135], fill=(34, 139, 34))
    draw.ellipse([45, 60, 130, 140], fill=(0, 128, 0))
    draw.ellipse([95, 50, 180, 130], fill=(0, 100, 0))
    return img

images = {
    "cat_drawing": make_cat_image(),
    "red_car":     make_car_image(),
    "green_tree":  make_tree_image(),
}

# Save so you can see them
for name, img in images.items():
    path = f"{name}.png"
    img.save(path)
    print(f"Created image: {path} ({img.size})")

# ============================================================
# Step 3: Define text descriptions to compare against
# ============================================================
text_descriptions = [
    "a drawing of a cat with whiskers",
    "a photo of a dog",
    "a red car on a road",
    "a green tree with leaves",
    "a beautiful sunset over the ocean",
    "a person riding a bicycle",
]

# ============================================================
# Step 4: Compute similarity between EVERY image and EVERY text
# ============================================================
print("\n" + "=" * 62)
print("IMAGE-TEXT SIMILARITY MATRIX")
print("=" * 62)

for img_name, img in images.items():
    inputs = processor(
        text=text_descriptions,
        images=img,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image: cosine similarities scaled by learned temperature
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=0)

    print(f"\n--- Image: {img_name} ---")
    print(f"{'Text description':<42} {'Score':>8} {'Prob':>8}")
    print("-" * 60)

    sorted_indices = logits.argsort(descending=True)
    for idx in sorted_indices:
        text = text_descriptions[idx]
        score = logits[idx].item()
        prob = probs[idx].item()
        marker = " << BEST" if idx == sorted_indices[0] else ""
        print(f"{text:<42} {score:>8.2f} {prob:>7.1%}{marker}")

# ============================================================
# Step 5: Under the hood — raw embedding vectors
# ============================================================
print("\n" + "=" * 62)
print("UNDER THE HOOD: RAW EMBEDDING VECTORS")
print("=" * 62)

img_list = list(images.values())
img_inputs = processor(images=img_list, return_tensors="pt")
with torch.no_grad():
    img_embeds = model.get_image_features(**img_inputs)

txt_inputs = processor(text=text_descriptions[:3], return_tensors="pt", padding=True)
with torch.no_grad():
    txt_embeds = model.get_text_features(**txt_inputs)

# Normalize so dot product = cosine similarity
img_embeds = model.get_image_features(**img_inputs)
if not isinstance(img_embeds, torch.Tensor):
    img_embeds = img_embeds.pooler_output
txt_embeds = model.get_text_features(**txt_inputs)
if not isinstance(txt_embeds, torch.Tensor):
    txt_embeds = txt_embeds.pooler_output

print(f"\nImage embedding shape: {img_embeds.shape}  (num_images x 512)")
print(f"Text embedding shape:  {txt_embeds.shape}   (num_texts x 512)")

print(f"\nFirst 8 values of cat image vector:    {img_embeds[0, :8].numpy().round(3)}")
print(f"First 8 values of 'cat with whiskers': {txt_embeds[0, :8].numpy().round(3)}")

# Manual cosine similarity
cos_sim = (img_embeds @ txt_embeds.T).numpy()
img_names = list(images.keys())
print(f"\nManual cosine similarity (images x texts):")
for i, img_name in enumerate(img_names):
    for j, text in enumerate(text_descriptions[:3]):
        print(f"  {img_name:<14} vs '{text}': {cos_sim[i, j]:.4f}")

# ============================================================
# Step 6: Zero-shot classification
# ============================================================
print("\n" + "=" * 62)
print("ZERO-SHOT CLASSIFICATION")
print("=" * 62)
print("Classify into categories the model was NEVER specifically")
print("trained on. Just describe the categories in text.\n")

categories = ["an animal", "a vehicle", "a plant", "a building", "food"]

for img_name, img in images.items():
    inputs = processor(
        text=[f"a photo of {c}" for c in categories],
        images=img,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image[0].softmax(dim=0)
    best_idx = probs.argmax().item()

    print(f"Image '{img_name}' → {categories[best_idx]} ({probs[best_idx]:.1%})")
    for i, cat in enumerate(categories):
        bar = "█" * int(probs[i] * 30)
        print(f"  {cat:<16} {probs[i]:>6.1%} {bar}")
    print()
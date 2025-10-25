import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from src import model  # Ensure your 'src/model.py' defines Harmonizer


# ----------------------------- Harmonizer Model Function (GPU) -----------------------------
def harmonize_image_gpu(composite_path, mask_path, pretrained='', save_path='harmonized.png'):
    """
    Harmonize a single composite image using a deep learning model only (no classical blending).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Check paths ---
    for p, name in [(composite_path, "Composite"), (mask_path, "Mask"), (pretrained, "Pretrained model")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # --- Load images ---
    comp = Image.open(composite_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # --- Load model ---
    print("Loading Harmonizer model...")
    harmonizer = model.Harmonizer().to(device)
    harmonizer.load_state_dict(torch.load(pretrained, map_location=device), strict=True)
    harmonizer.eval()

    # --- Prepare tensors ---
    comp_tensor = tf.to_tensor(comp).unsqueeze(0).to(device)
    mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)

    # --- Run harmonization ---
    print("Running harmonization (deep model only)...")
    with torch.no_grad():
        args = harmonizer.predict_arguments(comp_tensor, mask_tensor)
        harmonized = harmonizer.restore_image(comp_tensor, mask_tensor, args)[-1]

    # --- Convert tensor to image ---
    harmonized_np = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))
    print("Harmonization complete (deep model output).")

    # --- Save result ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    harmonized_img.save(save_path)
    print(f"Deep harmonized image saved to: {save_path}")

    return harmonized_img


# ----------------------------- Harmonizer Model Function (CPU) -----------------------------
def harmonize_image_cpu(composite_path, mask_path, pretrained='', save_path='harmonized_cpu.png'):
    """
    Harmonize a single composite image using a deep learning model on CPU.
    """
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Check paths ---
    for p, name in [(composite_path, "Composite"), (mask_path, "Mask"), (pretrained, "Pretrained model")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} not found: {p}")

    # --- Load images ---
    comp = Image.open(composite_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # --- Load model ---
    print("Loading Harmonizer model (CPU mode)...")
    harmonizer = model.Harmonizer().to(device)
    state_dict = torch.load(pretrained, map_location=device)
    harmonizer.load_state_dict(state_dict, strict=True)
    harmonizer.eval()

    # --- Prepare tensors ---
    comp_tensor = tf.to_tensor(comp).unsqueeze(0).to(device)
    mask_tensor = tf.to_tensor(mask).unsqueeze(0).to(device)

    # --- Run harmonization ---
    print("Running harmonization (CPU, deep model only)...")
    with torch.no_grad():
        args = harmonizer.predict_arguments(comp_tensor, mask_tensor)
        harmonized = harmonizer.restore_image(comp_tensor, mask_tensor, args)[-1]

    # --- Convert tensor to image ---
    harmonized_np = np.transpose(harmonized[0].cpu().numpy(), (1, 2, 0)) * 255
    harmonized_img = Image.fromarray(harmonized_np.astype(np.uint8))
    print("Harmonization complete (CPU deep model output).")

    # --- Save result ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    harmonized_img.save(save_path)
    print(f"Deep harmonized image saved to: {save_path}")

    return harmonized_img


# ----------------------------- Example Usage -----------------------------
if __name__ == "__main__":
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths
    composite_image = os.path.join(BASE_DIR, "capture_result", "composite.tiff")
    mask_image = os.path.join(BASE_DIR, "capture_result", "mask.jpg")
    pretrained_model = os.path.join(BASE_DIR, "pretrained", "harmonizer.pth")

    output_path = os.path.join(BASE_DIR, "demo", "image_harmonization", "example", "harmonized_deep_only.tiff")

    # Run the CPU version for portability
    harmonize_image_cpu(composite_image, mask_image, pretrained_model, output_path)

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

from models.bemunet.bemunet import BEMUNet

# ===================== CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VM2_DIR = os.path.join(os.path.dirname(BASE_DIR), 'VM2-UNet')

MODEL_CONFIGS = {
    'ISIC2017': {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'input_size': (256, 256),
        'ckpt_path': os.path.join(BASE_DIR, 'ISIC2017', 'best-epoch39-miou0.8065.pth'),
        'norm_mean': 148.429,
        'norm_std': 25.748,
        'description': 'Skin Lesion Segmentation (ISIC 2017)',
        'metrics': {
            'mIoU':  0.8065,
            'DSC':   0.8930,
            'Acc':   0.9650,
            'Spe':   0.9810,
            'Sen':   0.8835,
        },
    },
    'ISIC2018': {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'input_size': (256, 256),
        'ckpt_path': os.path.join(BASE_DIR, 'ISIC2018', 'best-epoch212-miou0.8142.pth'),
        'norm_mean': 149.034,
        'norm_std': 32.022,
        'description': 'Skin Lesion Segmentation (ISIC 2018)',
        'metrics': {
            'mIoU':  0.8142,
            'DSC':   0.8976,
            'Acc':   0.9512,
            'Spe':   0.9750,
            'Sen':   0.8780,
        },
    },
    'Synapse': {
        'num_classes': 9,
        'input_channels': 3,
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'input_size': (224, 224),
        'ckpt_path': os.path.join(BASE_DIR, 'SYNAPSE', 'best-epoch263-mean_dice0.8493-mean_hd9512.2378.pth'),
        'description': 'Multi-Organ Segmentation (Synapse)',
        'metrics': {
            'Mean Dice': 0.8493,
            'Mean HD95': 12.2378,
        },
    },
}

SYNAPSE_CLASSES = {
    0: 'Background',
    1: 'Aorta',
    2: 'Gallbladder',
    3: 'Left Kidney',
    4: 'Right Kidney',
    5: 'Liver',
    6: 'Pancreas',
    7: 'Spleen',
    8: 'Stomach',
}

SYNAPSE_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0],
    5: [255, 0, 255],
    6: [0, 255, 255],
    7: [255, 128, 0],
    8: [128, 0, 255],
}

# ===================== MODEL LOADING =====================
# cuDNN 9.1.x chưa hỗ trợ đầy đủ Blackwell (RTX 50xx), disable để fallback sang native CUDA kernels
torch.backends.cudnn.enabled = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_models = {}


def load_model(dataset_name):
    if dataset_name in loaded_models:
        return loaded_models[dataset_name]

    cfg = MODEL_CONFIGS[dataset_name]
    model = BEMUNet(
        input_channels=cfg['input_channels'],
        num_classes=cfg['num_classes'],
        depths=cfg['depths'],
        depths_decoder=cfg['depths_decoder'],
        drop_path_rate=cfg['drop_path_rate'],
        load_ckpt_path=None,
    )

    checkpoint = torch.load(cfg['ckpt_path'], map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    loaded_models[dataset_name] = model
    print(f"[OK] Loaded {dataset_name} model on {device}")
    return model


# ===================== PREPROCESSING =====================
def preprocess_isic(image_np, cfg):
    img = image_np.astype(np.float64)
    img_normalized = (img - cfg['norm_mean']) / cfg['norm_std']
    img_normalized = ((img_normalized - img_normalized.min())
                      / (img_normalized.max() - img_normalized.min() + 1e-8)) * 255.0
    tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    h, w = cfg['input_size']
    tensor = F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
    return tensor


def preprocess_synapse(image_np, cfg):
    if len(image_np.shape) == 2:
        img = np.stack([image_np] * 3, axis=-1)
    else:
        img = image_np
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    h, w = cfg['input_size']
    tensor = F.interpolate(tensor, size=(h, w), mode='bilinear', align_corners=False)
    return tensor


# ===================== INFERENCE =====================
def predict(image, dataset_name):
    if image is None:
        return None, "Please upload an image."

    cfg = MODEL_CONFIGS[dataset_name]
    model = load_model(dataset_name)
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    if dataset_name in ['ISIC2017', 'ISIC2018']:
        tensor = preprocess_isic(image_np, cfg)
    else:
        tensor = preprocess_synapse(image_np, cfg)

    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    if cfg['num_classes'] == 1:
        mask = output.squeeze().cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8)
        mask_resized = np.array(Image.fromarray(mask_binary).resize((orig_w, orig_h), Image.NEAREST))

        overlay = image_np.copy()
        overlay[mask_resized == 1] = (
            overlay[mask_resized == 1] * 0.5 +
            np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)

        confidence = float(mask.mean())
        info = f"**Dataset:** {cfg['description']}\n\n"
        info += f"**Lesion Coverage:** {mask_resized.mean() * 100:.1f}%\n\n"
        info += f"**Mean Confidence:** {confidence:.4f}\n\n"
        info += "---\n**Model Performance:**\n\n"
        for k, v in cfg['metrics'].items():
            info += f"- **{k}:** {v}\n"

        return overlay, info

    else:
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        pred_resized = np.array(
            Image.fromarray(pred.astype(np.uint8)).resize((orig_w, orig_h), Image.NEAREST)
        )

        color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for cls_id, color in SYNAPSE_COLORS.items():
            color_mask[pred_resized == cls_id] = color

        overlay = image_np.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        non_bg = pred_resized > 0
        overlay[non_bg] = (
            overlay[non_bg] * 0.4 + color_mask[non_bg] * 0.6
        ).astype(np.uint8)

        info = f"**Dataset:** {cfg['description']}\n\n"
        info += "**Detected Organs:**\n\n"
        unique_classes = np.unique(pred_resized)
        for cls_id in unique_classes:
            if cls_id == 0:
                continue
            ratio = (pred_resized == cls_id).mean() * 100
            info += f"- **{SYNAPSE_CLASSES.get(cls_id, f'Class {cls_id}')}:** {ratio:.1f}%\n"

        info += "\n---\n**Model Performance:**\n\n"
        for k, v in cfg['metrics'].items():
            info += f"- **{k}:** {v}\n"

        return overlay, info


# ===================== METRICS DISPLAY =====================
def get_metrics_html(dataset_name):
    cfg = MODEL_CONFIGS[dataset_name]

    metrics_rows = ""
    for k, v in cfg['metrics'].items():
        display = f"{v * 100:.2f}%" if isinstance(v, float) and v < 1 else str(v)
        metrics_rows += f"""
            <div style="display:flex; justify-content:space-between; align-items:baseline;
                        padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                <span style="color:#555; font-size:0.875em;">{k}</span>
                <span style="font-size:1.05em; font-weight:600; color:#1a1a1a;">{display}</span>
            </div>
        """

    if dataset_name in ['ISIC2017', 'ISIC2018']:
        extra = "<p style='margin:10px 0 0; color:#888; font-size:0.8em;'>Binary segmentation · 256×256 input</p>"
    else:
        legend_items = ""
        for cls_id, cls_name in SYNAPSE_CLASSES.items():
            if cls_id == 0:
                continue
            r, g, b = SYNAPSE_COLORS[cls_id]
            legend_items += f"""
                <div style="display:flex; align-items:center; gap:6px; padding:2px 0;">
                    <div style="width:9px; height:9px; border-radius:2px; flex-shrink:0;
                                background:rgb({r},{g},{b});"></div>
                    <span style="color:#555; font-size:0.8em;">{cls_name}</span>
                </div>
            """
        extra = f"""
            <div style="margin-top:12px;">
                <p style="margin:0 0 7px; color:#888; font-size:0.8em;">Multi-organ · 224×224 input · 8 classes</p>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:3px 10px;">
                    {legend_items}
                </div>
            </div>
        """

    return f"""
    <div style="background:#fff; border:1px solid #e5e7eb; border-radius:10px;
                padding:16px 18px; font-family:system-ui,sans-serif;">
        <p style="margin:0 0 10px; font-size:0.75em; text-transform:uppercase;
                  letter-spacing:0.8px; color:#9ca3af; font-weight:500;">Performance</p>
        {metrics_rows}
        {extra}
    </div>
    """


def on_dataset_change(dataset_name):
    return get_metrics_html(dataset_name)


# ===================== GRADIO UI =====================
CUSTOM_CSS = """
.gradio-container { max-width: 1160px !important; }
footer { display: none !important; }

#run-btn {
    background: #2563eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
}
#run-btn:hover { background: #1d4ed8 !important; }
"""

with gr.Blocks(
    css=CUSTOM_CSS,
    title="BEM-UNet Segmentation",
    theme=gr.themes.Base(
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
    ).set(
        body_background_fill="#f9fafb",
        block_background_fill="#ffffff",
        block_border_color="#e5e7eb",
        block_border_width="1px",
        block_radius="10px",
        block_label_text_color="#6b7280",
        block_label_text_size="0.78em",
        input_background_fill="#ffffff",
        button_primary_background_fill="#2563eb",
        button_primary_text_color="#ffffff",
    ),
) as demo:

    # Header
    gr.HTML("""
    <div style="padding: 28px 8px 20px; font-family: system-ui, sans-serif;">
        <h1 style="margin: 0 0 4px; font-size: 1.5em; font-weight: 700;
                   color: #111827; letter-spacing: -0.3px;">
            BEM-UNet
        </h1>
        <p style="margin: 0; color: #6b7280; font-size: 0.9em;">
            Boundary-Enhanced Medical Image Segmentation
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            dataset_selector = gr.Radio(
                choices=['Synapse', 'ISIC2017', 'ISIC2018'],
                value='Synapse',
                label="Model",
                info="Synapse: multi-organ CT    ISIC: skin lesion",
            )
            metrics_display = gr.HTML(value=get_metrics_html('Synapse'))
            input_image = gr.Image(type="pil", label="Input image", height=240, sources=["upload"])
            predict_btn = gr.Button("Run segmentation", variant="primary", size="lg", elem_id="run-btn")

        with gr.Column(scale=6):
            output_image = gr.Image(type="numpy", label="Result", height=420)
            output_info = gr.Markdown(value="Upload an image and press **Run segmentation**.")

    dataset_selector.change(fn=on_dataset_change, inputs=dataset_selector, outputs=metrics_display)
    predict_btn.click(fn=predict, inputs=[input_image, dataset_selector], outputs=[output_image, output_info])

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)

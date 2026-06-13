import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import random

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🐠 Coral Reef Classifier",
    page_icon="🐠",
    layout="wide"
)

st.markdown("<h1 style='text-align: center; color: teal;'>🐠 Coral Reef Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select a model and upload an image to predict its class.</p>", unsafe_allow_html=True)
st.divider()

# -------------------------
# Dataset directories
# -------------------------
DATA_DIR = "Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Safety checks
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    st.error("❌ Dataset folders not found. Ensure Dataset/train and Dataset/val exist.")
    st.stop()

class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
num_classes = len(class_names)

if num_classes == 0:
    st.error("❌ No class folders found in Dataset/train.")
    st.stop()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🧠 Coral Reef AI")
st.sidebar.caption("Deep Learning Image Classifier")
st.sidebar.divider()

# Device selection
device_option = st.sidebar.radio("🖥️ Compute Device", ["Auto", "CPU", "GPU"], horizontal=True)
if device_option == "CPU":
    device = torch.device("cpu")
elif device_option == "GPU":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    torch.backends.cudnn.enabled = False

st.sidebar.success(f"Using `{device}`")

# -------------------------
# Model selection
# -------------------------
model_options = [
    "MobileNetV2",
    "MobileNetV3-Large",
    "EfficientNet-B0",
    "ResNet-50",
    "ConvNeXt",
    "Densenet121",
    "ViT"
]

selected_model_name = st.sidebar.selectbox("🤖 Model Architecture", model_options)
st.sidebar.divider()

# Image source
image_source = st.sidebar.radio("🖼️ Image Source", ["Upload Image", "Random From Validation"])
selected_class = None
if image_source == "Random From Validation":
    selected_class = st.sidebar.selectbox("Select Coral Class", class_names)

# -------------------------
# Model paths
# -------------------------
model_paths = {
    "MobileNetV2": "saved_models/mobilenet_v2_best.pth",
    "MobileNetV3-Large": "saved_models/mobilenet_v3_large_best.pth",
    "EfficientNet-B0": "saved_models/efficientnet_b0_best.pth",
    "ResNet-50": "saved_models/resnet50_best.pth",
    "ConvNeXt": "saved_models/convnext_base_best.pth",
    "Densenet121": "saved_models/densenet121_best.pth",
    "ViT": "saved_models/VIT_best.pth"
}

# -------------------------
# Model loaders
# -------------------------
def load_mobilenet_v2(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_mobilenet_v3(path):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_efficientnet(path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_resnet50(path):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_convnext(path):
    model = models.convnext_base(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_densenet121(path):
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def load_vit(path):
    model = models.vit_b_16(weights=None)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

# -------------------------
# Cached model loader
# -------------------------
@st.cache_resource
def load_model(name):
    path = model_paths[name]
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        st.stop()
    loaders = {
        "MobileNetV2": load_mobilenet_v2,
        "MobileNetV3-Large": load_mobilenet_v3,
        "EfficientNet-B0": load_efficientnet,
        "ResNet-50": load_resnet50,
        "ConvNeXt": load_convnext,
        "Densenet121": load_densenet121,
        "ViT": load_vit
    }
    return loaders[name](path)

model = load_model(selected_model_name)

# -------------------------
# Image transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Prediction function
# -------------------------
def predict_image(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)
    k = min(5, num_classes)
    topk_prob, topk_idx = torch.topk(probs, k)
    topk = [(class_names[topk_idx[0][i]], topk_prob[0][i].item()) for i in range(k)]
    return class_names[pred.item()], conf.item(), topk

# -------------------------
# Random image picker
# -------------------------
def get_random_image(class_name):
    class_path = os.path.join(VAL_DIR, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        return None, None
    img_file = random.choice(images)
    return Image.open(os.path.join(class_path, img_file)).convert("RGB"), img_file

# -------------------------
# Image input
# -------------------------
image = None
image_path = None

st.subheader("📥 Input Image")

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload coral reef image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
else:
    if st.button("🎲 Pick Random Image"):
        image, image_path = get_random_image(selected_class)
        if image is None:
            st.warning("No images found in this class.")

# -------------------------
# Prediction and display
# -------------------------
if image is not None:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption=image_path or "Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            pred_class, confidence, topk = predict_image(image)

        st.markdown("### ✅ Prediction Result")
        st.metric(label="Predicted Class", value=pred_class, delta=f"{confidence*100:.2f}% confidence")

        st.markdown("### 📊 Top Predictions")
        df = pd.DataFrame(topk, columns=["Class", "Probability"])
        chart_df = pd.DataFrame({row[0]: row[1] for row in topk}, index=[0]).T
        chart_df.columns = ["Probability"]
        st.bar_chart(chart_df)
        df["Probability"] = df["Probability"].apply(lambda x: f"{x*100:.2f}%")
        st.table(df)

# -------------------------
# Model Info Panel
# -------------------------
MODEL_INFO = {
    "MobileNetV2": "Lightweight & fast (mobile-friendly)",
    "MobileNetV3-Large": "Improved accuracy with low latency",
    "EfficientNet-B0": "Balanced depth, width, resolution",
    "ResNet-50": "Deep residual learning",
    "ConvNeXt": "Modern CNN inspired by Transformers",
    "Densenet121": "Strong feature reuse",
    "ViT": "Vision Transformer (attention-based model)"
}

with st.expander("ℹ️ Model Details"):
    st.write(f"**Architecture:** {selected_model_name}")
    st.write(f"**Description:** {MODEL_INFO[selected_model_name]}")
    st.write(f"**Number of Classes:** {num_classes}")
    st.write(f"**Device:** {device}")

# -------------------------
# Footer
# -------------------------
st.divider()
st.markdown("<p style='text-align:center;color:gray;'>Coral Reef Image Classifier • PyTorch & Streamlit • Made with ❤️</p>", unsafe_allow_html=True)

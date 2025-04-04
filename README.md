# 🖼️ Stable Diffusion Text-to-Image Generator

This is an end-to-end Streamlit application for generating images from text prompts using the **Stable Diffusion** model. Built with `diffusers` from Hugging Face, the app allows users to input custom prompts, adjust generation parameters, and download the generated images.

---

## 🚀 Features

- Generate images using Stable Diffusion from Hugging Face 🤗
- Adjustable parameters: guidance scale, steps, and image dimensions
- Save or download generated images
- Clean and interactive Streamlit UI

---

## 📁 Project Structure

```
.gitattributes
.gitignore
LICENSE
README.md
app.py
requirements.txt
```

## 🌐 Live Demo

Check out the live version of this project here:  
👉 **[Access the App](https://huggingface.co/spaces/Knight090/TextToImage)**

> *Note: It may take a few seconds to load due to model initialization.*

---

## 🛠️ Getting Started

Follow these steps to clone the project, set up a virtual environment, and run the app locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stable-diffusion-text2image.git
cd stable-diffusion-text2image
```

### 2. Create a Virtual Environment

- On **Windows**:

```bash
python -m venv venv
```

- On **macOS/Linux**:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

- On **Windows**:

```bash
venv\Scripts\activate
```

- On **macOS/Linux**:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

----

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).

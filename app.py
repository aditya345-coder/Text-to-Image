# import streamlit as st
# import torch
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler
# import io
# from PIL import Image
# import base64
# import time
# import uuid
# import os

# # Set page configuration
# st.set_page_config(
#     page_title="Text to Image Generator",
#     page_icon="🎨",
#     layout="wide"
# )

# # Create directory for saved images if it doesn't exist
# if not os.path.exists("generated_images"):
#     os.makedirs("generated_images")

# # Define styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1E3A8A;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #2563EB;
#         margin-top: 1rem;
#     }
#     .parameter-section {
#         background-color: #F3F4F6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#     }
#     .image-section {
#         background-color: #EFF6FF;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         margin-top: 1rem;
#     }
#     .footer {
#         margin-top: 2rem;
#         font-size: 0.8rem;
#         color: #6B7280;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Header
# st.markdown('<p class="main-header">Text to Image Generator</p>', unsafe_allow_html=True)
# st.markdown("Transform your ideas into images.")

# # Sidebar for model selection and parameters
# st.sidebar.markdown('<p class="sub-header">Model Selection</p>', unsafe_allow_html=True)

# # Model selection dropdown
# model_options = {
#     "Realistic Images": "runwayml/stable-diffusion-v1-5",
#     "Anime Style": "Linaqruf/anything-v3.0",
#     "Dreamlike Art": "dreamlike-art/dreamlike-diffusion-1.0",
#     "Fantasy Art": "prompthero/openjourney"
# }

# selected_model = st.sidebar.selectbox("Choose a model style", list(model_options.keys()))
# model_id = model_options[selected_model]

# # Display model info
# model_descriptions = {
#     "Realistic Images": "General purpose model that produces realistic images.",
#     "Anime Style": "Specialized in anime and manga-style art.",
#     "Dreamlike Art": "Creates surreal, dreamlike imagery.",
#     "Fantasy Art": "Fantasy-oriented model inspired by Midjourney."
# }

# st.sidebar.info(model_descriptions[selected_model])

# # Advanced parameters section
# st.sidebar.markdown('<p class="sub-header">Advanced Parameters</p>', unsafe_allow_html=True)

# with st.sidebar.expander("Sampling Parameters", expanded=True):
#     # Sampling method
#     scheduler_options = {
#         "DPM++ 2M": DPMSolverMultistepScheduler,
#         "Euler": EulerDiscreteScheduler,
#         "DDIM": DDIMScheduler
#     }
#     scheduler_name = st.selectbox("Sampling method", list(scheduler_options.keys()))
    
#     # Number of inference steps
#     num_inference_steps = st.slider("Number of inference steps", 20, 100, 50, 
#                                   help="Higher values = more detail but slower generation")
    
#     # Guidance scale
#     guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5, 
#                              help="How closely to follow the prompt (higher = more adherence)")

# with st.sidebar.expander("Image Parameters", expanded=True):
#     # Image dimensions
#     height_options = [512, 576, 640, 704, 768]
#     width_options = [512, 576, 640, 704, 768]
    
#     height = st.select_slider("Image height", options=height_options, value=512)
#     width = st.select_slider("Image width", options=width_options, value=512)
    
#     # Seed for reproducibility
#     use_random_seed = st.checkbox("Use random seed", value=True)
#     if use_random_seed:
#         seed = int(time.time())
#     else:
#         seed = st.number_input("Seed", value=42, min_value=0, max_value=2147483647)

# # Main content area
# st.markdown('<p class="sub-header">Generate Your Image</p>', unsafe_allow_html=True)

# # Prompt input
# prompt = st.text_area("Enter your prompt", 
#                          placeholder="A majestic castle on a floating island, fantasy art, detailed, 4k, trending on artstation",
#                          height=100)


# # Negative prompt input
# negative_prompt = st.text_input("Negative prompt (things to avoid)", 
#                               placeholder="blurry, low quality, distorted, deformed, ugly",
#                               help="Elements you want the model to avoid")

# # Generation button
# generate_button = st.button("Generate Image", type="primary", use_container_width=True)

# # Function to generate image
# @st.cache_resource
# def load_model(model_id, scheduler_class):
#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         safety_checker=None,
#         requires_safety_checker=False
#     )
#     pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
#     pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
#     return pipe

# def generate_image():
#     # Show loading spinner
#     with st.spinner(f"Generating your image with {selected_model}..."):
#         # Set the seed for reproducibility
#         torch.manual_seed(seed)
        
#         # Load model
#         pipe = load_model(model_id, scheduler_options[scheduler_name])
        
#         # Generate image
#         image = pipe(
#             prompt=prompt,
#             negative_prompt=negative_prompt,
#             height=height,
#             width=width,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale,
#         ).images[0]
        
#         return image

# # Display the image and parameters when generated
# if generate_button and prompt:
#     try:
#         # Generate image
#         image = generate_image()
        
#         # Display image
#         st.markdown('<div class="image-section">', unsafe_allow_html=True)
#         st.markdown("### Generated Image")
#         st.image(image, use_column_width=True)
        
#         # Save image functionality
#         img_filename = f"generated_{uuid.uuid4().hex[:8]}.png"
#         img_path = os.path.join("generated_images", img_filename)
#         image.save(img_path)
        
#         # Download button
#         with open(img_path, "rb") as img_file:
#             btn = st.download_button(
#                 label="Download Image",
#                 data=img_file,
#                 file_name=img_filename,
#                 mime="image/png"
#             )
        
#         # Display parameters used
#         st.markdown("### Generation Parameters")
#         st.markdown(f"""
#         - **Model**: {selected_model} ({model_id})
#         - **Prompt**: {prompt}
#         - **Negative Prompt**: {negative_prompt if negative_prompt else "None"}
#         - **Sampler**: {scheduler_name}
#         - **Steps**: {num_inference_steps}
#         - **Guidance Scale**: {guidance_scale}
#         - **Dimensions**: {width}x{height}
#         - **Seed**: {seed}
#         """)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#     except Exception as e:
#         st.error(f"Error generating image: {str(e)}")

# # Add explanation section
# with st.expander("Key Parameter Explanation", expanded=False):
#     st.markdown("""
#     - **Sampling Method**: Different algorithms for the denoising process
#         - DPM++ 2M: Fast with good quality
#         - Euler: Good balance between speed and quality
#         - DDIM: More deterministic results
        
#     - **Steps**: More steps = more refined details but slower generation
    
#     - **Guidance Scale**: Controls how closely the model follows your prompt
#         - Higher values: More literal interpretation but may look unnatural
#         - Lower values: More creative but might not follow prompt exactly
    
#     - **Seed**: Controls the initial random noise pattern
#         - Same seed + same parameters = reproducible results
#     """)

# =====================================================
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DDIMScheduler
import io
from PIL import Image
import base64
import time
import uuid
import os

# Set page configuration
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Create directory for saved images if it doesn't exist
if not os.path.exists("generated_images"):
    os.makedirs("generated_images")

# Define styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2563EB;
        margin-top: 1rem;
    }
    .parameter-section {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .image-section {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .footer {
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Text to Image Generator</p>', unsafe_allow_html=True)
st.markdown("Transform your ideas into images using state-of-the-art Stable Diffusion models.")

# Sidebar for model selection and parameters
st.sidebar.markdown('<p class="sub-header">Model Selection</p>', unsafe_allow_html=True)

# Model selection dropdown
model_options = {
    "Realistic Images": "runwayml/stable-diffusion-v1-5",
    "Anime Style": "Linaqruf/anything-v3.0",
    "Dreamlike Art": "dreamlike-art/dreamlike-diffusion-1.0",
    "Fantasy Art": "prompthero/openjourney"
}

selected_model = st.sidebar.selectbox("Choose a model style", list(model_options.keys()))
model_id = model_options[selected_model]

# Display model info
model_descriptions = {
    "Realistic Images": "General purpose model that produces realistic images.",
    "Anime Style": "Specialized in anime and manga-style art.",
    "Dreamlike Art": "Creates surreal, dreamlike imagery.",
    "Fantasy Art": "Fantasy-oriented model inspired by Midjourney."
}

st.sidebar.info(model_descriptions[selected_model])

# Advanced parameters section
st.sidebar.markdown('<p class="sub-header">Advanced Parameters</p>', unsafe_allow_html=True)

with st.sidebar.expander("Sampling Parameters", expanded=True):
    # Sampling method
    scheduler_options = {
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "Euler": EulerDiscreteScheduler,
        "DDIM": DDIMScheduler
    }
    scheduler_name = st.selectbox("Sampling method", list(scheduler_options.keys()))
    
    # Update minimum inference steps based on scheduler
    # Some schedulers have minimum step requirements
    min_steps = 21 if scheduler_name == "DPM++ 2M" else 20
    
    # Number of inference steps
    num_inference_steps = st.slider("Number of inference steps", min_steps, 100, max(30, min_steps), 
                                  help="Higher values = more detail but slower generation")
    
    # Guidance scale
    guidance_scale = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5, 
                             help="How closely to follow the prompt (higher = more adherence)")

with st.sidebar.expander("Image Parameters", expanded=True):
    # Image dimensions
    height_options = [512, 576, 640, 704, 768]
    width_options = [512, 576, 640, 704, 768]
    
    height = st.select_slider("Image height", options=height_options, value=512)
    width = st.select_slider("Image width", options=width_options, value=512)
    
    # Seed for reproducibility
    use_random_seed = st.checkbox("Use random seed", value=True)
    if use_random_seed:
        seed = int(time.time())
    else:
        seed = st.number_input("Seed", value=42, min_value=0, max_value=2147483647)

# Main content area
st.markdown('<p class="sub-header">Generate Your Image</p>', unsafe_allow_html=True)

# Prompt input
prompt = st.text_area("Enter your prompt", 
                         placeholder="A majestic castle on a floating island, fantasy art, detailed, 4k, trending on artstation",
                         height=100)


# Negative prompt input
negative_prompt = st.text_input("Negative prompt (things to avoid)", 
                              placeholder="blurry, low quality, distorted, deformed, ugly",
                              help="Elements you want the model to avoid")

# Generation button
generate_button = st.button("Generate Image", type="primary", use_container_width=True)

# Function to generate image
@st.cache_resource
def load_model(model_id, scheduler_class):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_image():
    # Show loading spinner
    with st.spinner(f"Generating your image with {selected_model}..."):
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        
        # Load model
        pipe = load_model(model_id, scheduler_options[scheduler_name])
        
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
        return image

# Display the image and parameters when generated
if generate_button and prompt:
    try:
        # Generate image
        image = generate_image()
        
        # Display image
        st.markdown('<div class="image-section">', unsafe_allow_html=True)
        st.markdown("### Generated Image")
        st.image(image, use_column_width=True)
        
        # Save image functionality
        img_filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        img_path = os.path.join("generated_images", img_filename)
        image.save(img_path)
        
        # Download button
        with open(img_path, "rb") as img_file:
            btn = st.download_button(
                label="Download Image",
                data=img_file,
                file_name=img_filename,
                mime="image/png"
            )
        
        # Display parameters used
        st.markdown("### Generation Parameters")
        st.markdown(f"""
        - **Model**: {selected_model} ({model_id})
        - **Prompt**: {prompt}
        - **Negative Prompt**: {negative_prompt if negative_prompt else "None"}
        - **Sampler**: {scheduler_name}
        - **Steps**: {num_inference_steps}
        - **Guidance Scale**: {guidance_scale}
        - **Dimensions**: {width}x{height}
        - **Seed**: {seed}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        st.warning("If you're getting an index out of bounds error, try increasing the number of inference steps.")

# Add explanation section
with st.expander("Key Parameters Explained", expanded=False):
    st.markdown("""    
    - **Sampling Method**: Different algorithms for the denoising process
        - DPM++ 2M: Fast with good quality (requires minimum 21 steps)
        - Euler: Good balance between speed and quality
        - DDIM: More deterministic results
        
    - **Steps**: More steps = more refined details but slower generation
    
    - **Guidance Scale**: Controls how closely the model follows your prompt
        - Higher values: More literal interpretation but may look unnatural
        - Lower values: More creative but might not follow prompt exactly
    
    - **Seed**: Controls the initial random noise pattern
        - Same seed + same parameters = reproducible results
    """)



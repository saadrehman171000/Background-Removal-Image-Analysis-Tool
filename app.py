import streamlit as st
import os
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from u2net_model import U2NET # Import U2NET model
import openai
import numpy as np
import base64
import json
from datetime import datetime
import re # Import regex library

# Initialize OpenAI client with API key
openai.api_key = ""

# Load pre-trained model
@st.cache_resource
def load_model():
    # Load the U2NET model
    model = U2NET(in_ch=3, out_ch=1) # U2NET for background removal has 1 output channel

    model_path = "./saved_models/u2net/u2net.pth"
    
    if not os.path.exists(model_path):
        st.error(f"U2NET weights not found at {model_path}. Please download the weights.")
        st.stop()

    # Load the weights. Map location 'cpu' is generally safe.
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess image for the U2NET model"""
    # U2NET typically uses a fixed input size, e.g., 320x320
    # And a different normalization
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]) # U2NET normalization
    ])
    return transform(image).unsqueeze(0)

def remove_background(image, model):
    """Remove background using U2NET model"""
    # Preprocess image
    input_tensor = preprocess_image(image)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get mask from output - U2Net returns a list of outputs, we want the first one
    # The output shape is [batch_size, channels, height, width]
    mask = output[0].squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
    
    # Convert mask to PIL Image
    mask_img = Image.fromarray(mask, 'L').resize(image.size, Image.BILINEAR)
    
    # Create transparent background
    output = image.copy()
    output.putalpha(mask_img)
    
    return output

def get_image_metadata(image):
    """Get image metadata using OpenAI Vision API"""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    # Save as PNG to preserve transparency
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Encode bytes to base64
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

    # Define the desired JSON structure in the prompt
    prompt_text = """Analyze the following image and provide metadata in the following JSON format. Fill in the details based on the image content. Use appropriate values for each key. If a value cannot be determined, use null or an empty string/list as appropriate.

Return ONLY the JSON object within a markdown code block.

```json
{
    "item_id": "unique_id",
    "image_url": "processed_image_path",
    "bg_removed": true,
    "dominant_colors": [],
    "category": null,
    "subcategory": null,
    "gender": null,
    "attributes": [],
    "fit": null,
    "texture": null,
    "season": null,
    "occasion": [],
    "bounding_box": null,
    "aspect_ratio": null,
    "pose_estimation": null,
    "source_type": "user_upload",
    "created_at": "2023-10-26T10:30:00Z"
}
```
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500 # Increased max_tokens to allow for detailed JSON output
        )
        
        metadata_text = response.choices[0].message.content.strip()
        
        # Attempt to extract JSON from the response
        # Look for a JSON code block or a string starting with { and ending with }
        json_match = re.search(r'```json\s*({.*?})\s*```', metadata_text, re.DOTALL) # Search for JSON in a code block
        if json_match:
            json_string = json_match.group(1)
        else:
             # Fallback: Look for a string that looks like a JSON object
             json_match = re.search(r'({.*?})', metadata_text, re.DOTALL)
             if json_match:
                 json_string = json_match.group(1)
             else:
                 return f"Could not find JSON object in response. Response: {metadata_text}"

        try:
            metadata_json = json.loads(json_string)
            # Override fields that are known:
            # metadata_json['bg_removed'] = True # Already instructed the model to set this
            # metadata_json['source_type'] = 'user_upload' # Already instructed the model to set this
            metadata_json['created_at'] = datetime.utcnow().isoformat() + 'Z' # Ensure correct timestamp
            
            # Convert back to formatted JSON string for display
            return json.dumps(metadata_json, indent=4)
            
        except json.JSONDecodeError:
             return f"Could not parse extracted string as JSON. Extracted: {json_string}"

    except Exception as e:
        return f"Error getting metadata: {str(e)}"

def main():
    st.title("Background Removal & Image Analysis")
    st.write("Upload an image to remove its background and get detailed analysis")

    # Load model
    model = load_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "gif"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert("RGB") # Convert to RGB to ensure 3 channels for model
        st.image(image, caption="Original Image", use_column_width=True)

        # Remove background
        if st.button("Remove Background"):
            with st.spinner("Removing background..."):
                # Remove background
                output = remove_background(image, model)
                
                # Display result
                st.image(output, caption="Image with Background Removed", use_column_width=True)
                
                # Get and display metadata
                with st.spinner("Analyzing image..."):
                    metadata = get_image_metadata(output)
                    st.subheader("Image Analysis (JSON)") # Updated subheader
                    # Check if metadata is a JSON string before displaying as JSON
                    try:
                        st.json(json.loads(metadata))
                    except json.JSONDecodeError:
                        st.write(metadata)

                # Add download button
                buf = io.BytesIO()
                output.save(buf, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="removed_background.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main() 
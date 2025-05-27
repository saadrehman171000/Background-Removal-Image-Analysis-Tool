# Background Removal & Image Analysis Tool

A powerful tool that combines advanced background removal using the U2Net model with image analysis using OpenAI's GPT-4 Vision API.

## Features

- **Background Removal**: Removes image backgrounds using the U2Net deep learning model
- **Image Analysis**: Extracts detailed metadata using OpenAI's GPT-4 Vision API
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Download Support**: Save processed images with transparent backgrounds
- **Multiple Format Support**: Works with JPG, JPEG, PNG, WebP, BMP, TIFF, and GIF formats

## Technical Details

### Background Removal
The project uses the U2Net model, a state-of-the-art deep learning model specifically designed for salient object detection and background removal. The model:
- Processes images at 320x320 resolution
- Uses custom normalization (mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
- Generates high-quality alpha masks for precise background removal

### Image Analysis
The tool uses OpenAI's GPT-4 Vision API to analyze images and extract metadata including:
- Dominant colors
- Categories and subcategories
- Attributes and characteristics
- Fit and texture information
- Seasonal and occasion details
- Pose estimation
- And more

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BackgroundRemoval
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download U2Net weights**
   - Download the U2Net weights file (`u2net.pth`) from [Hugging Face](https://huggingface.co/danielgatis/rembg/resolve/main/u2net.pth)
   - Create a directory: `mkdir -p saved_models/u2net`
   - Place the downloaded `u2net.pth` file in the `saved_models/u2net` directory

4. **Configure OpenAI API**
   - The API key is already configured in the application
   - If you need to use a different key, update it in `app.py`

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser
2. Upload an image using the file uploader
3. Click "Remove Background" to process the image
4. View the processed image with transparent background
5. Review the extracted metadata
6. Download the processed image if desired

## Dependencies

- Python 3.7+
- Streamlit
- PyTorch
- TorchVision
- Pillow
- OpenAI
- NumPy

## Project Structure

```
BackgroundRemoval/
├── app.py              # Main application file
├── u2net_model.py      # U2Net model implementation
├── requirements.txt    # Python dependencies
├── saved_models/       # Directory for model weights
│   └── u2net/
│       └── u2net.pth   # U2Net weights file
└── README.md          # This documentation
```

## Notes

- The application requires an internet connection for the OpenAI API calls
- Processing time may vary depending on image size and complexity
- For best results, use high-quality images with clear subject boundaries

## License

This project is open source and available under the MIT License.

## Acknowledgments

- U2Net model by [Xuebin Qin](https://github.com/xuebinqin/U-2-Net)
- OpenAI for the GPT-4 Vision API
- Streamlit for the web interface framework

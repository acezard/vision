import gradio as gr
import io
from openai import OpenAI
import base64
import requests
import os
import glob
import logging
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the OpenAI client
client = OpenAI(base_url="http://192.168.1.65:1234/v1", api_key="lm-studio")


def categorize_description(description):
    """
    Categorizes the given image description into 'Photograph', 'Document', or 'Uncategorized'.

    Args:
    description (str): The description of the image as provided by the AI model.

    Returns:
    str: The category of the image based on the description.
    """
    description_lower = description.lower()

    # Criteria for categorizing as 'Photograph' or 'Document'
    photograph_keywords = [
        "trees",
        "sky",
        "road",
        "inside a car",
        "cars and trees",
        "photograph",
        "upside-down photo",
    ]
    document_keywords = [
        "receipt",
        "paper",
        "document",
        "form",
        "contract",
        "invoice",
        "acceptation",
        "insurance policy",
        "rental contract",
        "transaction",
    ]

    if any(keyword in description_lower for keyword in photograph_keywords):
        return "Photograph"
    elif any(keyword in description_lower for keyword in document_keywords):
        return "Document"
    else:
        return "Uncategorized"


def process_images_in_directory(directory_path):
    """
    Processes images in the given directory, categorizing each based on AI-generated descriptions.

    Args:
    directory_path (str): Path to the directory containing images to process.

    Returns:
    str: Summary of the document categorization results.
    """
    logging.info(f"Processing directory: {directory_path}")
    image_files = glob.glob(os.path.join(directory_path, "*.jpg")) + glob.glob(
        os.path.join(directory_path, "*.png")
    )

    category_counts = {"Photograph": 0, "Document": 0, "Uncategorized": 0}

    for image_path in image_files:
        logging.info(f"Processing image: {image_path}")
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logging.error(f"Error reading {image_path}: {e}")
            continue

        # Send the request and get the response
        completion = client.chat.completions.create(
            model="your-model-name",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What type of image or document is this?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
            stream=True,
        )

        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:  # Check if content is not None
                response_text += chunk.choices[0].delta.content

        logging.info(f"AI response: {response_text}")
        category = categorize_description(response_text)
        category_counts[category] += 1
        logging.info(f"Image categorized as: {category}")

    # Generate a bar chart for the category counts
    categories = list(category_counts.keys())
    counts = [category_counts[category] for category in categories]

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color=['blue', 'green', 'red'])
    plt.title('Document Summary')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Instead of saving to a file, convert the plot to a PIL Image
    plt.tight_layout()
    plot_buffer = io.BytesIO()
    plt.savefig(plot_buffer, format='png')
    plt.close()
    plot_buffer.seek(0)
    plot_image = Image.open(plot_buffer)

    # Return both the summary text and the PIL Image
    summary_text = "Document Summary:\n" + "\n".join([f"{category}: {count}" for category, count in category_counts.items()])
    logging.info(f"Summary text: {summary_text}")
    return summary_text, plot_image

# Gradio interface setup
iface = gr.Interface(
    fn=process_images_in_directory,
    inputs="text",
    outputs=["text", "image"],  # Updated to include image output for the plot
    description="Enter the path to your directory containing images. Please note, processing can take several minutes depending on the number of documents.",
)

# Run the interface
iface.launch(server_name="0.0.0.0")
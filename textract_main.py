import os
import boto3
import base64
import io
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import flet as ft
from flet import Page, Container, Text, TextButton, FilePicker, FilePickerResultEvent, Image as FletImage, border_radius

# Load environment variables
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION_TEXTRACT")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID_TEXTRACT")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY_TEXTRACT")

# AWS Textract Client
client = boto3.client(
    'textract',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def ShowBoundingBox(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline=boxColor, width=2)

def ShowSelectedElement(draw, box, width, height, boxColor):
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], fill=boxColor)

def draw_bounding_boxes(image_path, blocks):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for block in blocks:
        if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
            box = block['Geometry']['BoundingBox']
            if block['BlockType'] == "KEY_VALUE_SET":
                color = 'red' if block.get('EntityTypes', [''])[0] == "KEY" else 'green'
            elif block['BlockType'] == 'TABLE':
                color = 'blue'
            elif block['BlockType'] == 'CELL':
                color = 'yellow'
            elif block['BlockType'] == 'SELECTION_ELEMENT' and block.get('SelectionStatus') == 'SELECTED':
                ShowSelectedElement(draw, box, width, height, 'blue')
                continue
            else:
                color = 'red'
            ShowBoundingBox(draw, box, width, height, color)
    
    image_path_with_boxes = image_path.replace(".png", "_boxed.png")
    image.save(image_path_with_boxes)
    return image_path_with_boxes

def process_image(e, result_image, file_picker):
    if file_picker.result and file_picker.result.files:
        image_path = file_picker.result.files[0].path
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=["TABLES", "FORMS"])
            blocks = response.get('Blocks', [])
            boxed_image_path = draw_bounding_boxes(image_path, blocks)
            result_image.src = boxed_image_path
            result_image.visible = True
            result_image.update()
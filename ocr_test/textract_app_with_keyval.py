import os
import boto3
import base64
import io
import csv
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import flet as ft
from flet import Page, Container, Text, TextButton, FilePicker, FilePickerResultEvent, Image as FletImage, border_radius, TextField, Column, Row

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

def extract_key_values(response):
    key_values = {}
    for block in response.get('Blocks', []):
        if block['BlockType'] == 'KEY_VALUE_SET' and 'EntityTypes' in block and 'KEY' in block['EntityTypes']:
            key = None
            value = None
            if 'Relationships' in block:
                for rel in block['Relationships']:
                    if rel['Type'] == 'CHILD':
                        key = ''.join([w['Text'] for w in response['Blocks'] if w['Id'] in rel['Ids'] and 'Text' in w])
                    elif rel['Type'] == 'VALUE':
                        for value_id in rel['Ids']:
                            value_block = next((b for b in response['Blocks'] if b['Id'] == value_id), None)
                            if value_block and 'Relationships' in value_block:
                                for v_rel in value_block['Relationships']:
                                    if v_rel['Type'] == 'CHILD':
                                        value = ''.join([w['Text'] for w in response['Blocks'] if w['Id'] in v_rel['Ids'] and 'Text' in w])
            if key and value:
                key_values[key] = value
    return key_values

def save_to_csv(key_values):
    identity_number = key_values.get('IdentityNumber')
    if not identity_number:
        return "No Identity Number Found"
    filename = f"{identity_number}.csv"
    if os.path.exists(filename):
        return f"Data for ID {identity_number} already exists."
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in key_values.items():
            writer.writerow([key, value])
    return f"Data saved as {filename}"

def process_image(e, result_image, file_picker, query_output, details_container):
    if file_picker.result and file_picker.result.files:
        image_path = file_picker.result.files[0].path
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=["FORMS"])
            key_values = extract_key_values(response)
            message = save_to_csv(key_values)
            query_output.value = message
            query_output.visible = True
            query_output.update()
            
            details_container.content = Column(
                [
                    Row([
                        Text(f"{key}:", size=14, weight="bold"),
                        Text(f"{value}", size=14)
                    ]) for key, value in key_values.items()
                ],
                spacing=10
            )

            print(details_container.content)
            details_container.visible = True
            details_container.update()

def main(page: Page):
    page.title = "OCR Document Analyzer"
    file_picker = FilePicker()
    result_image = FletImage(width=350, height=350, border_radius=border_radius.all(10))
    select_image = TextButton(content=Text("Select Image"), on_click=lambda _: file_picker.pick_files(allow_multiple=False))
    query_output = Text(visible=False)
    details_container = Container(visible=False)
    file_picker.on_result = lambda e: process_image(e, result_image, file_picker, query_output, details_container)
    page.overlay.append(file_picker)
    page.add(Container(content=ft.Column([select_image, result_image, query_output, details_container])))

ft.app(main)

import os
import boto3
import numpy as np
import mindspore as ms
import time

from dotenv import load_dotenv
from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from PIL import Image, ImageDraw
import flet as ft
from flet import AppBar, CupertinoFilledButton, Page, Container, Text, View, FontWeight, colors, TextButton, padding, ThemeMode, border_radius, Image as FletImage, FilePicker, FilePickerResultEvent, icons

first_prompt_entered = True

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

def query_document(image_bytes, question):
    response = client.analyze_document(
        Document={'Bytes': image_bytes},
        FeatureTypes=["TABLES", "FORMS", "QUERIES"],
        QueriesConfig={'Queries': [{'Text': question}]}
    )
    
    answer = "No answer found."
    for block in response.get('Blocks', []):
        if block["BlockType"] == "QUERY_RESULT":
            answer = block["Text"]
            break
    
    return answer

def main(page: Page):
    page.title = "Botani"
    page.theme_mode = ThemeMode.DARK

    page.fonts = {
        "RobotoFlex": f"fonts/RobotoFlex-VariableFont.ttf",
        "RobotoMono": f"fonts/RobotoMono-VariableFont.ttf",
        "RobotoMonoItalic": f"fonts/RobotoMono-Italic-VariableFont.ttf"
    }

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
                prompt_container.visible = True
                select_image.visible = False
                restart_button.visible = True

                def process_query(e):
                    global first_prompt_entered

                    question = prompt_input.value

                    input_text = query_document(image_bytes, question)
                    type_speed = 0.001

                    if first_prompt_entered:
                        prompt_display.content.controls.append(Text("", size=14, font_family="RobotoMono", weight=FontWeight.W_300, color="#cbddd1"))
                        prompt_display.update()
                        first_prompt_entered = False
                    else:
                        prompt_display.content.controls[-1].value = ""
                        prompt_display.update()

                    for char in input_text:
                        prompt_display.content.controls[-1].value += char
                        prompt_display.update()
                        time.sleep(type_speed)

                prompt_input.on_submit = process_query

                result_image.update()
                prompt_input.update()
                prompt_container.update()
                restart_button.update()
                select_image.update()

    def restart_process(e):
        global first_prompt_entered

        result_image.visible = False
        select_image.visible = True
        restart_button.visible = False
        if not(first_prompt_entered):
            prompt_display.content.controls[-1].value = ""
        prompt_input.value = ""
        prompt_container.visible = False

        restart_button.update()
        result_image.update()
        select_image.update()
        prompt_input.update()
        prompt_display.update()
        prompt_container.update()

    restart_button = TextButton(content=Text("Start over", font_family="RobotoMono", size=14, weight=FontWeight.W_300, color="#cbddd1"), visible=False, on_click=restart_process)

    initial_image = ft.Image(src="assets/result_initial.png", width=350, height=350, border_radius=border_radius.all(10))
    result_image = ft.Image(src="assets/result_initial.png", width=400, height=500, border_radius=border_radius.all(10), fit=ft.ImageFit.CONTAIN)
    result_image.visible = False

    file_picker = FilePicker()
    file_picker.on_result = lambda e: process_image(e, result_image, file_picker)
    page.overlay.append(file_picker)

    select_image = TextButton(content=initial_image, on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    prompt_text = Text("Enter query:", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color="#cbddd1")
    prompt_input = ft.TextField(
        hint_text="Type your query here",
        width=350
    )

    #prompt_display = Text("LLM will respond here.", size=16, font_family="RobotoMono", weight=FontWeight.W_300, color="#cbddd1")
    prompt_display = Container(
        height = 450,
        content=ft.Column(scroll="auto")
    )

    prompt_container = Container(content=ft.Column([prompt_text, prompt_input, prompt_display]), visible=False)

    def route_change(e):
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    AppBar(title_spacing=50, title=Text("Botani", font_family="RobotoFlex", size=40, weight=FontWeight.W_700, color="#cbddd1"), bgcolor="#141d1f", toolbar_height=120,
                           actions=[
                               TextButton(content=Container(Text("GitHub", font_family="RobotoFlex", size=18, weight=FontWeight.W_400, color="#cbddd1"), padding=padding.only(right=25, left=25)), url="https://github.com/rjldg/mindspore_proj"),
                               TextButton(content=Container(Text("About Us", font_family="RobotoFlex", size=18, weight=FontWeight.W_400, color="#cbddd1"), padding=padding.only(right=25, left=25)), on_click=lambda _: page.go("/aboutus"))
                           ]
                    ),
                    Container(
                        content=ft.Column(
                            [
                                Text("DETECT TEA LEAF SICKNESS\nUSING ARTIFICIAL INTELLIGENCE", font_family="RobotoFlex", size=40, weight=FontWeight.W_800, color="#cbddd1"),
                                Text("Your tea leaf health companion, powered by AI.", font_family="RobotoMono", size=20, weight=FontWeight.W_500, color="#cbddd1"),
                                Text("Featured Models: MindSpore ResNet-50 and AWS Titan", font_family="RobotoMono", size=14, weight=FontWeight.W_300, color="#cbddd1"),
                            ]
                        ),
                        padding=padding.only(left=100, top=70)
                    ),
                    Container(
                        ft.ElevatedButton(content=Text("Get Started", font_family="RobotoFlex", size=18, weight=FontWeight.W_500, color="#cbddd1"), on_click=lambda _: page.go("/botani")),
                        padding=padding.only(left=100, top=70),
                    )
                ],
            )
        )
        
        if page.route == "/botani":
            page.views.append(
                View(
                    "/botani",
                    [
                        AppBar(color="#cbddd1", bgcolor="#141d1f"),
                        Container(
                            content=ft.Row(
                                [
                                    Container(
                                        content=ft.Column(
                                            [
                                                result_image,
                                                restart_button,
                                            ],
                                            alignment=ft.MainAxisAlignment.CENTER,
                                        ),
                                        width=400,
                                        padding=padding.only(left=10, top=25)
                                    ),
                                    Container(
                                        select_image,
                                        padding=padding.only(top=100)
                                    ),
                                    Container(
                                        content=prompt_container,
                                        width=400,
                                        padding=padding.only(left=10, top=50)
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=50
                            ),
                        ),
                    ],
                )
            )
        if page.route == "/aboutus":
            page.views.append(
                View(
                    "/aboutus",
                    [
                        AppBar(color="#cbddd1", bgcolor="#141d1f"),
                        Container(
                            Text("MEET THE TEAM", font_family="RobotoFlex", size=40, weight=FontWeight.W_700, color="#cbddd1"),
                            padding=padding.only(left=100)
                        ),
                        Container(
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/aaron.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Aaron Abadiano", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color="#cbddd1"),
                                                    Text("ababadiano@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color="#cbddd1"),
                                                ]
                                            )
                                            ),
                                        ]
                                    ),
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/clara.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Clara Capistrano", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color="#cbddd1"),
                                                    Text("cascapistrano@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color="#cbddd1"),
                                                ]
                                            )
                                            ),
                                        ]
                                    ),
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/rance_new.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Rance De Guzman", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color="#cbddd1"),
                                                    Text("rjldeguzman@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color="#cbddd1"),
                                                ]
                                            )
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            padding=padding.only(left=200),
                        ),
                    ],
                )
            )
        page.update()

    def view_pop(e):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    page.go(page.route)

ft.app(main)

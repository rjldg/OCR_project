import os
import numpy as np
import mindspore as ms
import time

from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from PIL import Image
import flet as ft
from flet import AppBar, CupertinoFilledButton, Page, Container, Text, View, FontWeight, colors, TextButton, padding, ThemeMode, border_radius, Image as FletImage, FilePicker, FilePickerResultEvent, icons

from image_classifier.resnet50_archi import resnet50
from main import predict, retrieve_and_generate_response, ref_class_names

first_prompt_entered = True

def main(page: Page):
    page.title = "Botani"
    page.theme_mode = ThemeMode.DARK

    page.fonts = {
        "RobotoFlex": f"fonts/RobotoFlex-VariableFont.ttf",
        "RobotoMono": f"fonts/RobotoMono-VariableFont.ttf",
        "RobotoMonoItalic": f"fonts/RobotoMono-Italic-VariableFont.ttf"
    }

    def pick_files_result(e: FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()

    def process_image(e):
        if file_picker.result and file_picker.result.files:
            image_path = file_picker.result.files[0].path
            print(image_path)

            if os.path.exists(image_path):
                predicted_class, confidence = predict(image_path)
                conf_converted = confidence * 100
                
                result_pred.value = f"Prediction: {ref_class_names.get(predicted_class)}"
                result_conf.value = f"[{conf_converted:.2f}% Confident]"
                result_image.src = image_path
                prompt_container.visible = True

                result_pred.color = "green" if predicted_class == "Normal" else "red"
                result_conf.color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"

                result_image.visible = True
                select_image.visible = False
                restart_button.visible = True
                result_pred.update()
                result_conf.update()
                prompt_container.update()
                restart_button.update()
                result_image.update()
                select_image.update()
            else:
                result_pred.value = f"Error: File '{image_path}' does not exist."
                result_pred.update()
                result_conf.update()

    def restart_process(e):
        result_pred.value = ""
        result_conf.value = ""
        result_image.visible = False
        select_image.visible = True
        restart_button.visible = False
        prompt_display.content.controls[-1].value = ""
        prompt_input.value = ""
        prompt_container.visible = False

        result_pred.update()
        result_conf.update()
        restart_button.update()
        result_image.update()
        select_image.update()
        prompt_input.update()
        prompt_display.update()
        prompt_container.update()

    file_picker = FilePicker(on_result=process_image)
    selected_files = Text()
    
    page.overlay.append(file_picker)

    restart_button = TextButton(content=Text("Start over", font_family="RobotoMono", size=14, weight=FontWeight.W_300, color="#cbddd1"), visible=False, on_click=restart_process)
    
    result_pred = Text(size=30, font_family="RobotoFlex", weight=FontWeight.W_700)
    result_conf = Text(size=20, font_family="RobotoMono", weight=FontWeight.W_500)

    initial_image = ft.Image(src="assets/result_initial_new.png", width=350, height=350, border_radius=border_radius.all(10))
    result_image = ft.Image(src="assets/result_initial_new.png", width=350, height=350, border_radius=border_radius.all(10), fit=ft.ImageFit.FILL)
    result_image.visible = False

    select_image = TextButton(content=initial_image, on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    prompt_text = Text("Enter prompt:", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color="#cbddd1")
    prompt_input = ft.TextField(
        hint_text="Type your prompt here",
        width=350,
        on_submit=lambda e: generating_response()
    )

    #prompt_display = Text("LLM will respond here.", size=16, font_family="RobotoMono", weight=FontWeight.W_300, color="#cbddd1")
    prompt_display = Container(
        height = 450,
        content=ft.Column(scroll="auto")
    )

    prompt_container = Container(content=ft.Column([prompt_text, prompt_input, prompt_display]), visible=False)

    def generating_response():
        global first_prompt_entered

        input_text = retrieve_and_generate_response(result_pred.value, prompt_input.value)
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
                                                result_pred,
                                                result_conf,
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
                                        padding=padding.only(left=10, top=25)
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

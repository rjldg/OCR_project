import os
import shutil
import tempfile
import flet as ft
import cv2

def main(page: ft.Page):
    page.title = "MindOCR"
    page.window_width = 750
    page.window_height = 500
    page.theme = ft.Theme(font_family="centurygothic")
    page.horizontal_alignment=ft.CrossAxisAlignment.CENTER

    # FUNCTIONS #
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_file = e.files[0]

            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, selected_file.name)
            shutil.copy(selected_file.path, temp_image_path)
            image_viewer.src = selected_file.path
            image_viewer.update()
    
    def show_result(e):
        # get OCR module here, returns as image at output folder
        image_viewer.src = "output/result.png"
    

    # COMPONENTS #
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    page.add(pick_files_dialog)

    image_viewer = ft.Image(
        src="placeholder.jpg",
        width=400,
        height=300,
        fit=ft.ImageFit.CONTAIN,
    )

    # USER CONTROLS #
    title = ft.Text("MindOCR - Image-To-Text Application Using Optical Character Recognition")
    mindicon = ft.Container(content=ft.Row([ft.Image(src="logo.png", width=32), title]), padding=ft.padding.symmetric(horizontal=20))
    file_picker_button = ft.ElevatedButton(
        "Pick files",
        icon=ft.icons.UPLOAD_FILE,
        on_click=lambda _: pick_files_dialog.pick_files(allow_multiple=False),
    )

    imageprocess = ft.ElevatedButton(
        "Convert",
        icon=ft.icons.REPLAY,
        on_click=show_result
    )

    # LAYOUT # 
    navbar = ft.Row([mindicon], alignment=ft.MainAxisAlignment.CENTER)
    function_rack = ft.Row([file_picker_button,imageprocess], alignment=ft.MainAxisAlignment.CENTER)
    layout = ft.Column([
        image_viewer,
        function_rack
    ], 
    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    alignment=ft.MainAxisAlignment.CENTER,
    width=650, 
    )
    
    page.add(navbar)
    page.add(layout)

ft.app(target=main, assets_dir="assets")

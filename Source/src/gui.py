from shape_model import ShapeModel

import PySimpleGUI as sg
import os.path
import cv2 as cv


# layout of the left column with directories and files
files_column = [
    [
        sg.Text("Select directory containing training data for the model"),
    ],
    [
        sg.Text("Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-",
        )
    ],
    [
        sg.Button("Select this directory", enable_events=True, key="-SELECT BUTTON-"),
        sg.In(size=(15, 1), enable_events=False, key="-MODEL NAME-"),
    ]
]

# layout of the right column with image overview
image_column = [
    [sg.Text("Image overview")],
    [sg.Text(size=(40, 1), key="-IMG NAME-")],
    [sg.Image(filename="", key="-IMAGE-")],
]

# layout of whole window
layout = [
    [
        sg.Column(files_column),
        sg.VSeparator(),
        sg.Column(image_column),
    ]
]

window = sg.Window("Choose training data", layout)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "-FOLDER-":     # folder name was filled in
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
            for file in os.listdir(folder):
                filename, extension = os.path.splitext(file)
                img = cv.imread(folder + "/" + file)
        except OSError:
            file_list = []

        files = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".jpg", ".png"))
        ]
        window["-FILE LIST-"].update(files)
    elif event == "-FILE LIST-":  # file was chosen from the listbox
        try:
            if len(values["-FILE LIST-"]) > 0:
                filename, extension = os.path.splitext(values["-FILE LIST-"][0])
                img = cv.imread(values["-FOLDER-"] + "/" + values["-FILE LIST-"][0])
                img_bytes = cv.imencode(".png", img)[1].tobytes()
                (window["-IMAGE-"]).update(data=img_bytes)
        except OSError:
            print("No element selected")
    elif event == "-SELECT BUTTON-":
        if len(values["-MODEL NAME-"]) <= 0:
            sg.PopupOK("Type in model name!")
        else:
            folder = values["-FOLDER-"]
            folder = folder.replace("\\", "/")
            model_name = values["-MODEL NAME-"]
            model = ShapeModel()
            model.read_train_data(folder, model_name)
            for img in model.training_images:
                print(img.name)

window.close()

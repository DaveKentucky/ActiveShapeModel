from shape_model import ShapeModel
from shape_creator import ShapeCreator

import PySimpleGUI as sg
import os.path
import cv2 as cv


def make_window_read_data():
    """
    Creates window with functionality of reading data for model

    :return: window reading data for model from file
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
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
            sg.Text("Model's name:"),
            sg.In(size=(15, 1), enable_events=False, key="-MODEL NAME-"),
        ],
        [
            sg.Button("Select this directory", enable_events=True, key="-SELECT BUTTON-"),
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

    return sg.Window("Choose training data", layout, finalize=True)


def make_window_mark_landmarks():
    """
    Creates window layout with instructions about marking landmark points on image

    :return: window with instructions about marking landmark points
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
    layout = [
        [
            sg.Text("Click on image to mark a landmark point")
        ],
        [
            sg.Text("Drag and drop marked point to replace it")
        ],
        [
            sg.Button("Delete", enable_events=True, key="-DELETE BUTTON-"),
            sg.Text("Delete last landmark point marked")
        ],
        [
            sg.Button("End contour", enable_events=True, key="-END CONTOUR BUTTON-"),
            sg.Text("End marking current contour")
        ],
        [
            sg.Button("Change type", enable_events=True, key="-TYPE BUTTON-"),
            sg.Text("Change type of current contour into opposite")
        ],
    ]

    return sg.Window("Manual", layout, finalize=True)


def cv_mouse_click(event, x, y, flags, creator):
    """
    Resolve mouse input on the image

    :param event: OpenCV event
    :type event: cv2.EVENT
    :param x: X coordinate of the mouse cursor
    :type x: int
    :param y: Y coordinate of the mouse cursor
    :type y: int
    :param flags: additional flags
    :param creator: image object
    :type creator: ShapeCreator
    :return: None
    """

    if event == cv.EVENT_LBUTTONDOWN:

        creator.add_point(x, y)

    cv.imshow(creator.window_name, creator.get_display_image())

    # if event == cv.EVENT_LBUTTONUP:
    #     if g_index == -1:
    #         img.add_landmark_point(x, y)
    #     else:
    #         img.set_landmark_point([x, y], g_index)
    #         g_index = -1

    # if event == cv.EVENT_MOUSEMOVE:
    #     if g_index != -1:
    #         img.set_landmark_point([x, y], g_index)
    #         cv.imshow("Image", img.get_display_image())


def mark_landmark_points(m_img):
    window = make_window_mark_landmarks()
    if m_img.is_loaded:
        win_name = m_img.name
        creator = ShapeCreator(m_img.image, win_name)
        cv.imshow(win_name, creator.get_display_image())
        cv.setMouseCallback(win_name, cv_mouse_click, creator)
    else:
        print(f"Failed to read model image: {m_img.name}\nMake sure the image is loaded")
        return None

    while True and cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow(creator.window_name, creator.get_display_image())
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-DELETE BUTTON-":
            creator.delete_point()
        elif event == "-END CONTOUR BUTTON-":
            creator.end_contour()
        elif event == "-TYPE BUTTON-":
            creator.flip_contour_type()

    window.close()
    # TODO: implement OpenCV event loop in separate thread


def select_training_data_files():
    window = make_window_read_data()
    model = None

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-FOLDER-":     # folder name was filled in
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
                model = ShapeModel()
                model_name = values["-MODEL NAME-"]
                model.read_train_data(folder, model_name)
                for img in model.training_images:
                    print(img.name)
                break

    window.close()
    return model


if __name__ == '__main__':
    m = select_training_data_files()
    mark_landmark_points(m.training_images[0])

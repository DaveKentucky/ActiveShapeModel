from asm_model import ASMModel
from shape_model import ModelImage
from shape_creator import ShapeCreator
import select_area as sa

import PySimpleGUI as sg
import os.path
import cv2 as cv
import sys
import statistics

sg.theme("Dark Grey 13")


def make_window_main_menu():
    """
    Creates window with main menu of the application

    :return: main menu window
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
    layout = [
        [
            sg.Button("Create model", enable_events=True, key="-CREATE-")
        ],
        [
            sg.Button("Search", enable_events=True, key="-SEARCH-")
        ],
        [
            sg.Button("Test model", enable_events=True, key="-TEST-")
        ],
        [
            sg.Button("Exit", enable_events=True, key="-EXIT-")
        ]
    ]

    return sg.Window("Main menu", layout, finalize=True)


def make_window_read_data(training):
    """
    Creates window with functionality of reading data for model

    :param training: defines if window should be used for selection of training or testing data
    :type training: bool
    :return: window reading data for model from file
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
    if training:
        data_type = 'training'
    else:
        data_type = 'test'

    # layout of the left column with directories and files
    files_column = [
        [
            sg.Button("Back", enable_events=True, key="-BACK-")
        ],
        [
            sg.Text(f"Select directory containing {data_type} data for the model")
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
            sg.Check("Use shape from file", key="-MARKED POINTS-"),
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

    return sg.Window(f"Choose {data_type} data", layout, element_justification="c", finalize=True)


def make_window_mark_landmarks(points):
    """
    Creates window layout with instructions about marking landmark points on image

    :param points: number of points that were marked in previous image in this model
    :type points: int
    :return: window with instructions about marking landmark points
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
    if points < 0:
        txt = ""
    else:
        txt = "Points to mark: " + str(points)
    layout = [
        [
            sg.T(' ' * 13),
            sg.Text("Click on image to mark a landmark point")
        ],
        [
            sg.Button("Delete", enable_events=True, key="-DELETE BUTTON-", size=(10, 1)),
            sg.Text("Delete last landmark point marked")
        ],
        [
            sg.Button("End contour", enable_events=True, key="-END CONTOUR BUTTON-", size=(10, 1)),
            sg.Text("End marking current contour")
        ],
        [
            sg.Button("Change type", enable_events=True, key="-TYPE BUTTON-", size=(10, 1)),
            sg.Text("Change type of current contour into opposite")
        ],
        [
            sg.Text(txt)
        ],
        [
            sg.T(' ' * 35),
            sg.Button("Submit", enable_events=True, key="-SUBMIT BUTTON-")
        ],
    ]

    return sg.Window("Manual", layout, finalize=True)


def make_window_test_model_params(models_list):
    """
    Creates window layout for performing testing of the model

    :return: window with functionality of setting parameters for testing the model
    :rtype: PySimpleGUI.PySimpleGUI.Window
    """
    layout = [
        [
            sg.Button("Back", enable_events=True, key="-BACK-")
        ],
        [
            sg.DropDown(models_list, size=(20, 10), enable_events=True, key="-MODEL-")
        ],
        [
            sg.Text("Distribution of test and train sets"),
        ],
        [
            sg.Slider(range=(1, 100), orientation='h', size=(40, 10), default_value=25, key="-TEST SIZE-")
        ],
        [
            sg.Text("Measures of quality")
        ],
        [
            sg.Check("Coefficient of determination", default=True, key="-R SQUARE-"),
        ],
        [
            sg.Check("Mean error of point's position", default=True, key="-MEAN ERROR-"),
        ],
        [
            sg.T(' ' * 35),
            sg.Button("Test model", enable_events=True, key="-SUBMIT BUTTON-")
        ],
    ]

    return sg.Window("Test params", layout, finalize=True)


def make_window_test_results(results):
    """
    Creates window layout for showing results of testing the model

    :param results: dictionary with test results
    :type results: dict[map[str, list]]
    :return: window with view of test results
    :rtype: PySimpleGUI.PySimpleGUI.Window 
    """""
    layout = []
    for key in results.keys():
        mean = round(statistics.mean(results[key]), 2)
        layout.append([sg.Text(f"Average {key} for all test images: {mean}")])
    layout.append([sg.T(' ' * 26),
                   sg.Button("OK", enable_events=True, key="-OK BUTTON-")])

    return sg.Window("Test results", layout, finalize=True)


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


def main_menu():
    """
    Creates and operates main menu window of the application

    :return: code of option selected in the menu
    :rtype int
    """
    response = -1
    window = make_window_main_menu()
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED or event == "-EXIT-":
            break
        elif event == "-CREATE-":
            response = 1    # code for creating a new model
            break
        elif event == "-SEARCH-":
            response = 2    # code for searching on an image with existing model
            break
        elif event == "-TEST-":
            response = 3    # code for testing an existing model
            break

    window.close()
    return response


def mark_landmark_points(m_img, h_img):
    """
    Creates and operates window where user can draw a shape on image

    :param m_img: ModelImage object with image loaded
    :type m_img: ModelImage
    :param h_img: ModelImage with already marked points for a pattern
    :type h_img: ModelImage
    :return: creator object with data about marked points and possibility of creating ShapeInfo object
    :rtype: ShapeCreator
    """
    n_points = -1
    if m_img.shape_info is not None:
        n_points = len(m_img.shape_info.point_info)

    window = make_window_mark_landmarks(n_points)
    if m_img.is_loaded:
        win_name = m_img.name
        creator = ShapeCreator(m_img.image, win_name)
        cv.imshow(win_name, creator.get_display_image())
        cv.setMouseCallback(win_name, cv_mouse_click, creator)
    else:
        print(f"Failed to read model image: {m_img.name}\nMake sure the image is loaded")
        return None
    if h_img is not None:
        cv.imshow("Help image", h_img.show(False))

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
        elif event == "-SUBMIT BUTTON-":
            if 0 < n_points != len(creator.points):
                sg.PopupOK(f"Number of marked points is incorrect!\nYou have to mark {n_points} points!")
            else:
                answer = sg.popup_yes_no("Are you sure you want to submit this shape?")
                if answer == 'Yes':
                    break

    window.close()
    cv.destroyWindow(creator.window_name)
    return creator


def select_training_data_files():
    """
    Creates and operates window where user can select directory with training data for a new model

    :return: response code, ASMModel object with set training images and if the landmark points should be read
            from prepared files
    :rtype: (int, ASMModel, bool)
    """
    response = -1
    window = make_window_read_data(True)
    model = None
    files = None

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-BACK-":
            response = 0
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
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".jpg", ".png", ".bmp"))
            ]
            window["-FILE LIST-"].update(files)
        elif event == "-FILE LIST-":  # file was chosen from the listbox
            try:
                if len(values["-FILE LIST-"]) > 0:
                    # filename, extension = os.path.splitext(values["-FILE LIST-"][0])
                    img = cv.imread(values["-FOLDER-"] + "/" + values["-FILE LIST-"][0])
                    img_bytes = cv.imencode(".png", img)[1].tobytes()
                    (window["-IMAGE-"]).update(data=img_bytes)
            except OSError:
                print("No element selected")
        elif event == "-SELECT BUTTON-":
            if files is None:
                sg.PopupOK("Select correct folder with data!")
            elif len(files) <= 0:
                sg.PopupOK("Select correct folder with data!")
            elif len(values["-MODEL NAME-"]) <= 0:
                sg.PopupOK("Type in model name!")
            else:
                folder = values["-FOLDER-"]
                folder = folder.replace("\\", "/")
                model = ASMModel()
                model_name = values["-MODEL NAME-"]
                model.read_train_data(folder, model_name, files)
                response = 1
                break

    window.close()
    return response, model, values["-MARKED POINTS-"]


def select_search_data_files():
    """
    Creates and operates window where user can select an image to search with the model

    :return: response code, image for searching and name of the model that should be used
    :rtype: (int, numpy.ndarray, str)
    """
    response = -1
    window = make_window_read_data(False)
    img = None
    files = None
    model_name = ''

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-BACK-":
            response = 0
            break
        elif event == "-FOLDER-":     # folder name was filled in
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
                for file in os.listdir(folder):
                    img = cv.imread(folder + "/" + file)
            except OSError:
                file_list = []

            files = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".jpg", ".png", ".bmp"))
            ]
            window["-FILE LIST-"].update(files)
        elif event == "-FILE LIST-":  # file was chosen from the listbox
            try:
                if len(values["-FILE LIST-"]) > 0:
                    img = cv.imread(values["-FOLDER-"] + "/" + values["-FILE LIST-"][0])
                    img_bytes = cv.imencode(".png", img)[1].tobytes()
                    (window["-IMAGE-"]).update(data=img_bytes)
            except OSError:
                print("No element selected")
        elif event == "-SELECT BUTTON-":
            if files is None:
                sg.PopupOK("Select correct folder with data!")
            elif len(files) <= 0:
                sg.PopupOK("Select correct folder with data!")
            elif len(values["-MODEL NAME-"]) <= 0:
                sg.PopupOK("Type in model name!")
            elif img is None:
                sg.PopupOK("Select an image for test data!")
            else:
                model_name = values["-MODEL NAME-"]
                response = 2
                break

    window.close()
    return response, img, model_name


def mark_search_area(image):
    """
    Enables the user to mark search area for the model on the given image

    :param image: image where the model should be applied
    :type image: numpy.ndarray
    :return: top left corner of the search area and its size
    :rtype: ((int, int), (int, int))
    """
    # Set recursion limit
    sys.setrecursionlimit(10 ** 9)

    # create an instance of rectangle mark operator class
    select_window = sa.DragRectangle

    # read image's width and height
    width = image.shape[1]
    height = image.shape[0]

    # initiate the operator object
    sa.init(select_window, image, "Image", width, height)
    cv.namedWindow(select_window.window_name)
    cv.setMouseCallback(select_window.window_name, sa.drag_rectangle, select_window)
    cv.imshow("Image", select_window.image)

    # loop until selected area is confirmed
    wait_time = 1000
    while cv.getWindowProperty("Image", cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow("Image", select_window.image)
        key = cv.waitKey(wait_time)
        if key == 27:
            cv.destroyWindow("Image")
        # if return_flag is True, break from the loop
        elif select_window.return_flag:
            break

    left, top, right, bottom = sa.get_area_rectangle(select_window)
    return (left, top), (right - left, bottom - top)


def set_model_test_params(models_list):
    """
    Creates and operates window where user can set parameters for testing the model

    :param models_list: list of tuples with models' names and number of images in the model
    :type: list[tuple[str, int]]
    :return: response code, tested model's name,
            proportion of the test set size and dictionary of quality measures that should be used
    :rtype: (int, str, int, dict[str, bool])
    """
    response = -1
    models_names = list()
    for elem in models_list:
        models_names.append(elem[0])
    window = make_window_test_model_params(models_names)
    model = ''
    test_size = 0.0
    measures = tuple()

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "-BACK-":
            response = 0
            break
        elif event == "-MODEL-":
            model = values["-MODEL-"]
            images_count = models_list[models_names.index(model)][1]
        elif event == "-SUBMIT BUTTON-":     # submission was called
            if values["-MODEL-"] == '':     # no model was selected
                sg.PopupOK("You have to select a model first!")
            else:
                test_size = float(values["-TEST SIZE-"] / 100)
                measures = {
                    'R square': values["-R SQUARE-"],
                    'mean error': values["-MEAN ERROR-"]}
                check = int(test_size * images_count)
                if check == images_count:
                    sg.PopupOK("Test set size is too large!")
                elif check > int(images_count / 2):
                    answer = sg.PopupYesNo(
                        "Test set is larger than training set. Are you sure you want to proceed with given parameters?")
                    if answer == 'Yes':
                        response = 3
                        break
                else:
                    response = 3
                    break

    window.close()
    return response, model, test_size, measures


def show_test_results(results):
    """
    Creates and operates window where user can see the results of testing the model

    :param results: dictionary with test results
    :type results: dict[map[str, list]]
    return: None
    """
    window = make_window_test_results(results)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "-OK BUTTON-":
            break

    window.close()

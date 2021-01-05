import gui
from asm_model import ASMModel, ASMFitResult
from database import Database

import cv2 as cv


def create_model():
    """
    Creates new ASM model from given files and saves it into database

    :return: response code
    :rtype: int
    """
    response, model, prepared_shape = gui.select_training_data_files()
    if response <= 0:
        return response
    if model is not None:     # successfully created shape model
        my_db = Database()
        m_id = my_db.insert_model(model)
        help_image = None
        if prepared_shape:
            model.set_from_asf()
            my_db.insert_shape_info(model.shape_info, m_id)
            for image in model.training_images:
                my_db.insert_image(image, m_id)
        else:
            for i, image in enumerate(model.training_images):   # loop through every training image
                creator = gui.mark_landmark_points(image, help_image)
                if creator is not None:     # received not None creator
                    if len(creator.points) > 0:     # creator's point list is not empty
                        if i == 0:
                            help_image = image  # save first image with points as help image
                        if model.shape_info is None:      # there is no shape info set yet
                            info = creator.create_shape_info()  # create new shape info based on created shape
                            model.set_shape_info(info)       # set model's shape info
                            s_id = my_db.insert_shape_info(info, m_id)
                        image.set_points_from_list(creator.points)    # set image points array with marked points
                        my_db.insert_image(image, m_id)
        return 0


def search_with_model():
    """
    Fits model to a test image

    :return: response code
    :rtype: int
    """
    response, image, model_name = gui.select_search_data_files()
    if response <= 0:
        return response
    if image is not None:
        top_left, size = gui.mark_search_area(image)
        if model_name != '':
            my_db = Database()
            model = my_db.read_model(model_name)
            model.build_model()
            result = model.fit_all(image, top_left, size, verbose=True)
            model.show_result(image, result)
            cv.waitKey(0)
            return response
    else:
        return response


def test_model():
    """
    Tests fitting algorithm performance on given model

    :return: response code
    :rtype: int
    """
    my_db = Database()
    models_names = my_db.get_all_models_names()
    response, model_name, test_size, measures = gui.set_model_test_params(models_names)
    if response <= 0:
        return response
    # make sure the window was not closed
    if model_name != '' and test_size > 0.0 and len(measures) > 0:
        # make sure any measure was selected
        if True not in measures.values():
            return 3

        model = my_db.read_model(model_name)
        performance = model.test_model(measures, test_size)
        gui.show_test_results(performance)
        return 0


if __name__ == '__main__':

    res = gui.main_menu()
    while True:
        if res < 0:
            break
        elif res == 0:
            res = gui.main_menu()
        elif res == 1:
            res = create_model()
        elif res == 2:
            res = search_with_model()
        elif res == 3:
            res = test_model()

    # search_with_model()

    # db = Database()
    # db.get_all_models_names()
    # mdl = db.read_model('face')
    # search_with_model()
    # test_model(mdl)
    # mdl.show_mean_shape(blank=True)
    # img = cv.imread("E:/Szkolne/Praca_inzynierska/ActiveShapeModel/src/Data/meat_database/F1011flb.bmp")
    # rt = mdl.fit_all(img, (190, 185), (230, 240), verbose=True)
    # mdl.show_result(img, rt)
    # cv.waitKey(0)

    # create_model()

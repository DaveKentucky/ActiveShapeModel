import gui
from asm_model import ASMModel, ASMFitResult
from database import Database


def create_model():
    """
    Creates new ASM model from given files and saves it into database

    :return: response code
    :rtype: int
    """
    m_id = -1
    response, model, prepared_shape = gui.select_training_data_files()
    if response <= 0:
        return response
    if model is not None:     # successfully created shape model
        my_db = Database()
        help_image = None
        help_creator = None
        if prepared_shape:
            wait = gui.wait_window("Reading model parameters...")
            model.set_from_asf()
            wait.close()
            m_id = my_db.insert_model(model)
            my_db.insert_shape_info(model.shape_info, m_id)
            for image in model.training_images:
                my_db.insert_image(image, m_id)
        else:
            for i, image in enumerate(model.training_images):   # loop through every training image
                creator, response = gui.mark_landmark_points(image, help_image, help_creator)
                if response <= 0:
                    if i > 0 and m_id >= 0:
                        my_db.delete_model(m_id)
                    break
                if creator is not None:     # received not None creator
                    if len(creator.points) > 0:     # creator's point list is not empty
                        if i == 0:
                            help_image = image  # save first image with points as help image
                            help_creator = creator
                        if model.shape_info is None:      # there is no shape info set yet
                            info = creator.create_shape_info()  # create new shape info based on created shape
                            m_id = my_db.insert_model(model)
                            model.set_shape_info(info)       # set model's shape info
                            s_id = my_db.insert_shape_info(info, m_id)
                        image.set_points_from_list(creator.points)    # set image points array with marked points
                        my_db.insert_image(image, m_id)
                else:
                    break
        return 0


def search_with_model():
    """
    Fits model to a test image and shows the result

    :return: response code
    :rtype: int
    """
    my_db = Database()
    models_names = my_db.get_all_models_names()
    response, image_name, image, model_name = gui.select_search_data_files(models_names)
    if response <= 0:
        return response
    if image is not None:
        top_left, size, response = gui.mark_search_area(image)
        if response <= 0:
            return 0
        if model_name != '':
            # read model from database
            wait = gui.wait_window("Reading model from database...")
            model = my_db.read_model(model_name)
            wait.close()
            # build model structure
            wait = gui.wait_window("Building ASM model structure...")
            if top_left == (0, 0) and size == (image.shape[1], image.shape[0]):
                for img in model.training_images:
                    if img.name == image_name:
                        top_left, size = img.get_shape_frame(10)
                        break
            model.build_model()
            wait.close()
            # fit model to the image
            wait = gui.wait_window("Fitting model to the image...")
            result = model.fit_all(image, top_left, size, verbose=False)
            wait.close()
            response = gui.visualise_result(model, image, result)
            return response if response <= 0 else 0
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

        # read model from database
        wait = gui.wait_window("Reading model from database...")
        model = my_db.read_model(model_name)
        wait.close()
        # test model performance
        wait = gui.wait_window("Testing model performance...")
        performance = model.test_model(measures, test_size)
        wait.close()
        gui.show_test_results(performance)
        return 0


def show_model():
    """
    Shows visualisation of a model and enables tweaking its parameters manually

    :return: response code
    :rtype: int
    """
    my_db = Database()
    models_names = my_db.get_all_models_names()
    response, model_name = gui.show_model(models_names)
    if response <= 0:
        return response
    # read model from database
    wait = gui.wait_window("Reading model from database...")
    model = my_db.read_model(model_name)
    wait.close()
    # build model structure
    wait = gui.wait_window("Building ASM model structure...")
    model.build_model()
    wait.close()
    return gui.visualise_model(model)


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
        elif res == 4:
            res = show_model()

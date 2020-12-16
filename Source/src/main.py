import gui
from database import Database


def create_model():

    model = gui.select_training_data_files()
    if model is not None:     # successfully created shape model

        my_db = Database()
        m_id = my_db.insert_model(model)
        help_image = None

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

        model.build_model()


if __name__ == '__main__':

    # create_model()
    my_db = Database()
    model = my_db.read_model('meat')
import gui
import database as db


def create_shape_model():

    shape_model = gui.select_training_data_files()
    if shape_model is not None:     # successfully created shape model

        my_db = db.Database()
        my_db.insert_model(shape_model)

        for image in shape_model.training_images:   # loop through every training image
            creator = gui.mark_landmark_points(image)
            if creator is not None:     # received not None creator
                if len(creator.points) > 0:     # creator's point list is not empty
                    if shape_model.shape_info is None:      # there is no shape info set yet
                        info = creator.create_shape_info()  # create new shape info based on created shape
                        shape_model.set_shape_info(info)       # set model's shape info
                    image.points = creator.points.copy()    # set image points array with marked points


if __name__ == '__main__':

    create_shape_model()

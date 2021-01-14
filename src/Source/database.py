from asm_model import ASMModel
from shape_info import ShapeInfo, PointInfo
from model_image import ModelImage

import numpy as np
import sqlite3


class Database:

    # name of the connected database
    name: str

    # connection object with database
    db: sqlite3.Connection

    # cursor for executing queries on established connection
    cursor: sqlite3.Cursor

    def __init__(self, db_name="asm_database"):
        """
        Initializes connection on localhost with existing database or creates a new one

        :param db_name: name of the database
        :type db_name: str
        """
        self.name = db_name
        self.db = sqlite3.connect(db_name + ".db")
        self.cursor = self.db.cursor()
        query = "select * from sqlite_master where type='table';"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        # check if the tables already exist
        if len(results) == 0:
            self.create_default_tables()
            self.cursor.execute(query)
            results = self.cursor.fetchall()
        else:
            print(f"\nUsing database {db_name} with tables:")
        for r in results:
            print(r[1])

    def create_default_tables(self):
        """
        Creates default tables needed for the ASM database

        :return: None
        """
        print("\nCreating default tables...")

        # description of tables in database
        tables = {
            "models":
                "create table models ("
                "model_id integer primary key not null,"
                "model_name text,"
                "directory text,"
                "pts_on_normal integer,"
                "search_pts_on_normal integer"
                ")",
            "shape_info":
                "create table shape_info ("
                "shape_info_id integer primary key not null,"
                "model_id integer,"
                "foreign key (model_id) references models(model_id) on delete cascade)",
            "point_info":
                "create table point_info ("
                "point_info_id integer primary key not null,"
                "shape_info_id integer,"
                "point_index integer not null,"
                "contour_id integer,"
                "contour_type integer,"
                "previous integer,"
                "next integer,"
                "foreign key (shape_info_id) references shape_info(shape_info_id) on delete cascade)",
            "images":
                "create table images ("
                "image_id integer primary key not null,"
                "model_id integer,"
                "image_name text,"
                "points_count integer,"
                "foreign key (model_id) references models(model_id) on delete cascade)",
            "points":
                "create table points ("
                "point_id integer primary key not null,"
                "image_id integer,"
                "point_index integer,"
                "x integer,"
                "y integer,"
                "foreign key (image_id) references images (image_id) on delete set null)",
            "contours":
                "create table contours ("
                "contour_id integer primary key not null,"
                "shape_info_id integer,"
                "is_closed integer,"
                "start_point_index integer,"
                "foreign key (shape_info_id) references shape_info(shape_info_id) on delete cascade)"
        }

        for table_name in tables:
            print(f"Creating table {table_name}...")
            query = tables[table_name]
            self.cursor.execute(query)
            print("OK")

    def close(self):
        """
        Commits all unsaved changes and closes the connection with database

        :return: None
        """
        self.cursor.close()
        self.db.commit()
        self.db.close()
        print(f"Connection with database {self.name} closed")

    def insert_model(self, asm_model):
        """
        Inserts ASMModel object into database

        :type asm_model: ASMModel
        :return: ID of the inserted model in database
        :rtype: int
        """
        query = "insert into models (model_name, directory, pts_on_normal, search_pts_on_normal)" \
                "values (?, ?, ?, ?)"
        data = (asm_model.name_tag, asm_model.directory, asm_model.points_on_normal, asm_model.search_points_on_normal)
        self.cursor.execute(query, data)    # insert model object
        self.db.commit()

        return self.cursor.lastrowid

    def insert_shape_info(self, shape_info, model_id):
        """
        Inserts ShapeInfo object into database

        :type shape_info: ShapeInfo
        :param model_id: ID of corresponding Model object in database
        :type model_id: int
        :return: ID of the inserted ShapeInfo object in database
        :rtype: int
        """
        query = "insert into shape_info (model_id) values (?)"
        self.cursor.execute(query, (model_id, ))    # insert ShapeInfo object

        shape_info_id = self.cursor.lastrowid
        for i in range(shape_info.n_contours):  # insert contours
            is_closed = shape_info.contour_is_closed[i]
            start_id = shape_info.contour_start_index[i]
            self.insert_contour(shape_info_id, is_closed, start_id)

        for i, point in enumerate(shape_info.point_info):   # insert points info
            self.insert_point_info(shape_info_id, i, point.contour, point.type, point.connect_from, point.connect_to)

        self.db.commit()
        print(f"Inserted ShapeInfo object with {len(shape_info.point_info)} points information")

        return shape_info_id

    def insert_contour(self, shape_info_id, contour_is_closed, contour_start_id):
        """
        Inserts contour data into database

        :param shape_info_id: ID of corresponding ShapeInfo object in database
        :type shape_info_id: int
        :param contour_is_closed: definition if the contour is open or closed
        :type contour_is_closed: int
        :param contour_start_id: index of the first point in this contour
        :type contour_start_id: int
        :return: ID o the inserted contour in database
        :rtype: int
        """
        query = "insert into contours (shape_info_id, is_closed, start_point_index) values (?, ?, ?)"
        data = (shape_info_id, contour_is_closed, contour_start_id)
        self.cursor.execute(query, data)    # insert contour

        self.db.commit()

        return self.cursor.lastrowid

    def insert_point_info(self, shape_info_id, point_id, contour_id, contour_type, connect_from, connect_to):
        """
        Inserts point info data into database

        :param shape_info_id: ID of corresponding ShapeInfo object in database
        :type shape_info_id: int
        :param point_id: index of the point info in ShapeInfo list of points
        :type point_id: int
        :param contour_id: index of contour that the point is part of
        :type contour_id: int
        :param contour_type: definition if the contour is open or closed
        :type contour_type: int
        :param connect_from: index of the previous point in contour
        :type connect_from: int
        :param connect_to:index of the next point in contour
        :type connect_to: int
        :return: ID of the inserted PointInfo object in database
        :rtype: int
        """
        query = "insert into point_info (shape_info_id, point_index, contour_id, contour_type, previous, next)" \
                "values (?, ?, ?, ?, ?, ?)"
        data = (shape_info_id, point_id, contour_id, contour_type, connect_from, connect_to)
        self.cursor.execute(query, data)    # insert PointInfo object

        self.db.commit()

        return self.cursor.lastrowid

    def insert_image(self, image, model_id):
        """
        Inserts model image into database

        :type image: ModelImage
        :param model_id: ID of corresponding model in database
        :type model_id: int
        :return: ID of the inserted ModelImage object in database
        :rtype: int
        """

        query = "insert into images (model_id, image_name, points_count) values (?, ?, ?)"
        data = (model_id, image.name, image.n_points)
        self.cursor.execute(query, data)    # insert ModelImage object

        image_id = self.cursor.lastrowid
        for i, point in enumerate(image.points):    # insert points
            self.insert_point(point[0], point[1], i, image_id)

        self.db.commit()
        print(f"Inserted {image.name} image with {image.n_points} points")

        return image_id

    def insert_point(self, x, y, point_index, image_id):
        """
        Inserts point into database

        :param x: X coordinate of the point
        :type x: int
        :param y: Y coordinate of the point
        :type y: int
        :param point_index: index of the point in array of points
        :type point_index: int
        :param image_id: ID of the corresponding image in database
        :type image_id: int
        :return: ID of the inserted point in database
        :rtype: int
        """

        query = "insert into points (image_id, x, y, point_index) values (?, ?, ?, ?)"
        data = (image_id, int(x), int(y), point_index)
        self.cursor.execute(query, data)    # insert point

        self.db.commit()

        return self.cursor.lastrowid

    def read_model(self, model_name):
        """
        Reads ASM model data from database

        :param model_name: name of the model
        :type model_name: str
        :return: model read from database
        :rtype: ASMModel
        """
        # read model
        query = f"select * from models where model_name = '{model_name}'"
        self.cursor.execute(query)  # read model with given name
        result = self.cursor.fetchone()
        if result is not None:  # model with given name found
            print(f"\nModel {model_name} found in the database! Reading data...")
            model = ASMModel(result[3], result[4])  # create model
            directory = result[2]
            mdl_id = result[0]
        else:
            print(f"Model {model_name} not found in the database")
            return

        # read images
        print("\nReading images...")
        image_ids, image_files = self.read_into_lists('image_id', 'image_name', 'images', 'model_id', mdl_id)
        model.read_train_data(directory, model_name, image_files)  # read training images from files
        print("Images read")

        # read shape info
        print("\nReading ShapeInfo...")
        query = f"select shape_info_id from shape_info where model_id = {mdl_id}"
        self.cursor.execute(query)  # read shape info of the model
        result = self.cursor.fetchone()
        if result is not None:
            shape_info = ShapeInfo()
            si_id = result[0]

            # read contours
            start_indices, types = \
                self.read_into_lists('start_point_index', 'is_closed', 'contours', 'shape_info_id', si_id)
            shape_info.contour_start_index = start_indices
            shape_info.contour_is_closed = types
            shape_info.n_contours = len(start_indices)

            # read points info
            query = f"select point_index, contour_id, contour_type, previous, next from point_info " \
                    f"where shape_info_id = {si_id}"
            self.cursor.execute(query)  # read point info for given shape info
            result = self.cursor.fetchall()
            pi_dict = {}
            for r in result:    # store all results in a dictionary
                pi = PointInfo(r[1], r[2], r[3], r[4])
                pi_dict[r[0]] = pi
            pi_list = []
            for key in sorted(pi_dict.keys()):  # sort the dictionary by point info index and save points info in a list
                pi_list.append(pi_dict[key])
            shape_info.point_info = pi_list
            model.set_shape_info(shape_info)
            print("ShapeInfo read")
        else:
            print(f"No ShapeInfo stored for the model {model_name}")

        # read points
        print("\nReading points...")
        for i, img_id in enumerate(image_ids):    # read points for every image
            query = f"select point_index, x, y from points where image_id = {img_id}"
            self.cursor.execute(query)  # read points for given image
            result = self.cursor.fetchall()
            pts_dict = {}
            pts = []
            for r in result:    # store all results in a dictionary
                pts_dict[(r[0])] = np.array([r[1], r[2]])
            for key in sorted(pts_dict.keys()):     # sort the dictionary by point index and save points in a list
                pts.append(pts_dict[key])
            mi = model.training_images[i]   # get ModelImage object
            mi.set_points_from_list(pts)    # set image points with values from database
        print("Points read")
        print("\nReading model from database completed!")

        return model

    def read_into_lists(self, col1, col2, table, key_name, key_value):
        """
        Reads two columns from database and stores the values in lists

        :param col1: name of the first column
        :type col1: str
        :param col2: name of the second column
        :type col2: str
        :param table: name of the table
        :type table: str
        :param key_name: name of the search key
        :type key_name: str
        :param key_value: value of the search key
        :type key_value: str
        :return: Lists with read values
        :rtype: (list, list)
        """
        query = f"select {col1}, {col2} from {table} where {key_name} = {key_value}"
        self.cursor.execute(query)  # read contour of the shape info
        result = self.cursor.fetchall()
        list1 = []
        list2 = []
        for r in result:  # create lists of information about contours
            list1.append(r[0])
            list2.append(r[1])

        return list1, list2

    def get_all_models_names(self):
        """
        Reads names of all the models in the database

        :return: list of tuples with models' names and number of images saved in the model
        :rtype: list[tuple[str, int]]
        """
        query = "select model_name from models where model_id is not null"
        self.cursor.execute(query)  # read all models' names
        result = self.cursor.fetchall()
        models_list = list()
        for r in result:
            models_list.append(r[0])

        for i, name in enumerate(models_list):
            query = f"select model_id from models where model_name = '{name}'"
            self.cursor.execute(query)  # read id of every model
            result = self.cursor.fetchone()
            query = f"select count(image_id) from images where model_id = {result[0]}"
            self.cursor.execute(query)  # read number of points of every model
            result = self.cursor.fetchone()
            models_list[i] = (models_list[i], result[0])

        return models_list

    def delete_model(self, model_id):
        """
        Deletes model with given name from database

        :param model_id: ID of the model in database
        :type model_id: int
        :return: None
        """
        query = f"delete from models where model_id = {model_id}"
        self.cursor.execute(query)
        return None


if __name__ == '__main__':

    my_db = Database()
    my_db.cursor.close()
    # my_db.read_model("hand")

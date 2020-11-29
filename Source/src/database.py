from shape_model import ShapeModel, ShapeInfo
from model_image import ModelImage

import mysql.connector
import numpy as np


class Database:

    db: mysql.connector.MySQLConnection

    cursor: mysql.connector.connection.CursorBase

    def __init__(self, username="root", password="password1", db_name="asm_database"):

        self.connect_to_database(username, password, db_name)

    def connect_to_database(self, username, password, db_name):
        """
        Connects to the MySQL database with given parameters

        :param username: name of the MySQL user
        :type username: str
        :param password: password of the MySQL user
        :type password: str
        :param db_name: name of the database to connect
        :type db_name: str
        """
        connection = mysql.connector.connect(host="localhost", user=username, password=password, database=db_name)

        if connection:
            cursor = connection.cursor()
            self.db = connection
            self.cursor = cursor

            cursor.execute("show databases like '" + db_name + "'")
            # check if any matching database was found
            if cursor.stored_results() is 0:
                print(f"Unable to find database {db_name}")
                self.create_database(db_name)

    def create_database(self, db_name):
        """
        Creates database with given name

        :param db_name: name of the database to connect
        :type db_name: str
        :return: None
        """
        self.cursor.execute("create database " + db_name)
        # create tables if database was created
        self.create_default_tables()

    def create_default_tables(self):
        """
        Create default tables needed for the project's database

        :param cursor: MySQL database cursor object
        :return: None
        """

        # TODO: Change database structure to fit new classes
        # create table od models
        sql = "create table models (model_id int primary key auto_increment not null," \
              "model_name varchar(40), points_count int)"
        self.cursor.execute(sql)

        # create table of shape_infos
        sql = "create table shape_info (shape_info_id int primary key auto_increment not null," \
              "model_id int, foreign key (model_id) references models(model_id) on delete cascade)"
        self.cursor.execute(sql)

        # create table od images
        sql = "create table images (image_id int primary key auto_increment not null, model_id int," \
              "image_name varchar(40), points_count int," \
              "foreign key (model_id) references models(model_id) on delete cascade)"
        self.cursor.execute(sql)

        # create table of points
        sql = "create table points (point_id int primary key auto_increment not null, image_id int," \
              "x int, y int, previous_point int, next_point int," \
              "foreign key (image_id) references images (image_id) on delete set null)"
        self.cursor.execute(sql)

        # create table of contours
        sql = "create table contours (contour_id int primary key auto_increment not null, shape_info_id int," \
              "is_closed int, start_point_index int," \
              "foreign key shape_info_id references shape_info(shape_info_id) on delete cascade)"
        self.cursor.execute(sql)

    def insert_model(self, shape_model: ShapeModel):
        """
        Inserts ShapeModel object into database
        :param shape_model: ShapeModel object
        :type shape_model: ShapeModel
        :return: ID of the inserted model in database
        :rtype: int
        """
        sql = "insert into models (model_name, points_count) values (%s, %s)"
        data = (shape_model.name_tag, shape_model.n_landmarks)
        self.cursor.execute(sql, data)
        self.db.commit()
        return self.cursor.lastrowid

    def insert_shape_info(self, shape_info: ShapeInfo, model_id):
        """
        Inserts ShapeInfo object into database
        :param shape_info: ShapeInfo object
        :type shape_info: ShapeInfo
        :param model_id: corresponding model ID
        :type model_id: int
        :return: ID of the inserted ShapeInfo object in database
        :rtype: int
        """
        sql = "insert into shape_info (model_id) values (%s)"
        self.cursor.execute(sql, (model_id, ))
        self.db.commit()

        for i in range(shape_info.n_contours):
            self.insert_contour(shape_info, self.cursor.lastrowid, i)

        return self.cursor.lastrowid

    def insert_contour(self, shape_info: ShapeInfo, shape_info_id, contour_index):
        """
        Inserts contour data into database
        :param shape_info: ShapeInfo object
        :type shape_info: ShapeInfo
        :param shape_info_id: ID of corresponding ShapeInfo object in database
        :type shape_info_id: int
        :param contour_index: index of the contour in contours list of the ShapeInfo
        :type contour_index: int
        :return: None
        """
        sql = "insert into contours (shape_info_id, is_closed, start_point_index) values (%s, %s, %s)"
        data = (shape_info_id, shape_info.contour_is_closed[contour_index], shape_info.contour_start_index[contour_index])
        self.cursor.execute(sql, data)
    # def insert_pdm(db, cursor, shapes, points_count, name):
    #     """
    #     Inserts a model into database
    #
    #     :param db: MySQL database connection object
    #     :type db: mysql.connector.connection
    #     :param cursor: MySQL database cursor object
    #     :type cursor: mysql.connector.cursor
    #     :param shapes: array of all shapes contained in the model (shape as 1D array)
    #     :type shapes: numpy.ndarray
    #     :param points_count: total count of points in a single shape
    #     :type points_count: int
    #     :param name: name of the model
    #     :type name: str
    #     :return: Message string about added model
    #     :rtype: str
    #     """
    #
    #     # check if model with this name already exists
    #     cursor.execute("select model_id from Models where model_name = '" + name + "'");
    #     index = cursor.fetchone()
    #     if index is not None:
    #         model_id = index[0]
    #         return_string = f"Shapes added to database to model with ID={model_id}"
    #     else:
    #         # insert a model object into the table
    #         model_data = (name, points_count)
    #         sql1 = "insert into Models (model_name, points_count) values (%s, %s)"  # query for inserting a model
    #         cursor.execute(sql1, model_data)
    #         # get added model's id
    #         model_id = cursor.lastrowid
    #         return_string = f"Shapes added to database to a new model with ID={model_id}"
    #
    #     sql2 = "insert into Images (model_id, mean_model) values (%s, %s)"  # query for inserting a shape
    #     sql3 = "insert into Points (image_id, point_index, x, y) values (%s, %s, %s, %s)"  # query for inserting a point
    #
    #     # insert all shapes objects of the model into the table
    #     for i, image in enumerate(shapes):
    #         # first shape in an array should always be the mean shape
    #         if i == 0:
    #             mean = 1
    #         else:
    #             mean = 0
    #         cursor.execute(sql2, (model_id, mean))
    #
    #         # get added shape's id
    #         last_id = cursor.lastrowid
    #
    #         # insert all points objects of the shape into the table
    #         shape = shapes[i]
    #         print(shape)
    #         for j in range(points_count):
    #             x = int(shape[2 * j])
    #             y = int(shape[2 * j + 1])
    #             cursor.execute(sql3, (last_id, j, x, y))
    #
    #     db.commit()
    #
    #     return return_string

    # def get_pdm(cursor, name):
    #     """
    #     Get a point distribution model from database
    #
    #     :param cursor: MySQL database cursor
    #     :param name: name of the model
    #     :type name: str
    #     :return: array of all shapes, total count of points in a single shape
    #     :rtype: (numpy.ndarray, numpy.ndarray, int)
    #     """
    #
    #     # get model's id
    #     cursor.execute("select model_id, points_count from models where model_name = '" + name + "'")
    #     result = cursor.fetchone()
    #     if result is None:
    #         print(f"No model '{name}' found in database")
    #         return None
    #
    #     model_id = result[0]
    #     points_count = result[1]
    #
    #     # get total count of shapes saved in the model
    #     cursor.execute("select count(image_id) from images where model_id = '" + str(model_id) + "'")
    #     result = cursor.fetchone()
    #     if result is None:
    #         return None
    #
    #     shapes_count = result[0]
    #
    #     # prepare array for points
    #     shapes = np.zeros((shapes_count - 1, 2 * points_count), int)
    #     mean_shape = np.zeros((1, 2 * points_count), int)
    #
    #     # get id of each shape saved in the model
    #     cursor.execute("select image_id from images where model_id = '" + str(model_id) + "'")
    #     result = cursor.fetchall()
    #     if result is None:
    #         print(f"No shapes associated with model '{name}' found in database")
    #         return None
    #
    #     shape_id = tuple(x[0] for x in result)
    #
    #     # get points belonging to each shape
    #     for i, id in enumerate(shape_id):
    #         cursor.execute("select x, y from points where image_id = " + str(id))
    #         result = cursor.fetchall()
    #
    #         cursor.execute("select mean_model from images where image_id = " + str(id))
    #         is_mean = cursor.fetchone()
    #         if is_mean[0] == 1:
    #             # save coordinates in mean shape array
    #             for j, point in enumerate(result):
    #                 mean_shape[0, j * 2] = result[j][0]
    #                 mean_shape[0, j * 2 + 1] = result[j][1]
    #         else:
    #             # save coordinates in array
    #             for j, point in enumerate(result):
    #                 shapes[i - 1][j * 2] = result[j][0]
    #                 shapes[i - 1][j * 2 + 1] = result[j][1]
    #
    #     print(f"Model '{name}' read from database successfully with {shapes_count} shapes")
    #     return mean_shape, shapes, points_count


if __name__ == '__main__':
    from pdm import PDM

    my_db = Database()

    # model = PDM("Face_images", 5, "face")
    # model.save_to_db()
    # get_pdm(my_cursor, "face")

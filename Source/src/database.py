from shape_model import ShapeModel
from shape_info import ShapeInfo
from model_image import ModelImage

import mysql.connector
from mysql.connector import errorcode
import numpy as np


class Database:

    # name of the connected database
    name: str

    # connection object with database
    db: mysql.connector.MySQLConnection

    # cursor for executing queries on established connection
    cursor: mysql.connector.connection.CursorBase

    def __init__(self, username="root", password="password1", db_name="asm_database"):
        """
        Initializes connection on localhost with existing database or creates a new one

        :param username: MySQL user name
        :type username: str
        :param password: MySQL user password
        :type password: str
        :param db_name: name of the database
        :type db_name: str
        """
        self.name = db_name
        self.connect_to_database(username, password, db_name)
        self.cursor.execute("show tables")
        results = self.cursor.fetchall()
        print(f"\nUsing database {db_name} with tables:")
        for r in results:
            print(r[0])

    def close(self):
        """
        Commits all unsaved changes and closes the connection with database

        :return: None
        """
        self.cursor.close()
        self.db.commit()
        self.db.close()
        print(f"Connection with database {self.name} closed")

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
        host = "localhost"
        try:
            print(f"Connecting to database {db_name}...")
            connection = mysql.connector.connect(host=host, user=username, password=password, database=db_name)
            if connection:
                cursor = connection.cursor()
                self.db = connection
                self.cursor = cursor
                print(f"Connected to database {db_name}")
        except mysql.connector.Error as er:

            if er.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Invalid username and/or password")
            elif er.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")

                connection = mysql.connector.connect(host=host, user=username, password=password)
                if connection:
                    cursor = connection.cursor()
                    self.db = connection
                    self.cursor = cursor
                    self.create_database(db_name)
            else:
                print(er)

    def create_database(self, db_name):
        """
        Creates database with given name

        :param db_name: name of the database to connect
        :type db_name: str
        :return: None
        """
        try:
            print(f"\nCreating new database {db_name}...")
            self.cursor.execute("create database " + db_name)
            self.cursor.execute("use " + db_name)
            print("OK")
            # create tables if database was created
            self.create_default_tables()
        except mysql.connector.Error as er:
            print("Failed creating database: {}".format(er))
            exit(1)

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
                "model_id int primary key auto_increment not null,"
                "model_name varchar(40),"
                "points_count int)",
            "shape_info":
                "create table shape_info ("
                "shape_info_id int primary key auto_increment not null,"
                "model_id int,"
                "foreign key (model_id) references models(model_id) on delete cascade)",
            "point_info":
                "create table point_info ("
                "point_info_id int primary key auto_increment not null,"
                "shape_info_id int,"
                "point_index int not null,"
                "contour_id int,"
                "contour_type int,"
                "previous int,"
                "next int,"
                "foreign key (shape_info_id) references shape_info(shape_info_id) on delete cascade)",
            "images":
                "create table images ("
                "image_id int primary key auto_increment not null, model_id int,"
                "image_name varchar(40),"
                "points_count int,"
                "foreign key (model_id) references models(model_id) on delete cascade)",
            "points":
                "create table points ("
                "point_id int primary key auto_increment not null,"
                "image_id int,"
                "x int,"
                "y int,"
                "previous_point int,"
                "next_point int,"
                "foreign key (image_id) references images (image_id) on delete set null)",
            "contours":
                "create table contours ("
                "contour_id int primary key auto_increment not null,"
                "shape_info_id int,"
                "is_closed int,"
                "start_point_index int,"
                "foreign key (shape_info_id) references shape_info(shape_info_id) on delete cascade)"
        }

        for table_name in tables:
            try:
                print(f"Creating table {table_name}...")
                query = tables[table_name]
                self.cursor.execute(query)
            except mysql.connector.Error as er:
                if er.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print(f"Table {table_name} already exists")
                else:
                    print(er.msg)
            else:
                print("OK")

    def insert_model(self, shape_model: ShapeModel):
        """
        Inserts ShapeModel object into database

        :param shape_model: ShapeModel object
        :type shape_model: ShapeModel
        :return: ID of the inserted model in database
        :rtype: int
        """
        query = "insert into models (model_name, points_count) values (%s, %s)"
        data = (shape_model.name_tag, shape_model.n_landmarks)
        self.cursor.execute(query, data)
        self.db.commit()

        return self.cursor.lastrowid

    def insert_shape_info(self, shape_info, model_id):
        """
        Inserts ShapeInfo object into database

        :param shape_info: ShapeInfo object
        :type shape_info: ShapeInfo
        :param model_id: corresponding model ID
        :type model_id: int
        :return: ID of the inserted ShapeInfo object in database
        :rtype: int
        """
        query = "insert into shape_info (model_id) values (%s)"
        self.cursor.execute(query, (model_id, ))

        shape_info_id = self.cursor.lastrowid
        for i in range(shape_info.n_contours):
            is_closed = shape_info.contour_is_closed[i]
            start_id = shape_info.contour_start_index[i]
            self.insert_contour(shape_info_id, is_closed, start_id)

        for i, point in enumerate(shape_info.point_info):
            self.insert_point_info(shape_info_id, i, point.contour, point.type, point.connect_from, point.connect_to)

        self.db.commit()

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
        query = "insert into contours (shape_info_id, is_closed, start_point_index) values (%s, %s, %s)"
        data = (shape_info_id, contour_is_closed, contour_start_id)
        self.cursor.execute(query, data)

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
                "values (%s, %s, %s, %s, %s, %s)"
        data = (shape_info_id, point_id, contour_id, contour_type, connect_from, connect_to)
        self.cursor.execute(query, data)

        self.db.commit()

        return self.cursor.lastrowid

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

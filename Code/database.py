import mysql.connector
import numpy as np


def connect_to_database(user="root", password="password1", db_name=None):
    """
    Connects to the MySQL database with given parameters

    :param user: name of the MySQL user
    :type user: str
    :param password: password of the MySQL user
    :type password: str
    :param db_name: name of the database to connect
    :type db_name: str
    :return: MySQL connection object if succeeded or None
    :rtype: mysql.connector.connection
    """

    if db_name is None:
        connection = mysql.connector.connect(host="localhost", user=user, password=password)
    else:
        connection = mysql.connector.connect(host="localhost", user=user, password=password, database=db_name)

    if connection:
        cursor = connection.cursor()
        return connection, cursor
    else:
        print("Failed to connect")
        return None


def create_database(user="root", password="password1", db_name="asm_database"):
    """
    Creates database with given parameters or connects to the database if it has been already created

    :param user: name of the MySQL user
    :type user: str
    :param password: password of the MySQL user
    :type password: str
    :param db_name: name of the database to connect
    :type db_name: str
    :return: MySQL database connection object and cursor to the database
    :rtype: (mysql.connector.connection, mysql.connector.cursor)
    """

    db, cursor = connect_to_database(user, password)

    if db_exists(db_name):
        cursor.execute("use " + db_name)
    else:
        cursor.execute("create database " + db_name)
        # create tables if database was created
        create_default_tables(cursor)

    # create_default_tables(cursor)

    return db, cursor


def db_exists(db_name):
    """
    Checks if database with given name exists

    :param db_name: name of the searched database
    :type db_name: str
    :return: if the database exists or not
    :rtype: bool
    """

    db, cursor = connect_to_database()
    if db and cursor:
        cursor.execute("show databases like '" + db_name + "'")

        # check if any matching database was found
        if cursor.stored_results() is 0:
            return False
        else:
            return True
    else:
        return False


def create_default_tables(cursor):
    """
    Create default tables needed for the project's database

    :param cursor: MySQL database cursor object
    :return: None
    """

    # create table od models
    sql = "create table Models (model_id int primary key auto_increment not null," \
          "model_name varchar(40), points_count int)"
    cursor.execute(sql)

    # create table od images
    sql = "create table Images (image_id int primary key auto_increment not null, model_id int, mean_model bool," \
          " foreign key (model_id) references Models(model_id))"
    cursor.execute(sql)

    # create table of points
    sql = "create table Points (point_id int primary key auto_increment not null , image_id int," \
          "point_index int, x int, y int, foreign key (image_id) references Images (image_id))"
    cursor.execute(sql)


def insert_pdm(db, cursor, shapes, points_count, name):
    """
    Inserts a model into database

    :param db: MySQL database connection object
    :type db: mysql.connector.connection
    :param cursor: MySQL database cursor object
    :type cursor: mysql.connector.cursor
    :param shapes: array of all shapes contained in the model (shape as 1D array)
    :type shapes: numpy.ndarray
    :param points_count: total count of points in a single shape
    :type points_count: int
    :param name: name of the model
    :type name: str
    :return: None
    """

    # insert a model object into the table
    model_data = (name, points_count)
    sql1 = "insert into Models (model_name, points_count) values (%s, %s)"  # query for inserting a model
    cursor.execute(sql1, model_data)

    # get added model's id
    model_id = cursor.lastrowid

    sql2 = "insert into Images (model_id, mean_model) values (%s, %s)"  # query for inserting a shape
    sql3 = "insert into Points (image_id, point_index, x, y) values (%s, %s, %s, %s)"   # query for inserting a point

    # insert all shapes objects of the model into the table
    for i, image in enumerate(shapes):
        # first shape in an array should always be the mean shape
        if i == 0:
            mean = 1
        else:
            mean = 0
        cursor.execute(sql2, (model_id, mean))

        # get added shape's id
        last_id = cursor.lastrowid

        # insert all points objects of the shape into the table
        shape = shapes[i]
        print(shape)
        for j in range(points_count):
            x = int(shape[2 * j])
            y = int(shape[2 * j + 1])
            cursor.execute(sql3, (last_id, j, x, y))

    db.commit()


def get_pdm(cursor, name):
    """
    Get a point distribution model from database

    :param cursor: MySQL database cursor
    :param name: name of the model
    :type name: str
    :return: array of all shapes, total count of points in a single shape
    :rtype: (numpy.ndarray, int)
    """

    # get model's id
    cursor.execute("select model_id, points_count from models where model_name = '" + name + "'")
    result = cursor.fetchone()
    if result is None:
        return None

    model_id = result[0]
    points_count = result[1]

    # get total count of shapes saved in the model
    cursor.execute("select count(image_id) from images where model_id = '" + str(model_id) + "'")
    result = cursor.fetchone()
    if result is None:
        return None

    shapes_count = result[0]

    # prepare array for points
    shapes = np.zeros((shapes_count, 2 * points_count), int)

    # get id of each shape saved in the model
    cursor.execute("select image_id from images where model_id = '" + str(model_id) + "'")
    result = cursor.fetchall()
    if result is None:
        return None

    shape_id = tuple(x[0] for x in result)

    # get points belonging to each shape
    for i, id in enumerate(shape_id):
        cursor.execute("select x, y from points where image_id = " + str(id))
        result = cursor.fetchall()

        # save coordinates in array
        for j, point in enumerate(result):
            shapes[i][j * 2] = result[j][0]
            shapes[i][j * 2 + 1] = result[j][1]

    print(shapes)
    return shapes, points_count


if __name__ == '__main__':

    from pdm import PDM

    my_db, my_cursor = create_database("root", "password1", "asm_database")

    # model = PDM("Sword_images", 3, "sword")
    # model.save_to_db()
    get_pdm(my_cursor, "swords")
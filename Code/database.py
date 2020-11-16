import mysql.connector


def connect_to_database(user="root", password="password1", db_name=None):
    """
    Connects to the MySQL server with given parameters

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
        print("Connection successful")
        return connection
    else:
        print("Failed to connect")
        return None


def create_database(user, password, db_name):
    """
    Creates database with given parameters or connects to the database if it has been already created

    :param user: name of the MySQL user
    :type user: str
    :param password: password of the MySQL user
    :type password: str
    :param db_name: name of the database to connect
    :type db_name: str
    :return: database connection object and cursor to the database
    :rtype: (mysql.connector.connection, mysql.connector.cursor)
    """

    db = connect_to_database(user, password)
    cursor = db.cursor()

    if check_if_db_exists(db_name):
        cursor.execute("use " + db_name)
    else:
        cursor.execute("create database " + db_name)
        # create tables if database was created
        create_default_tables(cursor)

    # cursor.execute("show tables")
    #
    # for i in cursor:
    #     print(i)

    return db, cursor


def check_if_db_exists(db_name):
    """
    Checks if database with given name exists

    :param db_name: name of the searched database
    :type db_name: str
    :return: if the database exists or not
    :rtype: bool
    """

    db = connect_to_database()
    if db:
        cursor = db.cursor()
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

    # create table of points
    sql = "create table Points (point_id int primary key not null, image_id int, point_index int, x int, y int," \
          "foreign key (image_id) references Images (image_id))"
    cursor.execute(sql)

    # create table od models
    sql = "create table Models (model_id int primary key not null, model_name varchar(40), points_count int)"
    cursor.execute(sql)

    # create table od images
    sql = "create table Images (image_id int primary key not null, model_id int, mean_model bool," \
          " foreign key (model_id) references Models(model_id))"
    cursor.execute(sql)


if __name__ == '__main__':

    create_database("root", "password1", "asm_database")

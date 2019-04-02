import mysql.connector


def create_database():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd=""
    )

    mycursor = mydb.cursor()

    mycursor.execute("CREATE DATABASE test_db")


def new_odom_table():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="test_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE odom_data (id int not null auto_increment primary key, "
                     "position_x float, "
                     "position_y float, "
                     "theta_data float, "
                     "vel_linear float, "
                     "vel_angule float)")


def new_test_table():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="test_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE time_record (id int not null auto_increment primary key, "
                     "classify_time float, "
                     "frame_time float)")


def test_float_table():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="123123",
        database="test_db"
    )
    mycursor = mydb.cursor()

    mycursor.execute("CREATE TABLE test (id int not null auto_increment primary key, "
                     "position_x decimal(10, 2), "
                     "position_y decimal(10, 2), "
                     "theta_data decimal(10, 2), "
                     "vel_linear decimal(10, 2), "
                     "vel_angule decimal(10, 2))")

def insert_test():
    my_db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="123123",
        database="test_db"
    )

    my_cursor = my_db.cursor()

    sql = "INSERT INTO test (position_x, position_y, theta_data, vel_linear, vel_angule) VALUES (%i, %i, %i, %i, %i)"
    val = [
      ('Google', 'https://www.google.com'),
      ('Github', 'https://www.github.com'),
      ('Taobao', 'https://www.taobao.com'),
      ('stackoverflow', 'https://www.stackoverflow.com/')
    ]
    my_cursor.execute("SHOW TABLES")
    for x in my_cursor:
      print(x)


new_test_table()

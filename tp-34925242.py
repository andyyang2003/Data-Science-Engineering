import pymysql
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn import linear_model, tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import graphviz
import math
#sleep_data = pd.read_csv('C:\\Users\\Andy\\Documents\\term project\\student_sleep_patterns.csv') 
# global variables
connection = None
# ----------------
def reload_data():
    global connection
    if connection == None:
        print("No db loaded")
        return  
    connection.close()
    print("closed connection")
    load_data()
    print("Reloaded Database")
def load_data():  
    global connection
    ''' option to use inputted database
    host = input("Enter hostname: ")
    user = input("Enter username: ")
    passwd = input("Enter password: ")
    db = input("Enter database to connect to: ")
    connection = pymysql.connect(
        host=host, user=user, passwd=passwd, db=db
    )
    '''
    connection = pymysql.connect(host='localhost', user='mp', passwd='eecs118', db='student_sleep')
    # test loaded database
    ''' option to test and show all tables in the schema
    cur = connection.cursor()
    cur.execute(f"SHOW TABLES")
    for row in cur.fetchall():
        print(row)
    ''' 
    print("Loaded Database")
def average():
    Gender = input("Enter Gender (Male/Female/Other): ")
    University_Year = int(input("Enter university year (1/2/3/4): "))
    if University_Year == 1:
        uni = "1st Year"
    elif University_Year == 2:
        uni = "2nd Year"
    elif University_Year == 3:
        uni = "3rd Year"
    elif University_Year == 4:
        uni = "4th Year"
    query = f"""
        SELECT AVG(Sleep_Duration)
        FROM student_sleep_patterns
        WHERE Gender = '{Gender}' AND University_Year = '{uni}';
    """
    cur = connection.cursor()
    cur.execute(query)
    average = cur.fetchone()
    print(f"Average Sleep for {Gender}'s and {uni}: {average[0]:.2f} hrs")
def caffeine(): 
    caffeine_intake = (input("How many caffeinated drinks (0,1,etc.): "))
    query = f"""
        SELECT Student_ID, Sleep_Duration, Caffeine_Intake FROM student_sleep_patterns
        WHERE Caffeine_Intake = %s;
    """
    cur = connection.cursor()
    cur.execute(query, (caffeine_intake,))
    rows = cur.fetchall()
    print(f"These are the student IDs of students who have {caffeine_intake} caffeinated drinks as well as their amount of sleep")
    for row in rows:
        print(f"Student {row[0]} drank {row[2]} caffeinated drinks and slept: {row[1]} hours")
def caffeine_exercise():
    print("Example input string: 240")
    exercise = (input("How many minutes of exercise (XXX): "))
    if len(exercise) < 3: # if input is below 3 digits fill with append zeros to left of number to match
        exercise.zfill(3)
    query = f"""
        SELECT Student_ID, Caffeine_Intake, Physical_Activity FROM student_sleep_patterns
        WHERE Physical_Activity = %s;
    """
    cur = connection.cursor()
    cur.execute(query, (exercise,))
    rows = cur.fetchall()
    print(f"These are the student IDs of students and their caffeine drink consumption if they have {exercise} minutes of exercise: ")
    for row in rows:
        print(f"Student {row[0]} drank {row[1]} drinks")
def find_above_six_hours_sleep():
    University_Year = int(input("Enter university year (1/2/3/4): "))
    if University_Year == 1:
        uni = "1st Year"
    elif University_Year == 2:
        uni = "2nd Year"
    elif University_Year == 3:
        uni = "3rd Year"
    elif University_Year == 4:
        uni = "4th Year"
    query = f"""
        SELECT Student_ID, Sleep_Duration FROM student_sleep_patterns
        WHERE Sleep_Duration > 6.0 AND University_Year = '{uni}';
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    print(f"These are the student IDs of {uni}'s who sleep more than six hours and how much they sleep")
    for row in rows:
        print(f"Student {row[0]} slept {row[1]} hrs")

def find_sleep_qual_quan():
    query = """
        SELECT Student_ID, Sleep_Quality, Sleep_Duration FROM student_sleep_patterns
        WHERE Sleep_Duration < 6.0 AND Sleep_Quality > 7
        ORDER BY Sleep_Quality DESC, Sleep_Duration ASC;
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    print(f"These are the student IDs of students who sleep more than six hours and how much they sleep")
    for row in rows:
        print(f"Student {row[0]} rated Sleep Quality as a {row[1]}/10 but slept {row[2]} hrs")

def lr_on_study_sleep_hours():
    query = """
        SELECT Study_Hours, Sleep_Duration
        FROM student_sleep_patterns
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    col = ['Study_Hours', 'Sleep_Duration']
    sleep_data = pd.DataFrame(rows, columns=col)
    
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = sleep_data[['Study_Hours']] # needs to be 2d array so use double brackets
    Y = sleep_data['Sleep_Duration']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    print("Linear Regression Model R score and line variables:")
    print("r_score = ", model.score(X_test, Y_test))
    print("a = ", model.coef_) # correlation
    print("b = ", model.intercept_)
    x_range = np.arange(X.min()[0], X.max()[0]+1, 1)
    y_range = (model.coef_) * x_range + model.intercept_
    plt.scatter(sleep_data['Study_Hours'], sleep_data['Sleep_Duration'], color='b')
    plt.plot(y_range, color='r')
    plt.xlabel('Study Hours')
    plt.ylabel('Sleep_Duration')
    plt.title('Study Hours vs. Sleep_Duration with linear regression')
    plt.savefig("Study Hours vs. Sleep_Duration.png")
    plt.show()
    print("saved linear regression Study Hours vs. Sleep_Duration.png")
    plt.clf()
def predict_sleep_quality():
    # sql side
    query = """
        SELECT Sleep_Quality, Study_Hours, Caffeine_Intake, Sleep_Duration
        FROM student_sleep_patterns
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    #pd/sklearn create dataframe and create model
    col = ['Sleep_Quality', 'Study_Hours', 'Caffeine_Intake', 'Sleep_Duration']
    sleep_data = pd.DataFrame(rows, columns=col)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = sleep_data[['Study_Hours', 'Caffeine_Intake', 'Sleep_Duration']]
    Y = sleep_data['Sleep_Quality']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    # error scores
    print("Model error scores: ")
    mean_sq_error = mean_squared_error(Y_test, Y_pred)
    root_mean_sq_error = math.sqrt(mean_squared_error(Y_test, y_pred=Y_pred))
    print("mean sq error = ", mean_sq_error)
    print("root mean sq error = ", root_mean_sq_error)
    # user input prediction
    study_hours = float(input("Please enter study hours with input form (X.X): "))
    caffeine_intake = int(input("Please enter caffeine drink intake per day (Integer input): "))
    sleep_duration = float(input("Enter sleep duration in hours as a float (X.X): "))
    user_answer = pd.DataFrame([[study_hours, caffeine_intake, sleep_duration]], columns=['Study_Hours', 'Caffeine_Intake', 'Sleep_Duration']) 
    user_prediction = model.predict(user_answer)
    print(f"The model predicts you had a sleep quality: {user_prediction[0]:.2f} / 10")
    
def physactivity_vs_sleep_duration():
    query = """
        SELECT Physical_Activity, Sleep_Duration
        FROM student_sleep_patterns
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    col = ['Physical_Activity', 'Sleep_Duration']
    sleep_data = pd.DataFrame(rows, columns=col)
    
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = sleep_data[['Physical_Activity']] # needs to be 2d array so use double brackets
    Y = sleep_data['Sleep_Duration']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    print("r_score = ", model.score(X_test, Y_test))
    print("a = ", model.coef_) # correlation
    print("b = ", model.intercept_)
    x_range = np.arange(X.min()[0], X.max()[0]+1, 1)
    y_range = (model.coef_) * x_range + model.intercept_
    plt.scatter(sleep_data['Physical_Activity'], sleep_data['Sleep_Duration'], color='b')
    plt.plot(y_range, color='r')
    plt.xlabel('University_Year')
    plt.ylabel('Sleep_Duration')
    plt.title('University_Year vs. Sleep_Duration with linear regression')
    plt.savefig("University_Year vs. Sleep_Duration.png")
    plt.show()
    print("saved linear regression University_Year vs. Sleep_Duration.png")
    plt.clf()
def show_decision_tree():
    query = """
        SELECT Age, Sleep_Duration, Study_Hours, Screen_Time, Caffeine_Intake, Physical_Activity, Sleep_Quality
        FROM student_sleep_patterns
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    col = ['Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 'Sleep_Quality']

    all_sleep_data = pd.DataFrame(rows, columns=col)
    all_sleep_data['Sleep_Quality_Status'] = all_sleep_data['Sleep_Quality'].apply(lambda x: 1 if x>7 else 0)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = all_sleep_data[['Age', 'Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity']]
    Y = all_sleep_data['Sleep_Quality_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = tree.DecisionTreeClassifier(random_state=42)
    model = model.fit(X_train, Y_train)
    print("model score is: ", model.score(X_test, Y_test))

    dot = tree.export_graphviz(model, feature_names=X.columns, class_names=['Slept real good', 'Slept not so good'])
    graph = graphviz.Source(dot)
    graph.render('decision_tree_graph')
    graph.view()
def print_main_menu():
    print(" ")
    print("-----MAIN MENU-----")
    print("Please select a query type by typing its number")
    print("Enter 0 to exit at anytime")
    print("(0) Exit")
    print("(1) Load Database")
    print("(2) Reload Database")
    print("--- Relational Queries ---")
    print("(3) What is the average sleep duration given a gender and university year")
    print("(4) Find students’ sleep hours who have X caffeinated beverage per day")
    print("(5) Find caffeine consumption of students who have XXX minutes of exercise")
    print("(6) Find all students per a certain university year who get above 6 hours of sleep")
    print("(7) Find students with a sleep quality of above a 7/10 with the sleep quantity of less than 6 hours")
    print("--- Non-Relational Queries ---")
    print("(8) Use linear regression to plot the relation between study hours and sleep hours")
    print("(9) Predict sleep quality given study hours, caffeine consumption, and sleep duration using linear regression")
    print("(10) Use linear regression to plot the relation between physical activity and sleep hours")
    print("(11) Visualize the decision tree model with the target column being Sleep Quality>7)")
    print("(12) Predict sleep duration using linear regression given screen time, caffeine intake, and physical activity") 
def predict_based_on_fitness():
    # sql side
    query = """
        SELECT Sleep_Duration, Screen_Time, Caffeine_Intake, Physical_Activity
        FROM student_sleep_patterns
    """
    cur = connection.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    #pd/sklearn create dataframe and create model
    col = ['Sleep_Duration', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity']
    sleep_data = pd.DataFrame(rows, columns=col)
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None
    X = sleep_data[['Screen_Time', 'Caffeine_Intake', 'Physical_Activity']]
    Y = sleep_data['Sleep_Duration']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    # error scores
    mean_sq_error = mean_squared_error(Y_test, Y_pred)
    root_mean_sq_error = math.sqrt(mean_squared_error(Y_test, y_pred=Y_pred))
    print(f"Model error scores: mean square {mean_sq_error} and root msq: {root_mean_sq_error}")
    # user input prediction
    screen_hours = float(input("Please enter screen many hours per day input form (X.X): "))
    caffeine_intake = int(input("Please enter caffeine drink intake per day (Integer input): "))
    phys_activity = float(input("Enter physical activity duration in hours as a float (X.X): "))
    user_answer = pd.DataFrame([[screen_hours, caffeine_intake, phys_activity]], columns=['Screen_Time', 'Caffeine_Intake', 'Physical_Activity']) 
    user_prediction = model.predict(user_answer)
    print(f"The model predicts you slept: {user_prediction[0]:.2f} hours based on your fitness inputs")
load_data()
while True:
    print_main_menu()
    main_choice = int(input("Select a choice: "))
    if main_choice == 0:
        connection.close()
        break
    elif main_choice == 1:
        load_data()
    elif main_choice == 2:
        reload_data()
    elif main_choice == 3:
        average()
    elif main_choice == 4:
        caffeine()
    elif main_choice == 5:
        caffeine_exercise() 
    elif main_choice == 6:
        find_above_six_hours_sleep()
    elif main_choice == 7:
        find_sleep_qual_quan()
    elif main_choice == 8:
        lr_on_study_sleep_hours()
    elif main_choice == 9:
        predict_sleep_quality()
    elif main_choice == 10:
        physactivity_vs_sleep_duration()
    elif main_choice == 11:
        show_decision_tree()
    elif main_choice == 12:
        predict_based_on_fitness()


#query_choice = int(input()) # if 1 its relational 2 its non relational


    
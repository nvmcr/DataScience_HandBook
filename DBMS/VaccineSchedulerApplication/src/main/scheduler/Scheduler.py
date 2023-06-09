from model.Vaccine import Vaccine
from model.Caregiver import Caregiver
from model.Patient import Patient
from util.Util import Util
from db.ConnectionManager import ConnectionManager
import pymssql
import datetime


'''
objects to keep track of the currently logged-in user
Note: it is always true that at most one of currentCaregiver and currentPatient is not null
        since only one user can be logged-in at a time
'''
current_patient = None

current_caregiver = None


def create_patient(tokens):
    """
    TODO: Part 1
    """
    if len(tokens) != 3:
        print("Failed to create given user!")
        return

    username = tokens[1]
    password = tokens[2]

    password=str(password.strip())
    

    # Extra credit strong password check
    if not Util.is_strong_password(password):
        print("Password should be at least 8 characters \n A mixture of both uppercase and lowercase letters. \n A mixture of letters and numbers.\n Inclusion of at least one special character, from “!”, “@”, “#”, “?”.")
        return

    # check if the username has been taken already
    if username_exists_patient(username):
        print("Username taken, try again!")
        return

    salt = Util.generate_salt()
    hash = Util.generate_hash(password, salt)

    # create the patient
    patient = Patient(username, salt=salt, hash=hash)

    # save patient information to our database
    try:
        patient.save_to_db()
    except pymssql.Error as e:
        print("Failed to create given user!")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Failed to create given user!")
        print(e)
        return
    print("Created user ", username)

def username_exists_patient(username):
    cm = ConnectionManager()
    conn = cm.create_connection()

    select_username = "SELECT * FROM Patient WHERE Username = %s"
    try:
        cursor = conn.cursor(as_dict=True)
        cursor.execute(select_username, username)
        #  returns false if the cursor is not before the first record or if there are no rows in the ResultSet.
        for row in cursor:
            return row['Username'] is not None
    except pymssql.Error as e:
        print("Error occurred when checking username")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Error occurred when checking username")
        print("Error:", e)
    finally:
        cm.close_connection()
    return False


def create_caregiver(tokens):
    # create_caregiver <username> <password>
    # check 1: the length for tokens need to be exactly 3 to include all information (with the operation name)
    if len(tokens) != 3:
        print("Failed to create user.")
        return

    username = tokens[1]
    password = tokens[2]
    # check 2: check if the username has been taken already
    if username_exists_caregiver(username):
        print("Username taken, try again!")
        return

    salt = Util.generate_salt()
    hash = Util.generate_hash(password, salt)

    # create the caregiver
    caregiver = Caregiver(username, salt=salt, hash=hash)

    # save to caregiver information to our database
    try:
        caregiver.save_to_db()
    except pymssql.Error as e:
        print("Failed to create user.")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Failed to create user.")
        print(e)
        return
    print("Created user ", username)


def username_exists_caregiver(username):
    cm = ConnectionManager()
    conn = cm.create_connection()

    select_username = "SELECT * FROM Caregivers WHERE Username = %s"
    try:
        cursor = conn.cursor(as_dict=True)
        cursor.execute(select_username, username)
        #  returns false if the cursor is not before the first record or if there are no rows in the ResultSet.
        for row in cursor:
            return row['Username'] is not None
    except pymssql.Error as e:
        print("Error occurred when checking username")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Error occurred when checking username")
        print("Error:", e)
    finally:
        cm.close_connection()
    return False


def login_patient(tokens):
    """
    TODO: Part 1
    """
    global current_patient
    if current_caregiver is not None or current_patient is not None:
        print("User already logged in.")
        return

    if len(tokens) != 3:
        print("Login failed.")
        return

    username = tokens[1]
    password = tokens[2]

    patient = None
    try:
        patient = Patient(username, password=password).get()
    except pymssql.Error as e:
        print("Login failed.")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Login failed.")
        print("Error:", e)
        return

    # check if the login was successful
    if patient is None:
        print("Login failed.")
    else:
        print("Logged in as: " + username)
        current_patient = patient



def login_caregiver(tokens):
    # login_caregiver <username> <password>
    # check 1: if someone's already logged-in, they need to log out first
    global current_caregiver
    if current_caregiver is not None or current_patient is not None:
        print("User already logged in.")
        return

    # check 2: the length for tokens need to be exactly 3 to include all information (with the operation name)
    if len(tokens) != 3:
        print("Login failed.")
        return

    username = tokens[1]
    password = tokens[2]

    caregiver = None
    try:
        caregiver = Caregiver(username, password=password).get()
    except pymssql.Error as e:
        print("Login failed.")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Login failed.")
        print("Error:", e)
        return

    # check if the login was successful
    if caregiver is None:
        print("Login failed.")
    else:
        print("Logged in as: " + username)
        current_caregiver = caregiver


def search_caregiver_schedule(tokens):
    """
    TODO: Part 2
    """
    if current_patient is None and current_caregiver is None:
        print("Please login first")
        return

    cm = ConnectionManager()
    conn = cm.create_connection()
    search_caregiver = "SELECT A.Username, V.Name, V.Doses FROM Availabilities AS A, Vaccines AS V WHERE A.Time = %s ORDER BY A.Username;"
    try:
        cursor = conn.cursor(as_dict=True)
        cursor.execute(search_caregiver,tokens[1])
        for row in cursor:
            print(row['Username']+" "+row['Name']+" "+str(row['Doses']))
            
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()
    return


def reserve(tokens):
    """
    TODO: Part 2
    """
    if current_caregiver is None and current_patient is None:
        print("Please login first!")
        return

    if current_patient is None:
        print("Please login as patient!")
        return

      # check 2: the length for tokens need to be exactly 3 to include all information (with the operation name)
    if len(tokens) != 3:
        print("Please provide date and name of the vaccine for appointment")
        return
    
    requested_date = tokens[1]
    requested_Vaccine = tokens[2]
    avail_caregiver = None
    avail_dose=None
    appt_id = None

    cm = ConnectionManager()
    conn = cm.create_connection()
    search_caregiver = "SELECT TOP 1 A.Username FROM Availabilities AS A WHERE A.time=%s ORDER BY A.Username ASC;"
    try:
        cursor = conn.cursor(as_dict=True)
        cursor.execute(search_caregiver,requested_date)
        if cursor.rowcount ==0:
            print('No Caregiver is available!')
            return
        avail_caregiver = cursor.fetchone()['Username']
        # print("available caregiver is",avail_caregiver)
        conn.commit()
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()


    # Check if number of doses for vaccines is enough
    cm = ConnectionManager()
    conn = cm.create_connection()
    cursor = conn.cursor()
    cursor = conn.cursor(as_dict=True)

    search_avail_doses = "SELECT V.Doses FROM Vaccines AS V WHERE V.Name=%s;"
    try: 
        cursor.execute(search_avail_doses,requested_Vaccine)
        avail_dose = cursor.fetchone()['Doses']
        # print("available Doses is",avail_dose)
        conn.commit()
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()

    if avail_dose<=0:
        print("Not enough available doses!")
        return
    
    # inserting into appointments table
    cm = ConnectionManager()
    conn = cm.create_connection()
    cursor = conn.cursor()
    cursor = conn.cursor(as_dict=True)
    appt_id = Util.gen_appt_id(requested_date)
    

    insert_to_appointments = "INSERT INTO Appointments VALUES(%s, (SELECT Username FROM Caregivers WHERE Username=%s),(SELECT Username FROM Patient WHERE Username=%s),(SELECT Name FROM Vaccines WHERE Name=%s));"
    try: 
        cursor.execute(insert_to_appointments,(appt_id,avail_caregiver,current_patient.get_username(),requested_Vaccine))
        conn.commit()
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()


    # decrease vaccine doses
    cm = ConnectionManager()
    conn = cm.create_connection()
    cursor = conn.cursor()
    cursor = conn.cursor(as_dict=True)
    dec_vaccine_dose = "Update Vaccines set Doses=Doses-1 WHERE Name=%s AND Doses>0;"
    try: 
        cursor.execute(dec_vaccine_dose,requested_Vaccine)
        conn.commit()
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()
    

    # Updating availability table (deleting caregiver)
    cm = ConnectionManager()
    conn = cm.create_connection()
    cursor = conn.cursor()
    cursor = conn.cursor(as_dict=True)
    

    delete_avail = "DELETE FROM Availabilities WHERE Username=%s AND Time=%s;"
    try: 
        cursor.execute(delete_avail,(avail_caregiver,requested_date))
        print("Appointment ID: {}, Caregiver username: {}" .format(appt_id,avail_caregiver))
        conn.commit()
    except Exception as e:
        print("Please try again!")
        print("Error:", e)
        return
    finally:
        cm.close_connection()
   
    return


def upload_availability(tokens):
    #  upload_availability <date>
    #  check 1: check if the current logged-in user is a caregiver
    global current_caregiver
    if current_caregiver is None:
        print("Please login as a caregiver first!")
        return

    # check 2: the length for tokens need to be exactly 2 to include all information (with the operation name)
    if len(tokens) != 2:
        print("Please try again!")
        return

    date = tokens[1]
    # assume input is hyphenated in the format mm-dd-yyyy
    date_tokens = date.split("-")
    month = int(date_tokens[0])
    day = int(date_tokens[1])
    year = int(date_tokens[2])
    try:
        d = datetime.datetime(year, month, day)
        current_caregiver.upload_availability(d)
    except pymssql.Error as e:
        print("Upload Availability Failed")
        print("Db-Error:", e)
        quit()
    except ValueError:
        print("Please enter a valid date!")
        return
    except Exception as e:
        print("Error occurred when uploading availability")
        print("Error:", e)
        return
    print("Availability uploaded!")


def cancel(tokens):
    """
    TODO: Extra Credit
    """
    pass


def add_doses(tokens):
    #  add_doses <vaccine> <number>
    #  check 1: check if the current logged-in user is a caregiver
    global current_caregiver
    if current_caregiver is None:
        print("Please login as a caregiver first!")
        return

    #  check 2: the length for tokens need to be exactly 3 to include all information (with the operation name)
    if len(tokens) != 3:
        print("Please try again!")
        return

    vaccine_name = tokens[1]
    doses = int(tokens[2])
    vaccine = None
    try:
        vaccine = Vaccine(vaccine_name, doses).get()
    except pymssql.Error as e:
        print("Error occurred when adding doses")
        print("Db-Error:", e)
        quit()
    except Exception as e:
        print("Error occurred when adding doses")
        print("Error:", e)
        return

    # if the vaccine is not found in the database, add a new (vaccine, doses) entry.
    # else, update the existing entry by adding the new doses
    if vaccine is None:
        vaccine = Vaccine(vaccine_name, doses)
        try:
            vaccine.save_to_db()
        except pymssql.Error as e:
            print("Error occurred when adding doses")
            print("Db-Error:", e)
            quit()
        except Exception as e:
            print("Error occurred when adding doses")
            print("Error:", e)
            return
    else:
        # if the vaccine is not null, meaning that the vaccine already exists in our table
        try:
            vaccine.increase_available_doses(doses)
        except pymssql.Error as e:
            print("Error occurred when adding doses")
            print("Db-Error:", e)
            quit()
        except Exception as e:
            print("Error occurred when adding doses")
            print("Error:", e)
            return
    print("Doses updated!")


def show_appointments(tokens):
    '''
    TODO: Part 2
    '''
    if current_caregiver is None and current_patient is None:
        print("Please login first!")
        return

    if len(tokens) != 1:
        print("Invalid command. Please make sure the length of command is right")
        return

    if current_caregiver is not None:

        cm = ConnectionManager()
        conn = cm.create_connection()
        search_caregiver = "SELECT * FROM Appointments WHERE caregiverName=%s;"
        try:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(search_caregiver,current_caregiver.get_username())
            for row in cursor:
                print("Appointment ID:"+row['appointmentID']+"\t Vaccine:"+ row['vaccineName']+"\tdate:"+ Util.get_date(row['appointmentID']) + "\t Patient name:"+ row['patientName'])
                
        except Exception as e:
            print("Please try again!")
            print("Error:", e)
            return
        finally:
            cm.close_connection()
        return
    elif current_patient is not None:
        # for patient
        cm = ConnectionManager()
        conn = cm.create_connection()
        search_caregiver = "SELECT * FROM Appointments WHERE patientName=%s;"
        try:
            cursor = conn.cursor(as_dict=True)
            cursor.execute(search_caregiver,current_patient.get_username())
            for row in cursor:
                print("Appointment ID:"+row['appointmentID']+"\t Vaccine:"+ row['vaccineName']+"\tdate:"+ Util.get_date(row['appointmentID']) + "\t caregiver name:"+ row['caregiverName'])
                
        except Exception as e:
            print("Please try again!")
            print("Error:", e)
            return
        finally:
            cm.close_connection()
        return
        
    else: 
        print("Please login first!")
    return


def logout(tokens):
    """
    TODO: Part 2
    """
    if len(tokens) != 1:
        print("Invalid command. Please make sure the length of command is right")
        return
    
    global current_caregiver,current_patient 
    
    try:
        if current_caregiver is not None:
            print("logging out caregiver:",str(Caregiver.get_username(current_caregiver)))
            current_caregiver = None
            print("Successfully logged out!")
            return
        elif current_patient is not None:
            print("Logging out patient : ",str(Patient.get_username(current_patient)))
            current_patient = None
            print("Successfully logged out!")
            return
        else:
            print("Please login first!")
    except Exception as e:
            print("Please try again!")
            print("Error:", e)
            return


def start():
    stop = False
    print()
    print(" *** Please enter one of the following commands *** ")
    print("> create_patient <username> <password>")  # //TODO: implement create_patient (Part 1)
    print("> create_caregiver <username> <password>")
    print("> login_patient <username> <password>")  # // TODO: implement login_patient (Part 1)
    print("> login_caregiver <username> <password>")
    print("> search_caregiver_schedule <date>")  # // TODO: implement search_caregiver_schedule (Part 2)
    print("> reserve <date> <vaccine>")  # // TODO: implement reserve (Part 2)
    print("> upload_availability <date>")
    print("> cancel <appointment_id>")  # // TODO: implement cancel (extra credit)
    print("> add_doses <vaccine> <number>")
    print("> show_appointments")  # // TODO: implement show_appointments (Part 2)
    print("> logout")  # // TODO: implement logout (Part 2)
    print("> Quit")
    print()
    while not stop:
        response = ""
        print("> ", end='')

        try:
            response = str(input())
        except ValueError:
            print("Please try again!")
            break

        response = response.lower()
        tokens = response.split(" ")
        if len(tokens) == 0:
            ValueError("Please try again!")
            continue
        operation = tokens[0]
        if operation == "create_patient":
            create_patient(tokens)
        elif operation == "create_caregiver":
            create_caregiver(tokens)
        elif operation == "login_patient":
            login_patient(tokens)
        elif operation == "login_caregiver":
            login_caregiver(tokens)
        elif operation == "search_caregiver_schedule":
            search_caregiver_schedule(tokens)
        elif operation == "reserve":
            reserve(tokens)
        elif operation == "upload_availability":
            upload_availability(tokens)
        elif operation == cancel:
            cancel(tokens)
        elif operation == "add_doses":
            add_doses(tokens)
        elif operation == "show_appointments":
            show_appointments(tokens)
        elif operation == "logout":
            logout(tokens)
        elif operation == "quit":
            print("Bye!")
            stop = True
        else:
            print("Invalid operation name!")


if __name__ == "__main__":
    '''
    // pre-define the three types of authorized vaccines
    // note: it's a poor practice to hard-code these values, but we will do this ]
    // for the simplicity of this assignment
    // and then construct a map of vaccineName -> vaccineObject
    '''

    # start command line
    print()
    print("Welcome to the COVID-19 Vaccine Reservation Scheduling Application!")

    start()

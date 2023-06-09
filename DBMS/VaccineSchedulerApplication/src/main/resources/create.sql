CREATE TABLE Caregivers (
    Username varchar(255),
    Salt BINARY(16),
    Hash BINARY(16),
    PRIMARY KEY (Username)
);

CREATE TABLE Availabilities (
    Time date,
    Username varchar(255) REFERENCES Caregivers,
    PRIMARY KEY (Time, Username)
);

CREATE TABLE Vaccines (
    Name varchar(255),
    Doses int,
    PRIMARY KEY (Name)
);

CREATE TABLE Patient (
    Username varchar(255),
    Salt BINARY(16),
    Hash BINARY(16),
    PRIMARY KEY (Username)
);

CREATE TABLE Appointments (
    appointmentID varchar(255),
    caregiverName varchar(255) REFERENCES Caregivers(Username),
    patientName varchar(255) REFERENCES Patient(Username),
    vaccineName varchar(255) REFERENCES Vaccines(Name),
    PRIMARY KEY (appointmentID)
);
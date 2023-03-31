Q1 
First, create a simple table using the following steps:

Q2.1

Write the SQL statement to create a table Edges(Source, Destination) where both Source and Destination are integers.
```
CREATE TABLE Edges (
    Source INT,
    Destination INT
   );
```
Q2.2

Write the SQL statement(s) to insert the tuples (10,5), (6,25), (1,3), and (4,4)
```
INSERT INTO Edges
VALUES 
   (10, 5), (6, 25), (1, 3), (4, 4);
```
Q2.3

Write a SQL statement that returns all tuples.
```
SELECT *
FROM Edges;
```
Q2.4

Write a SQL statement that returns only column Source for all tuples.
```
SELECT Source
FROM Edges;
```
Q2.5

Write a SQL statement that returns all tuples where Source > Destination.
```
SELECT *
FROM Edges
WHERE Source > Destination;
```
Q2.6

Now write a query to insert the tuple ('-1','2000'). Do you get an error? Does this behavior match what you would expect based on the relational model? If not, why? This is a tricky question, you might want to check the documentation.
```
INSERT INTO Edges VALUES ('-1', '2000');
```
It doesn't give an error.  I wouldn't expect this from a relational model as we gave string values in the integer-specified columns. But when I queried the table values, it is stored as integers itself. This is explained under Type Affinity in SQLite3 documentation.
Q3

Next, you will create a table with attributes of types integer, varchar, date, and Boolean. However, SQLite does not have date and Boolean: you will use varchar and int instead. 
Some notes:

0 (false) and 1 (true) are the values used to interpret Booleans. 
Date strings in SQLite are in the form: 'YYYY-MM-DD'. 
Examples of valid date strings include: '1988-01-15', '0000-12-31', and '2011-03-28'. 
Examples of invalid date strings include: '11-11-01', '1900-1-20', '2011-03-5', and '2011-03-50'. 
Examples of date operations on date strings (feel free to try them): 
select date('2011-03-28'); 
select date('now'); 
select date('now', '-5 year'); 
select date('now', '-5 year', '+24 hour'); 
select case when date('now') < date('2011-12-09') then 'Taking classes' when date('now') < date('2011-12-16') then 'Exams' else 'Vacation' end;

Create a table called MyRestaurants with the following attributes (you can pick your own names for the attributes, just make sure it is clear which one is for which):

Name of the restaurant: a varchar field 
Type of food they make: a varchar field 
Distance (in minutes) from your house: an int 
Date of your last visit: a varchar field, interpreted as date 
Whether you like it or not: an int, interpreted as a Boolean
```
CREATE TABLE MyRestaurants (
    Name VARCHAR,
    FoodType VARCHAR,
    DistanceInMin INT,
    DateOfLastVisit VARCHAR,
    Liking INT
);
```
Q41

Insert at least five tuples using the SQL INSERT statement. 
 You should write five (or more) INSERT statements.

Insert at least one restaurant you liked, at least one restaurant you did not like, and at least one restaurant where you leave the “I like” field NULL.
```
INSERT INTO MyRestaurants
VALUES
   ('Taste of India', 'Indian', 5, '2023-03-12',1),
   ('Aladdin', 'Middle Eastern', 7, '2023-03-16',1),
   ('Shawarma Times', 'Middle Eastern', 10, '2023-02-27',0),
   ('Panda Noodle Bar', 'Chinese', 7, '2023-03-14',1),
   ('Jewel of India', 'Indian', 8, NULL, NULL)
;
```
Q51

Below you will write a SQL query that returns every row in your MyRestaurants table. Experiment with a few of SQLite's output formats and repeat this query six times, formatting the output in the following ways.

Remember to include both the commands you use to format the output as well as the SQL query. When we run your code for 5.1 and 5.2 we should see the table printed 6 times with the specified formatting.

Q5.1

Write the code (SQL and Sqlite commands) to turn column headers on, then return the results in these three formats:
1. print the results in comma-separated form
2. print the results in list form, delimited by "|"
3. print the results in column form and make every column has a width >= 15.  Ensure that every column has this width, not just the first one.
```
.headers on
.mode csv
SELECT * FROM MyRestaurants;

.mode list
SELECT * FROM MyRestaurants;

.mode column
.width 17 15 13 15 10
SELECT * FROM MyRestaurants;
```
Q5.2

Now write the code to turn column headers off, and return the results again in the three formats.
```
.headers off
.mode csv
SELECT * FROM MyRestaurants;

.mode list
SELECT * FROM MyRestaurants;

.mode column
.width 17 15 13 15 10
SELECT * FROM MyRestaurants;
```
Q6

Write a SQL query that returns only the name and distance of all restaurants within and including 20 minutes of your house. The query should list the restaurants in alphabetical order of names.
```
SELECT Name, DistanceInMin AS D 
FROM MyRestaurants
WHERE D <= 20
ORDER BY Name ASC;
```
Q7

Write a SQL query that returns all restaurants that you like, but has not visited since more than 3 months ago. Make sure that you use the date() function to calculate the date 3 months ago.
```
SELECT * FROM MyRestaurants
WHERE Liking = 1 
              AND date(DateOfLastVisit) < date('now', '-3 months');
```
Q8

Write a SQL query that returns all restaurants that are within and including 10 mins from your house.
```
SELECT * FROM MyRestaurants
WHERE DistanceInMin <= 10;
```

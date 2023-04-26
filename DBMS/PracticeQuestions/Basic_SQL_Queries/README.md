# Basic SQL Queries
 
**Objectives:** To create and import databases and to practice simple SQL queries using SQLite.

## Assignment Details
The data in this database is abridged from the [Bureau of Transportation Statistics](http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time) 
The database consists of four tables regarding a subset of flights that took place in 2015:

```SQL
FLIGHTS (fid int, 
         month_id int,        -- 1-12
         day_of_month int,    -- 1-31 
         day_of_week_id int,  -- 1-7, 1 = Monday, 2 = Tuesday, etc
         carrier_id varchar(7), 
         flight_num int,
         origin_city varchar(34), 
         origin_state varchar(47), 
         dest_city varchar(34), 
         dest_state varchar(46), 
         departure_delay int, -- in mins
         taxi_out int,        -- in mins
         arrival_delay int,   -- in mins
         canceled int,        -- 1 means canceled
         actual_time int,     -- in mins
         distance int,        -- in miles
         capacity int, 
         price int            -- in $             
         )
         
CARRIERS (cid varchar(7), name varchar(83))
MONTHS (mid int, month varchar(9))
WEEKDAYS (did int, day_of_week varchar(9))
```
In addition, make sure you impose the following constraints to the tables above:
- The primary key of the `FLIGHTS` table is `fid`.
- The primary keys for the other tables are `cid`, `mid`, and `did` respectively. Other than these, *do not assume any other attribute(s) is a key / unique across tuples.*
- `Flights.carrier_id` references `Carriers.cid`
- `Flights.month_id` references `Months.mid`
- `Flights.day_of_week_id` references `Weekdays.did`

We provide the flights database as a set of plain-text data files in the linked 
`.tar.gz` archive. Each file in this archive contains all the rows for the named table, one row per line.

In this homework, you need to do two things:
1. import the flights dataset into SQLite
2. run SQL queries to answer a set of questions about the data.


### IMPORTING THE FLIGHTS DATABASE (20 points)

To import the flights database into SQLite, you will need to run sqlite3 with a new database file.
for example `sqlite3 hw2.db`. Then you can run `CREATE TABLE` statement to create the tables, 
choosing appropriate types for each column and specifying all key constraints as described above:

```SQL
CREATE TABLE IF NOT EXISTS CARRIERS (
    cid varchar(7) PRIMARY KEY,
    name varchar(83)
    );

CREATE TABLE IF NOT EXISTS MONTHS (
    mid int PRIMARY KEY,
    month varchar(9)
    );

CREATE TABLE IF NOT EXISTS WEEKDAYS (
    did int PRIMARY KEY,
    day_of_week varchar(9)
    );

PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS FLIGHTS (
    fid int PRIMARY KEY,
    month_id int REFERENCES MONTHS,
    day_of_month int CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_week_id int REFERENCES WEEKDAYS,
    carrier_id varchar(7) REFERENCES CARRIERS,
    flight_num int,
    origin_city varchar(34),
    origin_state varchar(47),
    dest_city varchar(34),
    dest_state varchar(46),
    departure_delay int,
    taxi_out int,
    arrival_delay int,
    canceled int,
    actual_time int,
    distance int,
    capacity int,
    price int
    )
;
```

Currently, SQLite does not enforce foreign keys by default. 
To enable foreign keys use the following command. 
The command will have no effect if you installed your own version of SQLite was not compiled with foreign keys enabled. 
In that case do not worry about it (i.e., you will need to enforce foreign key constraints yourself as 
you insert data into the table).

```
PRAGMA foreign_keys=ON;
```

Then, you can use the SQLite `.import` command to read data from each text file into its table after setting the input data to be in CSV (comma separated value) form:

```SQL
.mode csv

.import flight-dataset/carrier.csv CARRIERS
.import flight-dataset/months.csv MONTHS
.import flight-dataset/weekdays.csv WEEKDAYS
.import flight-dataset/flights-small.csv FLIGHTS
```

See examples of `.import` statements in the section notes, and also look at the SQLite 
documentation or sqlite3's help online for details.

Put all the code for this part (four `create table` statements and four `.import` statements) 
into a file called `create-tables.sql` inside the `hw2/submission` directory.


### Writing SQL QUERIES

**HINT: You should be able to answer all the questions below with SQL queries that do NOT contain any subqueries!**

**Important: The predicates in your queries should correspond to the English descriptions. For example, if a question asks you to find flights by Alaska Airlines Inc., the query should 
include a predicate that checks for that specific name as opposed to checking for the matching carrier ID. Same for predicates over months, weekdays, etc.**

**Also, make sure you name the output columns as indicated! Do not change the output column names / return more or fewer columns!**

In the following questions below flights **include canceled flights as well, unless otherwise noted.** 
Also, when asked to output times you can report them in minutes and don’t need to do minute-hour conversion.

If a query uses a `GROUP BY` clause, make sure that all attributes in your `SELECT` clause for that query 
are either grouping keys or aggregate values. SQLite will let you select other attributes but that is wrong
as we discussed in lectures. Other database systems would reject the query in that case. 


1. (10 points) List the distinct flight numbers of all flights from Seattle to Boston by Alaska Airlines Inc. on Mondays. 
Also notice that, in the database, the city names include the state. So Seattle appears as 
Seattle WA.  
   Name the output column `flight_num`.    
   [Hint: Output relation cardinality: 3 rows]
   
```SQL
SELECT DISTINCT F.flight_num AS flight_num
FROM FLIGHTS AS F
   JOIN CARRIERS AS C
      ON F.carrier_id = C.cid
   JOIN WEEKDAYS AS W
      ON F.day_of_week_id = W.did
WHERE F.origin_city = "Seattle WA"
   AND F.dest_city = "Boston MA"
   AND C.name = "Alaska Airlines Inc."
   AND W.day_of_week = "Monday"
;

/* Output

flight_num
12
24
734

*/
```

2. (10 points) Find all itineraries from Seattle to Boston on July 15th. Search only for itineraries that have one stop (i.e., flight 1: Seattle -> [somewhere], flight2: [somewhere] -> Boston).
Both flights must depart on the same day (same day here means the date of flight) and must be with the same carrier. It's fine if the landing date is different from the departing date (i.e., in the case of an overnight flight). You don't need to check whether the first flight overlaps with the second one since the departing and arriving time of the flights are not provided. 
 
The total flight time (`actual_time`) of the entire itinerary should be fewer than 7 hours 
(but notice that `actual_time` is in minutes). 
   For each itinerary, the query should return the name of the carrier, the first flight number, 
the origin and destination of that first flight, the flight time, the second flight number, 
the origin and destination of the second flight, the second flight time, and finally the total flight time. 
Only count flight times here; do not include any layover time.

Name the output columns `name` as the name of the carrier, `f1_flight_num`, `f1_origin_city`, `f1_dest_city`, `f1_actual_time`, `f2_flight_num`, `f2_origin_city`, `f2_dest_city`, `f2_actual_time`, and `actual_time` as the total flight time. List the output columns in this order.
    [Output relation cardinality: 1472 rows]

```SQL

SELECT C.name AS name,	
    F1.flight_num AS f1_flight_num,
    F1.origin_city AS f1_origin_city,
    F1.dest_city AS f1_dest_city,
    F1.actual_time AS f1_actual_time,
    
    F2.flight_num AS f2_flight_num,
    F2.origin_city AS f2_origin_city,
    F2.dest_city AS f2_dest_city,
    F2.actual_time AS f2_actual_time,
    
    (F1.actual_time + F2.actual_time) AS actual_time

FROM FLIGHTS AS F1

   JOIN FLIGHTS F2 ON F1.dest_city = F2.origin_city
   JOIN MONTHS AS M ON F1.month_id = M.mid AND F2.month_id = M.mid
   JOIN CARRIERS AS C ON F1.carrier_id = C.cid AND F2.carrier_id = C.cid

WHERE F1.origin_city = "Seattle WA"
    AND F1.dest_city = F2.origin_city
    AND F2.dest_city = "Boston MA"
    AND M.month = "July"
    AND F1.day_of_month = 15
    AND F2.day_of_month = F1.day_of_month --Depart on same day
    AND F1.carrier_id = F2.carrier_id --SAme carrier for both flights
    AND (F1.actual_time + F2.actual_time) < 420
;
```

3. (10 points) Find the day of the week with the longest average arrival delay. 
Return the name of the day and the average delay.   
   Name the output columns `day_of_week` and `delay`, in that order. (Hint: consider using `LIMIT`. Look up what it does!)   
   [Output relation cardinality: 1 row]

```SQL
SELECT W.day_of_week, AVG(arrival_delay)
FROM FLIGHTS AS F
   JOIN WEEKDAYS AS W 
      ON F.day_of_week_id = W.did
GROUP BY W.day_of_week
ORDER BY AVG(arrival_delay) DESC
LIMIT 1
;

/* OUTPUT

day_of_week,AVG(arrival_delay)
Friday,14.4725010477787

*/
```

4. (10 points) Find the names of all airlines that ever flew more than 1000 flights in one day 
(i.e., a specific day/month, but not any 24-hour period). 
Return only the names of the airlines. Do not return any duplicates 
(i.e., airlines with the exact same name).    
   Name the output column `name`.   
   [Output relation cardinality: 12 rows]

```SQL
SELECT DISTINCT C.name AS name
FROM FLIGHTS AS F
    JOIN CARRIERS AS C
        ON F.carrier_id = C.cid
GROUP BY C.cid, F.month_id, F.day_of_month
    HAVING COUNT(*) > 1000
;

/* OUTPUT

name
"American Airlines Inc."
"JetBlue Airways"
"Delta Air Lines Inc."
"ExpressJet Airlines Inc."
"Envoy Air"
"Northwest Airlines Inc."
"Comair Inc."
"SkyWest Airlines Inc."
"United Air Lines Inc."
"US Airways Inc."
"Southwest Airlines Co."
"ExpressJet Airlines Inc. (1)"

*/
```

5. (10 points) Find all airlines that had more than 0.5 percent of their flights out of Seattle be canceled. 
Return the name of the airline and the percentage of canceled flight out of Seattle. 
Order the results by the percentage of canceled flights in ascending order.    
   Name the output columns `name` and `percent`, in that order.   
   [Output relation cardinality: 6 rows]

```SQL
SELECT C.name AS name, (SUM(F.canceled=1)*100.0)/COUNT(*) as percentage
FROM FLIGHTS AS F
    JOIN CARRIERS AS C
        ON F.carrier_id = C.cid
WHERE F.origin_city = "Seattle WA"
GROUP BY C.name,C.cid
    HAVING percentage > 0.5
ORDER BY percentage ASC
;

/* OUTPUT

name,percentage
"SkyWest Airlines Inc.",0.728291316526611
"Frontier Airlines Inc.",0.840336134453782
"United Air Lines Inc.",0.983767830791933
"JetBlue Airways",1.00250626566416
"Northwest Airlines Inc.",1.4336917562724
"ExpressJet Airlines Inc.",3.2258064516129

*/
```

6. (10 points) Find the maximum price of tickets between Seattle and New York, NY 
(i.e. Seattle to New York or New York to Seattle).
Show the maximum price for each airline separately.
   Name the output columns `carrier` and `max_price`, in that order.
   [Output relation cardinality: 3 rows]

```SQL
SELECT C.name AS carrier, MAX(F.price) AS max_price
FROM FLIGHTS AS F
    JOIN CARRIERS AS C
        ON F.carrier_id = C.cid
WHERE (F.origin_city = 'Seattle WA' AND F.dest_city = 'New York NY')
      OR (F.origin_city = 'New York NY' AND F.dest_city = 'Seattle WA')
GROUP BY C.cid, C.name
;

/* OUTPUT 

carrier,max_price
"American Airlines Inc.",991
"JetBlue Airways",996
"Delta Air Lines Inc.",999

*/
```

7. (10 points) Find the total capacity of all direct flights that fly between Seattle and San Francisco, CA on July 10th 
(i.e. Seattle to San Francisco or San Francisco to Seattle).
   Name the output column `capacity`.
   [Output relation cardinality: 1 row]

```SQL
SELECT SUM(F.capacity) as capacity
FROM Flights AS F
    JOIN MONTHS AS M
        ON F.month_id = M.mid
WHERE ((F.origin_city = 'San Francisco CA' AND F.dest_city = 'Seattle WA') 
	OR (F.origin_city = 'Seattle WA' AND F.dest_city = 'San Francisco CA'))
	AND M.month = "July"
	AND F.day_of_month = 10
;

/* OUTPUT

capacity
680

*/
```
   
8. (10 points) Compute the total departure delay of each airline 
across all flights.
   Name the output columns `name` and `delay`, in that order.
   [Output relation cardinality: 22 rows]

```SQL
SELECT C.name AS name, SUM(F.departure_delay) AS delay
FROM FLIGHTS AS F
    JOIN CARRIERS AS C 
        ON F.carrier_id = C.cid
GROUP BY C.name
;

/* OUTPUT

name,delay
"ATA Airlines d/b/a ATA",38676
"AirTran Airways Corporation",473993
"Alaska Airlines Inc.",285111
"America West Airlines Inc. (Merged with US Airways 9/05. Stopped reporting 10/07.)",173255
"American Airlines Inc.",1849386
"Comair Inc.",282042
"Continental Air Lines Inc.",414226
"Delta Air Lines Inc.",1601314
"Envoy Air",771679
"ExpressJet Airlines Inc.",934691
"ExpressJet Airlines Inc. (1)",483171
"Frontier Airlines Inc.",165126
"Hawaiian Airlines Inc.",386
"Independence Air",201418
"JetBlue Airways",435562
"Northwest Airlines Inc.",531356
"SkyWest Airlines Inc.",682158
"Southwest Airlines Co.",3056656
"Spirit Air Lines",167894
"US Airways Inc.",577268
"United Air Lines Inc.",1483777
"Virgin America",52597

*/
```

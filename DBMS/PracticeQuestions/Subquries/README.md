Using the same data tables used in [Basic SQL Quries](https://github.com/nvmcr/Reference_Guide/tree/main/DBMS/PracticeQuestions/Basic_SQL_Queries).

1. (Output relation cardinality: 334 rows)

For each origin city, find the destination city (or cities) with the longest direct flight (by direct flight, we mean a flight with no intermediate stops). Judge the longest flight using duration, not distance. (15 points)

Name the output columns origin_city, dest_city, and time (the flight duration). Do not include duplicates of (origin city, destination city) pairs. Order the result by origin_city and then dest_city (ascending, i.e. alphabetically).

```SQL
SELECT DISTINCT F1.origin_city AS origin_city,
                F1.dest_city AS dest_city,
                f1.actual_time AS time
FROM FLIGHTS AS F1
WHERE f1.actual_time = (
    SELECT MAX(F2.actual_time)
    FROM FLIGHTS AS F2
    WHERE F2.origin_city = F1.origin_city
    GROUP BY F2.origin_city
    )
ORDER BY F1.origin_city,F1.dest_city ASC;
```

2. (Output relation cardinality: 109 rows)

Find all origin cities that only serve flights shorter than 3 hours. You should not include canceled flights in your determination.

Name the output column city and sort them in ascending order alphabetically. List each city only once in the result.

```SQL
SELECT DISTINCT F.origin_city AS city
FROM FLIGHTS AS F
WHERE F.origin_city NOT IN (
    SELECT DISTINCT F.origin_city
    FROM FLIGHTS AS F
    WHERE F.actual_time >= 180 AND F.canceled !=1
    ) 
    
ORDER BY city ASC;
```

3. (Output relation cardinality: 327 rows)

For each origin city, find the percentage of departed flights whose duration is shorter than 3 hours; canceled flights do not count as having departed.  Be careful to handle cities which do not have any flights shorter than 3 hours; you should return 0 as the result for these cities, not NULL (which is shown as a blank cell in Azure). 

Name the output columns origin_city and percentage. Order by percentage value, then city, ascending. Report percentages as percentages, not decimals (e.g., report 75.2534 rather than 0.752534). Do not round the percentages.

```SQL
SELECT F1.origin_city AS origin_city, ISNULL((SELECT COUNT(F2.fid)
    FROM FLIGHTS as F2
    WHERE F2.actual_time < 180 
	AND F2.canceled = 0
    AND F1.origin_city = F2.origin_city
    GROUP BY F2.origin_city)* 100.0 /COUNT(*) ,0.0) AS percentage
FROM FLIGHTS AS F1
GROUP BY origin_city
ORDER BY percentage, origin_city ASC;
```

4. (Output relation cardinality: 256 rows)

List all cities that can be reached from Seattle using exactly one stop.  In other words, the flight itinerary should use an intermediate city, but cannot be reached through a direct flight.  Do not include Seattle as one of these destinations (even though you could get back with two flights).

Name the output column city. Order the output ascending by city.

```SQL
SELECT DISTINCT F2.dest_city AS city
FROM FLIGHTS AS F1
    JOIN FLIGHTS AS F2
        ON F2.origin_city = F1.dest_city
WHERE F1.origin_city = 'Seattle WA'
    AND F2.dest_city NOT IN (
        SELECT F3.dest_city
        FROM FLIGHTS AS F3
        WHERE F3.origin_city = 'Seattle WA'
    )
    AND F2.dest_city != 'Seattle WA'
ORDER BY city ASC;
```

5. (Output relation cardinality: 3 or 4 rows, depending on what you consider to be the set of all cities)

List all cities that can be reached from Seattle, but which require two intermediate stops or more .  Warning: this query might take a while to execute; we will learn about how to speed this up in lecture.  You can assume all cities to be the collection of all origin_city or all dest_city. You can also assume all cities are reachable from Seattle via a finite number of stops.

Name the output column city. Order the output ascending by city.

```SQL
SELECT DISTINCT F.dest_city AS city
FROM FLIGHTS AS F
WHERE F.dest_city NOT IN (
    SELECT DISTINCT F2.dest_city
    FROM FLIGHTS AS F1
        JOIN FLIGHTS AS F2
            ON F2.origin_city = F1.dest_city
    WHERE F1.origin_city = 'Seattle WA'
    )
    AND F.dest_city NOT IN (
            SELECT DISTINCT F3.dest_city
            FROM FLIGHTS AS F3
            WHERE F3.origin_city = 'Seattle WA'
        )
ORDER BY city ASC;
```

6. (Output relation cardinality: 4 rows)

List the names of carriers that operate flights from Seattle to San Francisco, CA.  Return each carrier's name only once, and use a nested query to answer this question. (7 points)
Name the output column carrier. Order the output ascending by carrier.

```SQL
SELECT DISTINCT C.name AS carrier
FROM CARRIERS AS C, (
        SELECT F.carrier_id AS carrrier_id
        FROM FLIGHTS AS F
        WHERE F.origin_city = 'Seattle WA' AND
            F.dest_city = 'San Francisco CA'
    ) AS F1
WHERE C.cid = F1.carrrier_id
ORDER BY carrier ASC;
```

7. (Output relation cardinality: 4 rows)

Express the same query as above, but do so without using a subquery. As before, name the output column carrier. Order the output ascending by carrier.

```SQL
SELECT DISTINCT C.name AS carrier
FROM FLIGHTS AS F
    JOIN CARRIERS AS C
        ON F.carrier_id = C.cid
WHERE F.origin_city = 'Seattle WA' AND F.dest_city = 'San Francisco CA'
ORDER BY carrier ASC;
```

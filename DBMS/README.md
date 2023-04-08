# DataBase Management Systems
DBMS is a big program written by someone else that allows us to manage efficiently a large database and allows it to persist over long periods of time. 
> A Data Model is a mathematical formalism to describe data. It is how we can talk about data conceptually without having to think about implementation.

There are 3 parts in a Data Model.
1. Instance : The actual data rows
2. Schema: A description of what data is being stored.
3. Query Language: How to retrieve or manipulate the data.
## Note
This README file contains only the concepts related to DBMS. All the practice questions are in this [link](https://github.com/nvmcr/Reference_Guide/tree/main/DBMS/PracticeQuestions).

<details>
<summary>Table of Contents>

## Table of Contents
1. [SQL](#SQL)
      1. [Intro](#intro)
      2. [SELECT](#select-queries)
      3. [Constraints, Filtering, Sorting](#constraints-filtering-sorting)
      5. [Multiple Table Queries](#multi-table-queries)
      6. [Queries with Expressions](#queries-with-expressions)
      7. [Order of Execution](#order-of-execution)
      8. [Modifying Rows](#Modifying-Rows)
            1. [Inserting Rows](#inserting-new-rows)
            2. [Updating Rows](#updating-existing-rows)
            3. [Deleting Rows](#deleting-rows)
      9. [Table Queries](#table-queries)
            1. [Creating Tables](#creating-tables)
            2. [Altering Tables](#altering-tables)
            3. [Dropping Tables](#dropping-tables)
      10. [Subqueries](#Subqueries)
            1. [Correlated Subqueries](#Correlated-Subqueries)
      11. [SET Queries](#set-queries)

</details>

# SQL
## Intro
Most popular data model is a relational data model. SQL is the query language used.
* What is a relational database? 
> A realtional database represents a collection of tables. For example, an university realtional database might have a table with `StudentId`, `Name`, `YearofPassing`, `GPA` and many other columns. There can be many other *related* tabes in that database. 

Structured Query Language (SQL) is a language designed for users to query, manipulate and transform data from a relational database. Biggest advantage of SQL is that it is efficient and scalable for large and complex databases. Popular SQL databases include: SQLite, MySQL, PostgreSQL, Oracle and Microsoft SQL Server. 

## SELECT Queries
`SELECT` statements are used to retrieve data from a SQL database. These statements are often refered as *queries*.
> A query is just a statement that tells the database what we want. It could be to retrieve, update or modify the data.
Queries have syntax. Say, there is a table named `table1` which has details of students. To retrieve specific columns the qury will look like:
```
SELECT name, gpa
FROM table1;
```
If we need to see the entire table, asterisk (\*) can be used.
```
SELECT *
FROM table1;
```
The `SELECT` and `FROM` statements need not be seperate lines they can be used in a single line too.
```
SELECT year_of_passing, name, gpa FROM table1;
```
> The semi-colan (;) marje the end of a SQL statement similar to C. But in some databased it is optional. SQL is not case sensitive atleast not for keywords. Also it does not require indentation. But its better to use `;` at the end of statements, make keywords capitalized and use indentation for readability.
## Constraints, Filtering, Sorting
What if we don't need all the rows. Just like we *select* required columns, we can retrieve only the required rows using `WHERE` keyword.
```
SELECT name, gpa, year_of_passing
FROM table1
WHERE gpa > 3
      AND year_of_passing >= 2016;
```
We can compare above query with python for loop. `FROM` is analogus to `for`, `WHERE` to `if` and `SELECT` to the operation.
```
for each row in table1:
      if gpa>3:
            print(table1.name, table1.gpa, table1.year)
```
Common operators that can be used include:
|Operator|Example|
|-------|-------|
|=, !=(or)<>, <, >, <=, >=|col_name = 5|
|BETWEEN ... AND ...|col_name BETWEEN 2012 AND 2016|
|NOT BETWEEN ... AND ...|col_name NOT BETWEEN 2012 and 2016|
|IN|col_name IN (1,2,3,4,5)|
|IS NULL, IS NOT NULL|col_name IS NULL|
|LIKE|col_name LIKE 'ME%'|
> `LIKE` is used to filter data based on a specific pattern. It can use wildcard characters like `%` and `_`. `LIKE 'M%'` retrieves every row in that column that **contains** 'M' and `LIKE 'Me_ha'` retrieves every value **strictly** has 'Me' in front and a *single* middle string and 'ha' at the end. Use multiple underscores to retrieve multiple characters. 

> All strings should be represneted within single or double quotations. 
Many times databases are filled with duplicate values. To retrieve only distinct values, `DISTINCT` keyword is used.
```
SELECT DISTINCT gpa, name FROM table1;
```
Also databases agre generally not ordered. To arrange the rows, we can order them by a specific column using `ORDER BY col_name ASC/DESC` clause. ALong with this `LIMIT` and `OFFSET` are commonly used together. 
```
SELECT name, gpa
FROM table1
ORDER BY Id DESC
LIMIT 10 OFFSET 50;
```
After 50 rows, next 10 rows will be returned.
> The `LIMIT` will return the specified number of rows and `OFFSET` will specify where the `LIMIT` count should start from.
## Multi-Table Queries
In real world data, the data is broken down into pieces and stord across multiple orthogonal tables using *normalization*. This database normalization is useful as it minimizes duplicate data in a single table and also allows the data to grow independent of each other. 
For example, if there are two tables `table1` and `table2` where `table1` has `id`, `name`, `gpa` and `table2` has `studentid`, `state`, `city`. If we need top 10 highest gpa students from Texas, query looks like:
```
SELECT name, gpa
FROM table1
  INNER JOIN table2
    ON table1.id = table2.studentid
WHERE state = 'Texas'
ORDER BY gpa
LIMIT 10;
```
`ON` condition specifies how the tables need to be joined. `INNER JOIN`/`JOIN` will join only the rows that are common to both tables. Once the tables are joined, remaining keywords can be used similar to a single table. 
Say we have following query:
```
SELECT t1.name, t2.gpa
FROM table1 AS t1
   JOIN table2 as t2
      ON t1.id = t2.studentid
```
The same query can be implicity written as:
```
SELECT t1.name, t2.gpa
FROM table1 AS t1, table2 AS t2
WHERE t1.id = t2.studentid
```
When we think of above both query in programming sense, the execution looks like:
```
for row1 in table1:
   for row2 in table2:
      if row1.id == row2.studnetid:
         print(row1.name, row2.gpa)
 ```
Similar to `INNER JOIN`, other type of joins can also be used.
> All types of joins combine multiple tables. In specific, `INNER JOIN` will return only the rows that are common to both tables. `OUTER JOIN` will return all the rows from both tables, `LEFT JOIN` will return all rows from first table and will return common rows from second table. `RIGHT JOIN` is the reverse case of LEFT JOIN`. 

Usually joins other than `INNER JOIN` will result in null values. They can be retrieved using `IS/IS NOT NULL` in `WHERE` clause. Also we might need to use self joins too.
Say have a table with name and types of cars and we need to find who all drive mustang **and** ferrari.
```
SELECT t1.name, t2.car
FROM table1 AS t1, table1 AS t2 #Using same table
WHERE t1.name = t2.name AND t1.Car = 'mustang' AND t2.Car = 'ferrari'
```
## Queries with Expressions
Expressions are handy in writing complex logic for querying. The expressions can be combined with all other keywords that we saw before. For example, if we need is to retrieve all students who graduated in even years, the query looks like:
```
SELECT name, year_of_passing
FROM table1
  INNER JOIN table2
    ON table1.id = table2.studentid
WHERE year_of_passing % 2 = 0;
```
Similarly if we can use expressions to transform the data. But for readability we name the transformed column different using `AS` keyword. For example, we need to convert student gpa into 10 point scale, the query looks like:
```
SELECT name, (gpa * 2.5) AS 10_scale_gpa
FROM table1;
```
There are common aggregate functions available such as `COUNT()`, `MIN()`, `MAX()`, `AVG()` and `SUM()`. They can be combined with `GROUP BY` clause too. For example, if we need to find average gpa of students from each state, the query looks like:
```
SELECT State, AVG(gpa) AS Avergae_Gpa
FROM table1
  INNER JOIN table2
    ON table1.id = table2.studentid
GROUP BY State
```
> When using the GROUP BY clause, you should only include columns in the SELECT clause that are either: * Listed in the GROUP BY clause, or * Are included in an aggregate function like SUM(), AVG(), MIN(), MAX(), or COUNT().

What if we need to apply any transformations on the new column generated after `GROUP BY`? SQL provides another keyword, `HAVING` to use after `GROUP BY`. For example, if we need to find the number of students from each state who graduated in 2022. But we are intrested in the states where total package is more than 100000 the query looks like:
```
SELECT state, COUNT(*) as num_students, SUM(package) as total_package
FROM table1
  INNER JOIN table2
    ON table1.id = table2.studentid
WHERE year_of_graduation = 2022
GROUP BY state
HAVING SUM(package) > 100000;
```
> `HAVING` is only used with aggregate functions. Without aggregate functions `WHERE` can do the job.
## Order of Execution
Below numbering shows order of clauses and keywords executed in a complete query.
1. FROM and JOINs 

These are first executed as we need a table in the first place to work. 

2. WHERE 

Once we have the data, `WHERE` constraints are executed and any needless data is discarded.

3. GROUP BY 

The rows that are left are grouped accordingly. There will be as many rows as there are unique values in the column specified by `GROUP BY`.

4. HAVING 

As expected, it is executed immediately after `GROUP BY`.

5. SELECT

Now specified columns or `ALIAS` specified by `AS` are computed.

6. DISTINCT 

Of the remaining rows,the rows with duplicate values in specified column are discarded.

7. ORDER BY 

The rows are sorted either in ascending or descending order.

8. OFFSET and LIMIT 

What to display is controlled by these two at the end.
## Modifying Rows
*Schema* makes a SQL database efficient and consistent even with large amounts of data.
>The database schema describes the structure of each table and the datatypes that each column contain.
### Inserting New Rows
We use `INSERT INTO` statement to specify the table we are modifying and use `VALUES` to specify values. Say, we need to add two new rows to our student table that has columns `Id`,`Name`,`Gpa`, the query looks like;
```
INSERT INTO table1
VALUES
      (20, "Megha",3.94),
      (21, "Macha",3.96);
```
We need to specify all the column values in our new row. If we have incomplete rows to be added in a table that support default values, we can explicitly mention.
```
INSERT INTO table1 (Name,Gpa)
VALUES
      ("Megha", 3.94)
      ("Macha",3.96)
```
We can even specify value in form an equation like `("Megha",3.94*2.5)`.
### Updating Existing Rows
To update alreaday existing rows in the table, we use `UPDATE` and `SET`. In most cases, updating rows will also include `WHERE` to specify row as we dont want to update entire columns. Say we have a table with year of passing and package columns and we want increase package by 1000 and decrease gpa by 10% for all students graduated after 2020.
```
UPDATE table1
SET package = package + 1000, gpa = gpa*0.1
WHERE year > 2020;
```
### Deleting Rows
To delete the rows in a table, we use `DELETE FROM`. Using this clause without a `WHERE` statement will clear all the rows.
```
DELETE FROM table1
WHERE Gpa < 3;
```
> Before updating or deleting, its better to use `SELECT` statement first to check if we are updating/deleting the correct row.
## Table Queries
### Creating Tables
We create tables using `CREATE TABLE` statement. The syntax looks like:
```
CREATE TABLE IF NOT EXISTS tableName (
      columnName1 datatype constraints DEFAULT default_value,
      columnName2 datatype
      );
```
`IF NOT EXISTS` is an optional clause just to check if there is any other table with same name. If column name needs ro be seperated by spaces, enclose it in square brackets or quotation marks. 
Common Data Types include;
|Data Type|Description|
|--|--|
|INT/INTEGER|A signed or unsigned 32bit(depends on database) integer|
|BIGINT|A large integer (1000000)|
|BOOL/BOOLEAN|True/False or 1/0|
|FLOAT|A decimal value|
|DOUBLE|A large decimal value|
|CHAR/CHARACTER(num_char)|String types with fixed length. If input < num_char, extra space is filled with spaces|
|VARCHAR(num_char)|String type with variable length. `num_char` is just a max cap.|
|DATE|Common format includes YYYY-MM-DD|
|TIME|Common format includes HH:MM:SS|
|DATETIME|Mix of date and time, YYYY-MM-DD HH:MM:SS|
|INTERVAL|It stores time difference/interval like 3 hours/5days and 2 hours|
|YEAR/MONTH/DAY|Individual representation|
|BLOB|Binary Large Object store large binary data. They are ususally images, videos that can be queried with right metadata| 

Along with the mandatory datatype, a column can also have an optional constraint that limits what values can be inserted into a column. 
Commonly used constraints are:
|Constraint|Description|
|--|--|
|PRIMARY KEY|Each value in this column is unique and cant have NULL values. They can be used to identify each row (~id)|
|FOREIGN KEY|This establishes relationship between tables. Say there is a master table and a additional table, the ids in additional table are FORIEGN KEY and they should match with PRIMARY KEY in master table|
|AUTOINCREMENT|Integer values are automatically incremented and filled with each row insertion (Not supported by all database|
|UNIQUE|Ensures all values in that column are unique|
|NOT NULL|The column cant have NULL values|
|DEFAULT|A default value if no value is specified|
|CHECK (expression)|Custom expression to have a specific value in column `CHECK (gpa>0 AND gpa<4)`| 

Example:
```
CREATE TABLE table1 (
      id INT PRIMARY KEY AUTOINCREMENT,
      name VARCHAR(50),
      gpa DECIMAL(10,2) CHECK (gpa > 0 AND gpa<4),
      package FLOAT DEFAULT 30000,
);
```
We can even make two columns into a primary key by specificying primary key seperately like `PRIMARY KEY (id, name)`.
Foreign key query looks like:
```
CREATE TABLE table2 (
      id INT,
      name VARCHAR(50),
      year DATE,
      FOREIGN KEY (id) REFERENCES table1(id)
);
```
We cannot use a foreign key **without references**.
### Altering Tables
Altering a table maybe to add new columns, removing existing columns or renaming the table.
```
ALTER TABLE table1
ADD year_of_passing INT;
```
```
ALTER TABLE table1
DROP year_of_passing;
```
```
ALTER TABLE table1
RENAME TO table2;
```
There are additional features provided by specific databases. For example, MySQL provides `FIRST` or `AFTER` keywords for insertion of columns.
### Dropping Tables
```
DROP TABLE IF EXISTS table1;
```
`DELETE` without a `WHERE` will also remove all the rows which is nothing but deleting entire table.
> If there is a `FOREIGN KEY` depending on the table that is to be deleted, first all those dependent tables need to be updated.
## Subqueries
Subqueries, also known as nested queries, are a powerful feature in SQL that allow you to write complex queries by embedding one query inside another. 
> A subquery is a SELECT statement that is embedded within another SELECT, INSERT, UPDATE, or DELETE statement. 
The result of the subquery is then used as a value or a condition in the outer query. Subqueries are *enclosed in parantheses*. Will look at different ways to use subqueries. 
For example, we have two tables `students` and `parttimeEmployees`. We need to find out the students who are also part time employees.
```
SELECT * 
FROM students
WHERE StudentName IN (SELECT EmployeeName FROM parttimeEmployees);
```
Another example would be, say we need to update all gpa by 10% for all those who have gpa less than average gpa.
```
UPDATE students
SET gpa = gpa*1.1
WHERE gpa < (SELECT AVG(gpa) FROM students);
```
### Correlated Subqueries
A correlated subquery is a type of subquery that relies on the outerquery for its values.
> The key difference between a subquery and a correlated subquery is that a subquery is executed independently and its result is used by the outer query, while a correlated subquery is executed for each row of the outer query and its result depends on the values of the outer query.
For example, we need to find average gpa of students from each state where the average gpa is lower than the overall average gpa from all states. The states information is located in a different table. The correlated subquery looks like
```
SELECT State, AVG(gpa) AS avg_gpa
FROM table1
  INNER JOIN table2
    ON table1.id = table2.studentid
GROUP BY State
HAVING AVG(gpa) < (
      SELECT AVG(gpa)
      FROM table1
      WHERE id = table2.studentid
      );
```
In the above example, the subquery is calculating the total average gpa and is using `id` from main query making it correlated subquery.
>Generally, correlated subqueries are realted to main queries using `WHERE` statement.
## SET Queries
There are few set operations available to append one results of different queries. 

The `UNION` operator combines results from two or more `SELECT` statements into a single results set that includes all distinct rows returned by either `SELECT` statements. `UNION ALL` will return all rows without considering whether they are distinct or not. 

The `INTERSECT` operator returns only the common rows returned by `SELECT` two statements. 

The `EXCEPT` operator returns all rows returned by first `SELECT` and that are not present in second `SELECT`.
```
SELECT col1, col2
FROM table1
UNION/UNION ALL/ INTERSECT/ EXCEPT
SELECT col1, col2
FROM table2
```

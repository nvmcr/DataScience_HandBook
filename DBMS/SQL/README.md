# SQL
## Table of Contents
1. [Intro](#intro)
2. [SELECT](#select-queries)
3. [Constraints, Filtering, Sorting](#constraints-filtering-sorting)
4. [Multiple Table Queries](#multi-table-queries)
5. [Queries with Expressions](#queries-with-expressions)
6. [Order of Execution](#order-of-execution)
7. [Modifying Rows](#Modifying-Rows)
## Intro
Structured Query Language (SQL) is a language designed for users to query, manipulate and transform data from a relational database. Biggest advantage of SQL is that it is efficient and scalable for large and complex databases. Popular SQL databases include: SQLite, MySQL, PostgreSQL, Oracle and Microsoft SQL Server. 
* What is a relational database? 
> A realtional database represents a collection of tables. For example, an university realtional database might have a table with `StudentId`, `Name`, `YearofPassing`, `GPA` and many other columns. There can be many other *related* tabes in that database. 
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
Similar to `INNER JOIN`, other type of joins can also be used.
> All types of joins combine multiple tables. In specific, `INNER JOIN` will return only the rows that are common to both tables. `OUTER JOIN` will return all the rows from both tables, `LEFT JOIN` will return all rows from first table and will return common rows from second table. `RIGHT JOIN` is the reverse case of LEFT JOIN`. 
Usually joins other than `INNER JOIN` will result in null values. They can be retrieved using `IS/IS NOT NULL`.
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

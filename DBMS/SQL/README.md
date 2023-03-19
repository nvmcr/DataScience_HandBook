# SQL
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
## Queries with Constraints
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
> `LIKE` is used to filter data based on a specific pattern. It can use wildcard characters like `%` and `_`. `LIKE 'M%'` retrieves every row that **contains** with 'M' and `LIKE 'Me_ha'` retrieves every value **strictly** has 'Me' in front and a *single* middle string and 'ha' at the end. Use multiple underscores to retrieve multiple characters.
> All strings should be represneted within single or double quotations.
  

-- script that creates a table called first_table in the current database in your MySQL server.
-- $ cat 1-first_table.sql | mysql -hlocalhost -uroot -p db_0
-- $ echo "SHOW TABLES;" | mysql -hlocalhost -uroot -p db_0

CREATE TABLE IF NOT EXISTS first_table (
    id INT,
    name VARCHAR(256));

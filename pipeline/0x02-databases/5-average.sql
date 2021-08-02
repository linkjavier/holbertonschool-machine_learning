-- Script that computes the score average of all records
-- in the table second_table in your MySQL server
-- cat 5-average.sql | sudo mysql -hlocalhost -uroot -p db_0

SELECT AVG(score) AS average FROM second_table;

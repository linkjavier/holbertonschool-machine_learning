-- Script that creates a function SafeDiv that divides (and returns) the first
-- by the second number or returns 0 if the second number is equal to 0.

delimiter //
CREATE FUNCTION SafeDiv(a INTEGER, b INTEGER)
	RETURNS FLOAT
	BEGIN
		SET @result = 0;
		IF b <> 0 THEN
			SET @result = a/b;
		END IF;
		RETURN @result;
	END //
delimiter;
-- This script displays the average temperature (Fahrenheit) by city.
-- Results are ordered by average temperature in descending order.

SELECT city, AVG(temperature) AS average_temperature
FROM temperatures
GROUP BY city
ORDER BY average_temperature DESC;

-- This script ranks country origins of bands by the total number of non-unique fans.
-- It works on any database and handles any number of entries.

SELECT origin, SUM(nb_fans) AS nb_fans
FROM metal_bands
GROUP BY origin
ORDER BY nb_fans DESC;

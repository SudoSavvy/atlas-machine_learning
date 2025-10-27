-- This script lists all bands with Glam rock as their main style.
-- Each record displays: band_name - lifespan until 2020 (in years).
-- Results are ranked by longevity in descending order.

SELECT band_name, (IFNULL(split, 2020) - formed) AS lifespan
FROM metal_bands
WHERE style = 'Glam rock'
ORDER BY lifespan DESC;

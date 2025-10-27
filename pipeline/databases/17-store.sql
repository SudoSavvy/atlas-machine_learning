-- This script creates a trigger that decreases the quantity of an item
-- after a new order is inserted into the orders table.
-- Quantity in the items table can be negative.

DELIMITER $$

CREATE TRIGGER decrease_item_quantity
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE id = NEW.item_id;
END$$

DELIMITER ;

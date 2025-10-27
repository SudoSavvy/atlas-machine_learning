-- This script creates a trigger that decreases the quantity of an item
-- after a new order is added to the orders table.
-- Quantity in the items table can be negative.

DELIMITER $$

CREATE TRIGGER decrease_quantity_after_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.quantity
    WHERE id = NEW.item_id;
END$$

DELIMITER ;

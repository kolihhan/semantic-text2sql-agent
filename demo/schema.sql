PRAGMA foreign_keys = ON;
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country_code TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    total_amount REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(id),
    FOREIGN KEY(product_id) REFERENCES products(id)
);
INSERT INTO customers VALUES
(1, 'Alice', 'CZE'),
(2, 'Bob', 'MYS'),
(3, 'Carla', 'USA');
INSERT INTO products VALUES
(1, 'Widget', 10.0),
(2, 'Gadget', 25.0),
(3, 'Course Pack', 40.0);
INSERT INTO orders VALUES
(1, 1, 1, 120.0, '2024-01-12'),
(2, 1, 2, 250.0, '2024-05-03'),
(3, 2, 1, 90.0, '2024-06-15'),
(4, 3, 3, 400.0, '2023-11-20'),
(5, 1, 3, 500.0, '2023-07-09');

CREATE TABLE ventas (
    fecha DATE,
    producto VARCHAR(50),
    cantidad INT,
    precio DECIMAL(10,2)
);

INSERT INTO ventas (fecha, producto, cantidad, precio) VALUES
('2025-09-01', 'Laptop', 5, 1200),
('2025-09-01', 'Mouse', 10, 25),
('2025-09-02', 'Teclado', 7, 45),
('2025-09-02', 'Monitor', 3, 300),
('2025-09-03', 'Impresora', 2, 200);

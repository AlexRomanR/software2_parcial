CREATE TABLE ventas_diarias (
    fecha DATE,
    vendedor VARCHAR(50),
    producto VARCHAR(100),
    cantidad INT,
    precio_unitario DECIMAL(10,2),
    total DECIMAL(10,2)
);

INSERT INTO ventas_diarias (fecha, vendedor, producto, cantidad, precio_unitario, total) VALUES
('2024-09-01', 'Juan Pérez', 'Laptop HP', 2, 800.00, 1600.00),
('2024-09-01', 'María García', 'Mouse Logitech', 5, 25.00, 125.00),
('2024-09-01', 'Carlos López', 'Monitor Samsung', 1, 300.00, 300.00),
('2024-09-02', 'Juan Pérez', 'Teclado Mecánico', 3, 80.00, 240.00),
('2024-09-02', 'María García', 'Auriculares Sony', 2, 150.00, 300.00),
('2024-09-03', 'Carlos López', 'Smartphone iPhone', 1, 900.00, 900.00),
('2024-09-03', 'Juan Pérez', 'Tablet iPad', 1, 400.00, 400.00),
('2024-09-04', 'María García', 'Laptop HP', 3, 800.00, 2400.00),
('2024-09-04', 'Carlos López', 'Mouse Logitech', 8, 25.00, 200.00),
('2024-09-05', 'Juan Pérez', 'Monitor Samsung', 2, 300.00, 600.00),
('2024-09-08', 'María García', 'Mouse Logitech', 1, 25.00, 25.00),
('2024-09-09', 'Carlos López', 'Teclado Mecánico', 1, 80.00, 80.00),
('2024-09-10', 'Juan Pérez', 'Auriculares Sony', 1, 150.00, 150.00),
('2024-09-11', 'María García', 'Mouse Logitech', 2, 25.00, 50.00),
('2024-09-12', 'Carlos López', 'Teclado Mecánico', 1, 80.00, 80.00),
('2024-09-15', 'Juan Pérez', 'Laptop HP', 5, 800.00, 4000.00),
('2024-09-15', 'María García', 'Smartphone iPhone', 3, 900.00, 2700.00),
('2024-09-15', 'Carlos López', 'Monitor Samsung', 4, 300.00, 1200.00),
('2024-09-16', 'Juan Pérez', 'Tablet iPad', 6, 400.00, 2400.00),
('2024-09-16', 'María García', 'Laptop HP', 2, 800.00, 1600.00),
('2024-09-17', 'Carlos López', 'Smartphone iPhone', 2, 900.00, 1800.00),
('2024-09-17', 'Juan Pérez', 'Monitor Samsung', 3, 300.00, 900.00),
('2024-09-18', 'María García', 'Auriculares Sony', 4, 150.00, 600.00),
('2024-09-19', 'Carlos López', 'Laptop HP', 1, 800.00, 800.00),
('2024-09-22', 'Juan Pérez', 'Mouse Logitech', 10, 25.00, 250.00),
('2024-09-22', 'María García', 'Teclado Mecánico', 5, 80.00, 400.00),
('2024-09-22', 'Carlos López', 'Tablet iPad', 2, 400.00, 800.00);

CREATE TABLE inventario (
    producto VARCHAR(100),
    stock_actual INT,
    stock_minimo INT,
    estado VARCHAR(20)
);

INSERT INTO inventario (producto, stock_actual, stock_minimo, estado) VALUES
('Laptop HP', 5, 10, 'Critico'),
('Mouse Logitech', 50, 20, 'Normal'),
('Monitor Samsung', 8, 15, 'Critico'),
('Teclado Mecánico', 25, 10, 'Normal'),
('Auriculares Sony', 3, 5, 'Critico'),
('Smartphone iPhone', 12, 8, 'Normal'),
('Tablet iPad', 18, 12, 'Normal');

CREATE TABLE ventas_vendedor (
    vendedor VARCHAR(50),
    mes VARCHAR(7),
    total_ventas DECIMAL(10,2),
    num_ventas INT
);

INSERT INTO ventas_vendedor (vendedor, mes, total_ventas, num_ventas) VALUES
('Juan Pérez', '2024-09', 12190.00, 12),
('María García', '2024-09', 8800.00, 10),
('Carlos López', '2024-09', 6280.00, 9);

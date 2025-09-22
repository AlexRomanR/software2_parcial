-- Datos de ejemplo para sistema BI
-- Ventas de una empresa por trimestres

CREATE TABLE ventas (
    id INT PRIMARY KEY,
    fecha DATE,
    producto VARCHAR(100),
    categoria VARCHAR(50),
    vendedor VARCHAR(100),
    cantidad INT,
    precio_unitario DECIMAL(10,2),
    total DECIMAL(10,2),
    region VARCHAR(50),
    mes VARCHAR(20)
);

INSERT INTO ventas (id, fecha, producto, categoria, vendedor, cantidad, precio_unitario, total, region, mes) VALUES
(1, '2024-01-15', 'Laptop Dell', 'Tecnología', 'Juan Pérez', 2, 800.00, 1600.00, 'Norte', 'Enero'),
(2, '2024-01-20', 'iPhone 15', 'Tecnología', 'María García', 1, 1200.00, 1200.00, 'Sur', 'Enero'),
(3, '2024-01-25', 'Escritorio', 'Muebles', 'Carlos López', 3, 300.00, 900.00, 'Centro', 'Enero'),
(4, '2024-02-10', 'Monitor LG', 'Tecnología', 'Ana Torres', 2, 250.00, 500.00, 'Norte', 'Febrero'),
(5, '2024-02-15', 'Silla Gamer', 'Muebles', 'Luis Martín', 1, 450.00, 450.00, 'Sur', 'Febrero'),
(6, '2024-02-20', 'MacBook Pro', 'Tecnología', 'Juan Pérez', 1, 2500.00, 2500.00, 'Centro', 'Febrero'),
(7, '2024-03-05', 'Mesa Reuniones', 'Muebles', 'María García', 1, 800.00, 800.00, 'Norte', 'Marzo'),
(8, '2024-03-12', 'Tablet Samsung', 'Tecnología', 'Carlos López', 3, 350.00, 1050.00, 'Sur', 'Marzo'),
(9, '2024-03-18', 'Proyector', 'Tecnología', 'Ana Torres', 1, 600.00, 600.00, 'Centro', 'Marzo'),
(10, '2024-03-25', 'Auriculares', 'Tecnología', 'Luis Martín', 5, 80.00, 400.00, 'Norte', 'Marzo'),
(11, '2024-04-02', 'Laptop HP', 'Tecnología', 'Juan Pérez', 2, 750.00, 1500.00, 'Sur', 'Abril'),
(12, '2024-04-08', 'Armario', 'Muebles', 'María García', 1, 650.00, 650.00, 'Centro', 'Abril'),
(13, '2024-04-15', 'Teclado Mecánico', 'Tecnología', 'Carlos López', 4, 120.00, 480.00, 'Norte', 'Abril'),
(14, '2024-04-22', 'Sofá Oficina', 'Muebles', 'Ana Torres', 1, 1200.00, 1200.00, 'Sur', 'Abril'),
(15, '2024-05-03', 'Mouse Gaming', 'Tecnología', 'Luis Martín', 6, 60.00, 360.00, 'Centro', 'Mayo'),
(16, '2024-05-10', 'Estantería', 'Muebles', 'Juan Pérez', 2, 400.00, 800.00, 'Norte', 'Mayo'),
(17, '2024-05-17', 'Webcam 4K', 'Tecnología', 'María García', 3, 150.00, 450.00, 'Sur', 'Mayo'),
(18, '2024-05-24', 'Impresora', 'Tecnología', 'Carlos López', 1, 300.00, 300.00, 'Centro', 'Mayo'),
(19, '2024-06-05', 'Lámpara LED', 'Oficina', 'Ana Torres', 8, 45.00, 360.00, 'Norte', 'Junio'),
(20, '2024-06-12', 'Cafetera', 'Oficina', 'Luis Martín', 2, 180.00, 360.00, 'Sur', 'Junio'),
(21, '2024-06-18', 'Ventilador', 'Oficina', 'Juan Pérez', 4, 90.00, 360.00, 'Centro', 'Junio'),
(22, '2024-06-25', 'Pizarra', 'Oficina', 'María García', 3, 120.00, 360.00, 'Norte', 'Junio'),
(23, '2024-07-08', 'Cámara Seguridad', 'Tecnología', 'Carlos López', 5, 200.00, 1000.00, 'Sur', 'Julio'),
(24, '2024-07-15', 'Archivador', 'Muebles', 'Ana Torres', 2, 250.00, 500.00, 'Centro', 'Julio'),
(25, '2024-07-22', 'Router WiFi', 'Tecnología', 'Luis Martín', 3, 100.00, 300.00, 'Norte', 'Julio');

-- Tabla adicional: Empleados
CREATE TABLE empleados (
    id INT PRIMARY KEY,
    nombre VARCHAR(100),
    departamento VARCHAR(50),
    salario DECIMAL(10,2),
    fecha_ingreso DATE,
    edad INT,
    ciudad VARCHAR(50)
);

INSERT INTO empleados (id, nombre, departamento, salario, fecha_ingreso, edad, ciudad) VALUES
(1, 'Juan Pérez', 'Ventas', 3500.00, '2022-01-15', 28, 'La Paz'),
(2, 'María García', 'Ventas', 3200.00, '2022-03-20', 26, 'Santa Cruz'),
(3, 'Carlos López', 'Marketing', 4000.00, '2021-06-10', 32, 'Cochabamba'),
(4, 'Ana Torres', 'Ventas', 3800.00, '2023-02-05', 29, 'La Paz'),
(5, 'Luis Martín', 'TI', 5000.00, '2020-09-12', 35, 'Santa Cruz'),
(6, 'Sofia Morales', 'RRHH', 4200.00, '2021-11-08', 31, 'La Paz'),
(7, 'Diego Vargas', 'TI', 4800.00, '2022-07-22', 27, 'Cochabamba'),
(8, 'Elena Rojas', 'Marketing', 3600.00, '2023-01-18', 25, 'Santa Cruz');
-- Tabla de Departamentos
CREATE TABLE IF NOT EXISTS departamentos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(50),
    ubicacion VARCHAR(50)
);

INSERT INTO departamentos (nombre, ubicacion) VALUES
('Recursos Humanos', 'Santa Cruz'),
('Desarrollo', 'La Paz'),
('Ventas', 'Cochabamba');

-- Tabla de Empleados
CREATE TABLE IF NOT EXISTS empleados (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(50),
    puesto VARCHAR(50),
    salario DECIMAL(10,2),
    departamento_id INT REFERENCES departamentos(id)
);

INSERT INTO empleados (nombre, puesto, salario, departamento_id) VALUES
('Ana López', 'Desarrolladora', 5000, 2),
('Carlos Pérez', 'Vendedor', 3000, 3),
('Lucía Torres', 'Recursos Humanos', 3500, 1),
('Mario Gómez', 'Desarrollador', 5200, 2);

-- Tabla de Proyectos
CREATE TABLE IF NOT EXISTS proyectos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(50),
    presupuesto DECIMAL(10,2),
    departamento_id INT REFERENCES departamentos(id)
);

INSERT INTO proyectos (nombre, presupuesto, departamento_id) VALUES
('Sistema Web', 10000, 2),
('Campaña Ventas', 7000, 3),
('Capacitación Interna', 5000, 1);

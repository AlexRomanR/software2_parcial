-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Servidor: 127.0.0.1
-- Tiempo de generación: 14-09-2025 a las 20:48:59
-- Versión del servidor: 10.4.32-MariaDB
-- Versión de PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de datos: `iglesia`
--

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `actividades_tarea`
--

CREATE TABLE `actividades_tarea` (
  `id` int(11) NOT NULL,
  `curso_id` int(11) NOT NULL,
  `titulo` text NOT NULL,
  `descripcion` text DEFAULT NULL,
  `archivo` varchar(255) DEFAULT NULL,
  `fecha_entrega` date DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `actividades_tarea`
--

INSERT INTO `actividades_tarea` (`id`, `curso_id`, `titulo`, `descripcion`, `archivo`, `fecha_entrega`) VALUES
(1, 2, 'ddd', 'dddd', '433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-22'),
(2, 2, 'aa', 'assssdasdsad', 'Conectar el DHCP de un router a un switch a maquinas.docx', '2025-03-28');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `administradores`
--

CREATE TABLE `administradores` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `administradores`
--

INSERT INTO `administradores` (`id`, `user_id`) VALUES
(2, 3),
(3, 4),
(5, 7),
(7, 14),
(11, 18),
(10, 16);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `categoria`
--

CREATE TABLE `categoria` (
  `id` int(11) NOT NULL,
  `nombre` text NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `categoria`
--

INSERT INTO `categoria` (`id`, `nombre`) VALUES
(1, 'Matrimonio'),
(2, 'Bautizo'),
(3, 'Comunion'),
(4, 'Confirmacion');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `cursos`
--

CREATE TABLE `cursos` (
  `id` int(11) NOT NULL,
  `nombre` text NOT NULL,
  `descripcion` text DEFAULT NULL,
  `fecha_inicio` date NOT NULL,
  `fecha_fin` date NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `cursos`
--

INSERT INTO `cursos` (`id`, `nombre`, `descripcion`, `fecha_inicio`, `fecha_fin`) VALUES
(2, 'AAAAD', 'aaaa', '2025-03-07', '2025-03-29');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `eventos`
--

CREATE TABLE `eventos` (
  `id` int(11) NOT NULL,
  `titulo` varchar(100) NOT NULL,
  `descripcion` text NOT NULL,
  `fecha` date NOT NULL,
  `ministerio_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `eventos`
--

INSERT INTO `eventos` (`id`, `titulo`, `descripcion`, `fecha`, `ministerio_id`) VALUES
(2, 'xddd', 'aaa', '2025-03-28', 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `membresias`
--

CREATE TABLE `membresias` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `fecha_inicio` date NOT NULL,
  `fecha_fin` date DEFAULT NULL,
  `descripcion` text DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `membresias`
--

INSERT INTO `membresias` (`id`, `user_id`, `fecha_inicio`, `fecha_fin`, `descripcion`) VALUES
(2, 1, '2025-03-24', '2025-06-20', 'aasdasd');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `miembros`
--

CREATE TABLE `miembros` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `miembros`
--

INSERT INTO `miembros` (`id`, `user_id`) VALUES
(2, 8),
(6, 13);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `miembro_curso`
--

CREATE TABLE `miembro_curso` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `curso_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `miembro_curso`
--

INSERT INTO `miembro_curso` (`id`, `user_id`, `curso_id`) VALUES
(6, 3, 2),
(9, 4, 2),
(10, 1, 2);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `miembro_curso_actividad`
--

CREATE TABLE `miembro_curso_actividad` (
  `id` int(11) NOT NULL,
  `miembro_curso_id` int(11) NOT NULL,
  `actividad_tarea_id` int(11) NOT NULL,
  `nota` int(11) DEFAULT NULL,
  `archivo` varchar(255) DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `miembro_curso_actividad`
--

INSERT INTO `miembro_curso_actividad` (`id`, `miembro_curso_id`, `actividad_tarea_id`, `nota`, `archivo`) VALUES
(1, 6, 2, 70, '1743234376_1486472-rammus-5-article_image_bd-1.jpg');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `ministerio`
--

CREATE TABLE `ministerio` (
  `id` int(11) NOT NULL,
  `nombre` varchar(100) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `ministerio`
--

INSERT INTO `ministerio` (`id`, `nombre`, `user_id`) VALUES
(1, 'ministerio del amor', 3);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `ministerio_user`
--

CREATE TABLE `ministerio_user` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `ministerio_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `ministerio_user`
--

INSERT INTO `ministerio_user` (`id`, `user_id`, `ministerio_id`) VALUES
(3, 1, 1),
(5, 4, 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `model_has_role`
--

CREATE TABLE `model_has_role` (
  `user_id` int(11) NOT NULL,
  `role_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `model_has_role`
--

INSERT INTO `model_has_role` (`user_id`, `role_id`) VALUES
(7, 1),
(8, 4);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `permissions`
--

CREATE TABLE `permissions` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `permissions`
--

INSERT INTO `permissions` (`id`, `name`) VALUES
(1, 'crear'),
(2, 'editar'),
(3, 'eliminar'),
(4, 'subir'),
(5, 'calificar');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `roles`
--

CREATE TABLE `roles` (
  `id` int(11) NOT NULL,
  `name` varchar(50) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `roles`
--

INSERT INTO `roles` (`id`, `name`) VALUES
(1, 'Administrador'),
(4, 'Miembro');

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `role_has_permission`
--

CREATE TABLE `role_has_permission` (
  `role_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `role_has_permission`
--

INSERT INTO `role_has_permission` (`role_id`, `permission_id`) VALUES
(1, 1),
(1, 2),
(1, 3),
(4, 4);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `sacramentos`
--

CREATE TABLE `sacramentos` (
  `id` int(11) NOT NULL,
  `fecha` date NOT NULL,
  `lugar` text NOT NULL,
  `ministro` text NOT NULL,
  `categoria_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `sacramentos`
--

INSERT INTO `sacramentos` (`id`, `fecha`, `lugar`, `ministro`, `categoria_id`) VALUES
(1, '2025-03-07', 'asdsad', 'asdasd', 2),
(2, '2025-03-25', 'sss', 'ss', 1);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `sacramento_user`
--

CREATE TABLE `sacramento_user` (
  `id` int(11) NOT NULL,
  `sacramento_id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `sacramento_user`
--

INSERT INTO `sacramento_user` (`id`, `sacramento_id`, `user_id`) VALUES
(1, 1, 1),
(7, 2, 4),
(6, 2, 3);

-- --------------------------------------------------------

--
-- Estructura de tabla para la tabla `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `nombre` varchar(100) NOT NULL,
  `password` varchar(255) NOT NULL,
  `email` varchar(100) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `direccion` varchar(255) DEFAULT NULL,
  `telefono` bigint(20) DEFAULT NULL,
  `ci` int(11) DEFAULT NULL,
  `foto` varchar(255) DEFAULT NULL,
  `fecha_nacimiento` date DEFAULT NULL,
  `genero` varchar(20) DEFAULT NULL
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Volcado de datos para la tabla `users`
--

INSERT INTO `users` (`id`, `nombre`, `password`, `email`, `created_at`, `direccion`, `telefono`, `ci`, `foto`, `fecha_nacimiento`, `genero`) VALUES
(1, 'usuario1', '$2y$10$IOraQxcAm51IkWWSq6ncGODuIkmIPdhftzRabRUPgy9sUqi7r8MXi', 'user@gmail.com', '2025-03-22 03:33:48', NULL, NULL, NULL, NULL, NULL, NULL),
(3, 'admin', '12345678', 'admin@gmail.com', '2025-03-23 02:03:13', 'Santa Cruz', 7844521, 457456856, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-14', 'Masculino'),
(4, 'asdxdd123', '12345678', 'asdasd@gmail.com', '2025-03-23 02:44:00', 'Santa Cruz', 78403, 4523, '', '2025-03-14', 'Femenino'),
(8, 'user1', '12345678', 'user1@gmail.com', '2025-03-29 19:29:32', 'Santa Cruz', 78408033, 12311, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-13', 'Femenino'),
(7, 'adm@gmail.com', '12345678', 'adm@gmail.com', '2025-03-29 19:25:28', 'Santa Cruz', 78408033, 4213, '/uploads/1486472-rammus-5-article_image_bd-1.jpg', '2025-03-28', 'Masculino'),
(13, 'Usersdfsdfsdf', '12345678', 'fgdf@das.dcom', '2025-06-13 01:33:52', 'asdsad', 784, 1234, '', '2025-06-26', 'Masculino'),
(14, 'Addmminn', '12345678', 'asdasd@mgia.com', '2025-06-13 01:41:20', 'gfhfgh', 22222, 4563, '', '2025-06-12', 'Masculino'),
(16, 'usssrrr', '12345678', 'dsfsdf@sdf.co', '2025-06-13 04:19:30', 'ffff', 545, 32, '', '2025-06-12', 'Masculino'),
(18, 'addddmdd', '12345678', 'sdfdsf@dsfds.co', '2025-06-13 04:33:21', 'defdf', 87876, 56876, '', '2025-06-14', 'Masculino');

--
-- Índices para tablas volcadas
--

--
-- Indices de la tabla `actividades_tarea`
--
ALTER TABLE `actividades_tarea`
  ADD PRIMARY KEY (`id`),
  ADD KEY `curso_id` (`curso_id`);

--
-- Indices de la tabla `administradores`
--
ALTER TABLE `administradores`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indices de la tabla `categoria`
--
ALTER TABLE `categoria`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `cursos`
--
ALTER TABLE `cursos`
  ADD PRIMARY KEY (`id`);

--
-- Indices de la tabla `eventos`
--
ALTER TABLE `eventos`
  ADD PRIMARY KEY (`id`),
  ADD KEY `ministerio_id` (`ministerio_id`);

--
-- Indices de la tabla `membresias`
--
ALTER TABLE `membresias`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indices de la tabla `miembros`
--
ALTER TABLE `miembros`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indices de la tabla `miembro_curso`
--
ALTER TABLE `miembro_curso`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `curso_id` (`curso_id`);

--
-- Indices de la tabla `miembro_curso_actividad`
--
ALTER TABLE `miembro_curso_actividad`
  ADD PRIMARY KEY (`id`),
  ADD KEY `miembro_curso_id` (`miembro_curso_id`),
  ADD KEY `actividad_tarea_id` (`actividad_tarea_id`);

--
-- Indices de la tabla `ministerio`
--
ALTER TABLE `ministerio`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indices de la tabla `ministerio_user`
--
ALTER TABLE `ministerio_user`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`),
  ADD KEY `ministerio_id` (`ministerio_id`);

--
-- Indices de la tabla `model_has_role`
--
ALTER TABLE `model_has_role`
  ADD PRIMARY KEY (`user_id`,`role_id`),
  ADD KEY `role_id` (`role_id`);

--
-- Indices de la tabla `permissions`
--
ALTER TABLE `permissions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indices de la tabla `roles`
--
ALTER TABLE `roles`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indices de la tabla `role_has_permission`
--
ALTER TABLE `role_has_permission`
  ADD PRIMARY KEY (`role_id`,`permission_id`),
  ADD KEY `permission_id` (`permission_id`);

--
-- Indices de la tabla `sacramentos`
--
ALTER TABLE `sacramentos`
  ADD PRIMARY KEY (`id`),
  ADD KEY `categoria_id` (`categoria_id`);

--
-- Indices de la tabla `sacramento_user`
--
ALTER TABLE `sacramento_user`
  ADD PRIMARY KEY (`id`),
  ADD KEY `sacramento_id` (`sacramento_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indices de la tabla `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`nombre`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT de las tablas volcadas
--

--
-- AUTO_INCREMENT de la tabla `actividades_tarea`
--
ALTER TABLE `actividades_tarea`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `administradores`
--
ALTER TABLE `administradores`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT de la tabla `categoria`
--
ALTER TABLE `categoria`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT de la tabla `cursos`
--
ALTER TABLE `cursos`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `eventos`
--
ALTER TABLE `eventos`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `membresias`
--
ALTER TABLE `membresias`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `miembros`
--
ALTER TABLE `miembros`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=9;

--
-- AUTO_INCREMENT de la tabla `miembro_curso`
--
ALTER TABLE `miembro_curso`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT de la tabla `miembro_curso_actividad`
--
ALTER TABLE `miembro_curso_actividad`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT de la tabla `ministerio`
--
ALTER TABLE `ministerio`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `ministerio_user`
--
ALTER TABLE `ministerio_user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de la tabla `permissions`
--
ALTER TABLE `permissions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de la tabla `roles`
--
ALTER TABLE `roles`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT de la tabla `sacramentos`
--
ALTER TABLE `sacramentos`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT de la tabla `sacramento_user`
--
ALTER TABLE `sacramento_user`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;

--
-- AUTO_INCREMENT de la tabla `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=19;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;

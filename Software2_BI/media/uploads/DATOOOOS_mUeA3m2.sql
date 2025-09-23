--
-- PostgreSQL database dump
--

\restrict cBR7ipK0mmvSgk7DHwhaFedZ0ZLYjNpVZM59aNbHOcImyDROhqcMePcvQj457Bt

-- Dumped from database version 16.9
-- Dumped by pg_dump version 17.6

-- Started on 2025-09-14 17:43:01

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 9 (class 2615 OID 113609)
-- Name: public; Type: SCHEMA; Schema: -; Owner: grupo07sc
--

-- *not* creating schema, since initdb creates it


ALTER SCHEMA public OWNER TO grupo07sc;

--
-- TOC entry 3630 (class 0 OID 0)
-- Dependencies: 9
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: grupo07sc
--

COMMENT ON SCHEMA public IS '';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 234 (class 1259 OID 135620)
-- Name: cronograma; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.cronograma (
    codigo integer NOT NULL,
    dia character varying(191),
    hora_inicio time without time zone,
    hora_fin time without time zone,
    curso_id bigint,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.cronograma OWNER TO grupo07sc;

--
-- TOC entry 233 (class 1259 OID 135619)
-- Name: cronograma_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.cronograma_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.cronograma_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3632 (class 0 OID 0)
-- Dependencies: 233
-- Name: cronograma_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.cronograma_codigo_seq OWNED BY public.cronograma.codigo;


--
-- TOC entry 232 (class 1259 OID 135601)
-- Name: curso; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.curso (
    codigo integer NOT NULL,
    nombre character varying(191),
    duracion character varying(191),
    cupo integer,
    foto character varying(191),
    presencial boolean,
    profesor_id bigint,
    users_id bigint,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.curso OWNER TO grupo07sc;

--
-- TOC entry 231 (class 1259 OID 135600)
-- Name: curso_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.curso_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.curso_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3633 (class 0 OID 0)
-- Dependencies: 231
-- Name: curso_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.curso_codigo_seq OWNED BY public.curso.codigo;


--
-- TOC entry 224 (class 1259 OID 135564)
-- Name: estado; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.estado (
    codigo integer NOT NULL,
    nombre character varying(100),
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.estado OWNER TO grupo07sc;

--
-- TOC entry 223 (class 1259 OID 135563)
-- Name: estado_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.estado_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.estado_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3634 (class 0 OID 0)
-- Dependencies: 223
-- Name: estado_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.estado_codigo_seq OWNED BY public.estado.codigo;


--
-- TOC entry 236 (class 1259 OID 135632)
-- Name: estudiante; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.estudiante (
    codigo integer NOT NULL,
    nombre character varying(191),
    apellido character varying(191),
    ci integer,
    tipo_estudiante_id bigint,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.estudiante OWNER TO grupo07sc;

--
-- TOC entry 235 (class 1259 OID 135631)
-- Name: estudiante_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.estudiante_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.estudiante_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3635 (class 0 OID 0)
-- Dependencies: 235
-- Name: estudiante_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.estudiante_codigo_seq OWNED BY public.estudiante.codigo;


--
-- TOC entry 238 (class 1259 OID 135652)
-- Name: estudiante_curso; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.estudiante_curso (
    codigo integer NOT NULL,
    monto double precision,
    estudiante_id bigint,
    curso_id bigint,
    estado_id bigint,
    factura_id bigint,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.estudiante_curso OWNER TO grupo07sc;

--
-- TOC entry 237 (class 1259 OID 135651)
-- Name: estudiante_curso_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.estudiante_curso_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.estudiante_curso_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3636 (class 0 OID 0)
-- Dependencies: 237
-- Name: estudiante_curso_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.estudiante_curso_codigo_seq OWNED BY public.estudiante_curso.codigo;


--
-- TOC entry 226 (class 1259 OID 135571)
-- Name: factura; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.factura (
    codigo integer NOT NULL,
    users_id bigint,
    monto double precision,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.factura OWNER TO grupo07sc;

--
-- TOC entry 225 (class 1259 OID 135570)
-- Name: factura_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.factura_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.factura_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3637 (class 0 OID 0)
-- Dependencies: 225
-- Name: factura_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.factura_codigo_seq OWNED BY public.factura.codigo;


--
-- TOC entry 244 (class 1259 OID 150367)
-- Name: funcionalidad; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.funcionalidad (
    codigo integer NOT NULL,
    nombre character varying(100) NOT NULL,
    descripcion text,
    icono character varying(100),
    ruta character varying(255)
);


ALTER TABLE public.funcionalidad OWNER TO grupo07sc;

--
-- TOC entry 243 (class 1259 OID 150366)
-- Name: funcionalidad_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.funcionalidad_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.funcionalidad_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3638 (class 0 OID 0)
-- Dependencies: 243
-- Name: funcionalidad_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.funcionalidad_codigo_seq OWNED BY public.funcionalidad.codigo;


--
-- TOC entry 246 (class 1259 OID 150376)
-- Name: permiso; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.permiso (
    codigo integer NOT NULL,
    activo boolean NOT NULL,
    rol_id integer NOT NULL,
    funcionalidad_id integer NOT NULL
);


ALTER TABLE public.permiso OWNER TO grupo07sc;

--
-- TOC entry 245 (class 1259 OID 150375)
-- Name: permiso_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.permiso_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.permiso_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3639 (class 0 OID 0)
-- Dependencies: 245
-- Name: permiso_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.permiso_codigo_seq OWNED BY public.permiso.codigo;


--
-- TOC entry 247 (class 1259 OID 150393)
-- Name: permiso_recurso; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.permiso_recurso (
    activo boolean DEFAULT true NOT NULL,
    permiso_id integer NOT NULL,
    recurso_id integer NOT NULL
);


ALTER TABLE public.permiso_recurso OWNER TO grupo07sc;

--
-- TOC entry 251 (class 1259 OID 150424)
-- Name: precio; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.precio (
    codigo integer NOT NULL,
    precio double precision NOT NULL,
    curso_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    tipo_estudiante_id integer
);


ALTER TABLE public.precio OWNER TO grupo07sc;

--
-- TOC entry 250 (class 1259 OID 150423)
-- Name: precio_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.precio_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.precio_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3640 (class 0 OID 0)
-- Dependencies: 250
-- Name: precio_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.precio_codigo_seq OWNED BY public.precio.codigo;


--
-- TOC entry 230 (class 1259 OID 135592)
-- Name: profesor; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.profesor (
    codigo integer NOT NULL,
    nombre character varying(191),
    apellido character varying(191),
    titulo character varying(191),
    foto character varying(2048),
    ci integer,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.profesor OWNER TO grupo07sc;

--
-- TOC entry 229 (class 1259 OID 135591)
-- Name: profesor_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.profesor_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.profesor_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3641 (class 0 OID 0)
-- Dependencies: 229
-- Name: profesor_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.profesor_codigo_seq OWNED BY public.profesor.codigo;


--
-- TOC entry 240 (class 1259 OID 150349)
-- Name: recurso; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.recurso (
    codigo integer NOT NULL,
    nombre character varying(100) NOT NULL,
    descripcion text
);


ALTER TABLE public.recurso OWNER TO grupo07sc;

--
-- TOC entry 239 (class 1259 OID 150348)
-- Name: recurso_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.recurso_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.recurso_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3642 (class 0 OID 0)
-- Dependencies: 239
-- Name: recurso_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.recurso_codigo_seq OWNED BY public.recurso.codigo;


--
-- TOC entry 242 (class 1259 OID 150358)
-- Name: rol; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.rol (
    codigo integer NOT NULL,
    nombre character varying(100) NOT NULL,
    descripcion text
);


ALTER TABLE public.rol OWNER TO grupo07sc;

--
-- TOC entry 241 (class 1259 OID 150357)
-- Name: rol_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.rol_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.rol_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3643 (class 0 OID 0)
-- Dependencies: 241
-- Name: rol_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.rol_codigo_seq OWNED BY public.rol.codigo;


--
-- TOC entry 228 (class 1259 OID 135583)
-- Name: tipo_estudiante; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.tipo_estudiante (
    codigo integer NOT NULL,
    nombre text,
    status boolean,
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone
);


ALTER TABLE public.tipo_estudiante OWNER TO grupo07sc;

--
-- TOC entry 227 (class 1259 OID 135582)
-- Name: tipo_estudiante_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.tipo_estudiante_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.tipo_estudiante_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3644 (class 0 OID 0)
-- Dependencies: 227
-- Name: tipo_estudiante_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.tipo_estudiante_codigo_seq OWNED BY public.tipo_estudiante.codigo;


--
-- TOC entry 222 (class 1259 OID 135553)
-- Name: users; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.users (
    codigo integer NOT NULL,
    nombre character varying(191) NOT NULL,
    ci character varying(191) NOT NULL,
    email character varying(191) NOT NULL,
    email_verified_at timestamp(0) without time zone,
    password character varying(191) NOT NULL,
    remember_token character varying(100),
    created_at timestamp(0) without time zone,
    updated_at timestamp(0) without time zone,
    rol_id integer DEFAULT 2 NOT NULL
);


ALTER TABLE public.users OWNER TO grupo07sc;

--
-- TOC entry 221 (class 1259 OID 135552)
-- Name: users_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.users_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3645 (class 0 OID 0)
-- Dependencies: 221
-- Name: users_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.users_codigo_seq OWNED BY public.users.codigo;


--
-- TOC entry 249 (class 1259 OID 150411)
-- Name: visita; Type: TABLE; Schema: public; Owner: grupo07sc
--

CREATE TABLE public.visita (
    codigo integer NOT NULL,
    ruta character varying(255) NOT NULL,
    veces integer DEFAULT 1 NOT NULL,
    users_id integer
);


ALTER TABLE public.visita OWNER TO grupo07sc;

--
-- TOC entry 248 (class 1259 OID 150410)
-- Name: visita_codigo_seq; Type: SEQUENCE; Schema: public; Owner: grupo07sc
--

CREATE SEQUENCE public.visita_codigo_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.visita_codigo_seq OWNER TO grupo07sc;

--
-- TOC entry 3646 (class 0 OID 0)
-- Dependencies: 248
-- Name: visita_codigo_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: grupo07sc
--

ALTER SEQUENCE public.visita_codigo_seq OWNED BY public.visita.codigo;


--
-- TOC entry 3385 (class 2604 OID 135623)
-- Name: cronograma codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.cronograma ALTER COLUMN codigo SET DEFAULT nextval('public.cronograma_codigo_seq'::regclass);


--
-- TOC entry 3384 (class 2604 OID 135604)
-- Name: curso codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.curso ALTER COLUMN codigo SET DEFAULT nextval('public.curso_codigo_seq'::regclass);


--
-- TOC entry 3380 (class 2604 OID 135567)
-- Name: estado codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estado ALTER COLUMN codigo SET DEFAULT nextval('public.estado_codigo_seq'::regclass);


--
-- TOC entry 3386 (class 2604 OID 135635)
-- Name: estudiante codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante ALTER COLUMN codigo SET DEFAULT nextval('public.estudiante_codigo_seq'::regclass);


--
-- TOC entry 3387 (class 2604 OID 135655)
-- Name: estudiante_curso codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso ALTER COLUMN codigo SET DEFAULT nextval('public.estudiante_curso_codigo_seq'::regclass);


--
-- TOC entry 3381 (class 2604 OID 135574)
-- Name: factura codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.factura ALTER COLUMN codigo SET DEFAULT nextval('public.factura_codigo_seq'::regclass);


--
-- TOC entry 3390 (class 2604 OID 150370)
-- Name: funcionalidad codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.funcionalidad ALTER COLUMN codigo SET DEFAULT nextval('public.funcionalidad_codigo_seq'::regclass);


--
-- TOC entry 3391 (class 2604 OID 150379)
-- Name: permiso codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso ALTER COLUMN codigo SET DEFAULT nextval('public.permiso_codigo_seq'::regclass);


--
-- TOC entry 3395 (class 2604 OID 150427)
-- Name: precio codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.precio ALTER COLUMN codigo SET DEFAULT nextval('public.precio_codigo_seq'::regclass);


--
-- TOC entry 3383 (class 2604 OID 135595)
-- Name: profesor codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.profesor ALTER COLUMN codigo SET DEFAULT nextval('public.profesor_codigo_seq'::regclass);


--
-- TOC entry 3388 (class 2604 OID 150352)
-- Name: recurso codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.recurso ALTER COLUMN codigo SET DEFAULT nextval('public.recurso_codigo_seq'::regclass);


--
-- TOC entry 3389 (class 2604 OID 150361)
-- Name: rol codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.rol ALTER COLUMN codigo SET DEFAULT nextval('public.rol_codigo_seq'::regclass);


--
-- TOC entry 3382 (class 2604 OID 135586)
-- Name: tipo_estudiante codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.tipo_estudiante ALTER COLUMN codigo SET DEFAULT nextval('public.tipo_estudiante_codigo_seq'::regclass);


--
-- TOC entry 3378 (class 2604 OID 135556)
-- Name: users codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.users ALTER COLUMN codigo SET DEFAULT nextval('public.users_codigo_seq'::regclass);


--
-- TOC entry 3393 (class 2604 OID 150414)
-- Name: visita codigo; Type: DEFAULT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.visita ALTER COLUMN codigo SET DEFAULT nextval('public.visita_codigo_seq'::regclass);


--
-- TOC entry 3607 (class 0 OID 135620)
-- Dependencies: 234
-- Data for Name: cronograma; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.cronograma VALUES (2, 'Miércoles', '00:00:00', '00:00:00', 2, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.cronograma VALUES (3, 'Lunes', '08:00:00', '10:00:00', 1, '2025-06-26 11:40:52', '2025-06-26 11:40:52');
INSERT INTO public.cronograma VALUES (1, 'Martes', '09:00:00', '11:00:00', 1, '2025-06-25 23:45:59', '2025-06-26 11:41:59');
INSERT INTO public.cronograma VALUES (6, 'sabado', '07:00:00', '10:00:00', NULL, '2025-06-30 21:05:39', '2025-06-30 21:05:39');
INSERT INTO public.cronograma VALUES (7, 'domingo', '07:00:00', '09:00:00', NULL, '2025-06-30 21:05:40', '2025-06-30 21:05:40');
INSERT INTO public.cronograma VALUES (15, 'sabadoF', '07:01:00', '10:01:00', NULL, '2025-06-30 21:25:02', '2025-06-30 21:25:02');
INSERT INTO public.cronograma VALUES (16, 'LunesF', '10:01:00', '12:01:00', NULL, '2025-06-30 21:25:03', '2025-06-30 21:25:03');
INSERT INTO public.cronograma VALUES (8, 'martes', '10:00:00', '12:00:00', NULL, '2025-06-30 21:08:06', '2025-06-30 21:08:06');
INSERT INTO public.cronograma VALUES (25, 'lunes', '07:00:00', '12:00:00', 9, '2025-07-01 05:17:08', '2025-07-01 05:17:08');
INSERT INTO public.cronograma VALUES (26, 'martes', '07:00:00', '12:00:00', 9, '2025-07-01 05:17:09', '2025-07-01 05:17:09');
INSERT INTO public.cronograma VALUES (27, 'lunes', '10:00:00', '12:30:00', 8, '2025-07-01 05:18:31', '2025-07-01 05:18:31');
INSERT INTO public.cronograma VALUES (28, 'domingo', '07:00:00', '12:00:00', 8, '2025-07-01 05:18:32', '2025-07-01 05:18:32');
INSERT INTO public.cronograma VALUES (30, 'lunes', '07:00:00', '10:00:00', NULL, '2025-07-03 05:05:15', '2025-07-03 05:05:15');
INSERT INTO public.cronograma VALUES (31, 'Lunes', '08:30:00', '10:00:00', 11, '2025-07-03 12:38:05', '2025-07-03 12:38:05');
INSERT INTO public.cronograma VALUES (32, 'Miercoles', '08:30:00', '10:00:00', 11, '2025-07-03 12:38:05', '2025-07-03 12:38:05');


--
-- TOC entry 3605 (class 0 OID 135601)
-- Dependencies: 232
-- Data for Name: curso; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.curso VALUES (1, 'Introducción a Java', '40h', 30, '/images/curso1.jpg', true, 1, 1, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.curso VALUES (2, 'Desarrollo Web', '60h', 25, '/images/curso2.jpg', false, 2, 2, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.curso VALUES (8, 'Pruebaaaaa', '8 meses', 12, NULL, false, 6, 4, '2025-06-30 21:39:07', '2025-07-01 05:18:30');
INSERT INTO public.curso VALUES (9, 'Estadistica', '6 meses', 14, NULL, true, 8, 4, '2025-07-01 05:17:07', '2025-07-03 05:06:00');
INSERT INTO public.curso VALUES (11, 'Mantenimiento de computadoras', '40 horas', 9, NULL, true, 12, 4, '2025-07-03 12:38:05', '2025-08-18 19:23:14');


--
-- TOC entry 3597 (class 0 OID 135564)
-- Dependencies: 224
-- Data for Name: estado; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.estado VALUES (1, 'pendiente', '2025-06-25 22:51:33', '2025-06-25 22:51:33');
INSERT INTO public.estado VALUES (2, 'pagado', '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.estado VALUES (3, 'aprobado', '2025-06-25 23:45:59', '2025-06-25 23:45:59');


--
-- TOC entry 3609 (class 0 OID 135632)
-- Dependencies: 236
-- Data for Name: estudiante; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.estudiante VALUES (2, 'Laura', 'Diaz', 44444444, 2, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.estudiante VALUES (1, 'Carlos', 'Ramirez', 33333333, 2, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.estudiante VALUES (4, 'Lucía', 'Gómez', 99887766, 1, '2025-06-26 10:51:42', '2025-06-26 10:51:42');
INSERT INTO public.estudiante VALUES (5, 'Lucía', 'Gómez', 99887766, 1, '2025-06-26 16:21:47', '2025-06-26 16:21:47');
INSERT INTO public.estudiante VALUES (7, 'Prueba', 'Quispe', 345435, 1, '2025-07-02 03:21:44', '2025-07-02 03:21:44');
INSERT INTO public.estudiante VALUES (8, 'asdasd', 'sdfdf', 657657, 1, '2025-07-02 03:27:25', '2025-07-02 03:27:25');
INSERT INTO public.estudiante VALUES (9, 'dsfsf', 'dfsd', 65456, 1, '2025-07-02 03:29:48', '2025-07-02 03:29:48');
INSERT INTO public.estudiante VALUES (10, 'dfdsf', 'dfsdf', 3454656, 1, '2025-07-02 03:32:30', '2025-07-02 03:32:30');
INSERT INTO public.estudiante VALUES (11, 'dfgd', 'dfg', 7657, 1, '2025-07-02 03:33:44', '2025-07-02 03:33:44');
INSERT INTO public.estudiante VALUES (12, 'Pruebaa', 'aaaa', 45345, 1, '2025-07-02 03:42:50', '2025-07-02 03:42:50');
INSERT INTO public.estudiante VALUES (13, 'sfsd', 'cxv', 54646, 1, '2025-07-02 03:44:25', '2025-07-02 03:44:25');
INSERT INTO public.estudiante VALUES (14, 'rtytry', 'rtyrty', 345, 1, '2025-07-02 03:47:29', '2025-07-02 03:47:29');
INSERT INTO public.estudiante VALUES (15, 'xD', 'XDDD', 456443, 1, '2025-07-02 03:51:17', '2025-07-02 03:51:17');
INSERT INTO public.estudiante VALUES (16, 'asdsad', 'fgbcvbvcb', 65464, 1, '2025-07-02 03:54:51', '2025-07-02 03:54:51');
INSERT INTO public.estudiante VALUES (17, 'asdasd', 'dsfsdffff', 456565645, 1, '2025-07-02 03:57:12', '2025-07-02 03:57:12');
INSERT INTO public.estudiante VALUES (18, 'AAA', 'AAAAA', 43543, 1, '2025-07-03 05:05:58', '2025-07-03 05:05:58');
INSERT INTO public.estudiante VALUES (19, 'estudiante', 'nuevo', 1323932, 3, '2025-07-03 11:56:23', '2025-07-03 11:56:23');
INSERT INTO public.estudiante VALUES (20, 'estudiante', 'nuevo', 4545342, 1, '2025-07-03 12:40:39', '2025-07-03 12:40:39');
INSERT INTO public.estudiante VALUES (21, 'Josue', 'Veizaga', 12345678, 1, '2025-07-03 12:40:55', '2025-07-03 12:40:55');
INSERT INTO public.estudiante VALUES (22, 'asdf', 'asdff', 12341234, 1, '2025-07-04 15:41:16', '2025-07-04 15:41:16');
INSERT INTO public.estudiante VALUES (23, 'Ivan', 'Camargo', 9619700, 1, '2025-08-18 19:23:07', '2025-08-18 19:23:07');


--
-- TOC entry 3611 (class 0 OID 135652)
-- Dependencies: 238
-- Data for Name: estudiante_curso; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.estudiante_curso VALUES (4, 200, 2, 2, 1, NULL, '2025-06-25 23:48:53', '2025-06-25 23:48:53');
INSERT INTO public.estudiante_curso VALUES (3, 150, 1, 1, 2, 4, '2025-06-25 23:48:53', '2025-07-01 03:28:50');
INSERT INTO public.estudiante_curso VALUES (5, 150, 1, 2, 2, 4, '2025-06-26 11:13:37', '2025-07-01 03:28:51');
INSERT INTO public.estudiante_curso VALUES (8, 180, 2, 9, 1, NULL, '2025-07-02 01:24:32', '2025-07-02 01:24:32');
INSERT INTO public.estudiante_curso VALUES (2, 200, 2, 2, 3, 2, '2025-06-25 23:45:59', '2025-06-25 23:49:49');
INSERT INTO public.estudiante_curso VALUES (7, 180, 1, 9, 3, NULL, '2025-07-01 05:27:35', '2025-07-01 05:27:35');
INSERT INTO public.estudiante_curso VALUES (9, 120, 14, 9, 1, NULL, '2025-07-02 03:47:58', '2025-07-02 03:47:58');
INSERT INTO public.estudiante_curso VALUES (10, 120, 11, 9, 1, NULL, '2025-07-02 03:50:34', '2025-07-02 03:50:34');
INSERT INTO public.estudiante_curso VALUES (11, 120, 15, 9, 1, NULL, '2025-07-02 03:51:35', '2025-07-02 03:51:35');
INSERT INTO public.estudiante_curso VALUES (12, 120, 17, 9, 1, NULL, '2025-07-02 03:57:30', '2025-07-02 03:57:30');
INSERT INTO public.estudiante_curso VALUES (6, 100, 1, 8, 2, 5, '2025-07-01 04:35:57', '2025-07-03 02:11:43');
INSERT INTO public.estudiante_curso VALUES (13, 120, 18, 9, 1, NULL, '2025-07-03 05:06:00', '2025-07-03 05:06:00');
INSERT INTO public.estudiante_curso VALUES (15, 400, 21, 11, 1, NULL, '2025-07-03 12:40:57', '2025-07-03 12:40:57');
INSERT INTO public.estudiante_curso VALUES (14, 400, 20, 11, 2, 6, '2025-07-03 12:40:47', '2025-07-03 12:42:44');
INSERT INTO public.estudiante_curso VALUES (16, 400, 23, 11, 1, NULL, '2025-08-18 19:23:14', '2025-08-18 19:23:14');


--
-- TOC entry 3599 (class 0 OID 135571)
-- Dependencies: 226
-- Data for Name: factura; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.factura VALUES (1, 1, 500, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.factura VALUES (2, 1, 350, '2025-06-25 23:49:49', '2025-06-25 23:49:49');
INSERT INTO public.factura VALUES (3, 3, 420, '2025-06-26 11:28:18', '2025-06-26 11:29:03');
INSERT INTO public.factura VALUES (4, 4, 300, '2025-07-01 03:28:50', '2025-07-01 03:28:50');
INSERT INTO public.factura VALUES (5, 3, 100, '2025-07-03 02:11:43', '2025-07-03 02:11:43');
INSERT INTO public.factura VALUES (6, 3, 400, '2025-07-03 12:42:44', '2025-07-03 12:42:44');


--
-- TOC entry 3617 (class 0 OID 150367)
-- Dependencies: 244
-- Data for Name: funcionalidad; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.funcionalidad VALUES (9, 'roles', 'Gestión de los roles', 'bi bi-shield-shaded', 'roles.index');
INSERT INTO public.funcionalidad VALUES (1, 'usuarios', 'Gestión de usuarios', 'bi bi-person-standing', 'users.index');
INSERT INTO public.funcionalidad VALUES (3, 'estudiantes', 'Gestión de estudiantes', 'bi bi-tags-fill', 'estudiantes.index');
INSERT INTO public.funcionalidad VALUES (4, 'profesores', 'Gestión de profesores', 'bi bi-clipboard-check-fill', 'profesores.index');
INSERT INTO public.funcionalidad VALUES (5, 'facturas', 'Gestión de las facturas', 'bi bi-bag-check-fill', 'facturas.index');
INSERT INTO public.funcionalidad VALUES (2, 'cursos', 'Gestion de los Cursos', 'bi bi-speedometer', 'cursos.index');
INSERT INTO public.funcionalidad VALUES (6, 'inicio', 'vista de inicio', 'bi bi-microsoft', 'welcome');


--
-- TOC entry 3619 (class 0 OID 150376)
-- Dependencies: 246
-- Data for Name: permiso; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.permiso VALUES (1, true, 1, 9);
INSERT INTO public.permiso VALUES (2, true, 1, 1);
INSERT INTO public.permiso VALUES (3, true, 1, 2);
INSERT INTO public.permiso VALUES (4, true, 1, 3);
INSERT INTO public.permiso VALUES (5, true, 1, 4);
INSERT INTO public.permiso VALUES (7, true, 1, 6);
INSERT INTO public.permiso VALUES (16, true, 2, 5);


--
-- TOC entry 3620 (class 0 OID 150393)
-- Dependencies: 247
-- Data for Name: permiso_recurso; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.permiso_recurso VALUES (true, 16, 1);
INSERT INTO public.permiso_recurso VALUES (true, 16, 2);
INSERT INTO public.permiso_recurso VALUES (true, 2, 1);
INSERT INTO public.permiso_recurso VALUES (true, 2, 2);
INSERT INTO public.permiso_recurso VALUES (true, 16, 3);
INSERT INTO public.permiso_recurso VALUES (true, 3, 1);
INSERT INTO public.permiso_recurso VALUES (true, 3, 2);
INSERT INTO public.permiso_recurso VALUES (true, 4, 1);
INSERT INTO public.permiso_recurso VALUES (true, 4, 2);
INSERT INTO public.permiso_recurso VALUES (true, 5, 1);
INSERT INTO public.permiso_recurso VALUES (true, 5, 2);
INSERT INTO public.permiso_recurso VALUES (true, 7, 1);
INSERT INTO public.permiso_recurso VALUES (true, 7, 2);
INSERT INTO public.permiso_recurso VALUES (true, 1, 1);
INSERT INTO public.permiso_recurso VALUES (true, 1, 2);
INSERT INTO public.permiso_recurso VALUES (true, 1, 3);
INSERT INTO public.permiso_recurso VALUES (true, 1, 4);


--
-- TOC entry 3624 (class 0 OID 150424)
-- Dependencies: 251
-- Data for Name: precio; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.precio VALUES (4, 200, NULL, '2025-06-30 21:05:40', '2025-06-30 21:05:40', 2);
INSERT INTO public.precio VALUES (5, 100, NULL, '2025-06-30 21:05:41', '2025-06-30 21:05:41', 1);
INSERT INTO public.precio VALUES (6, 150, NULL, '2025-06-30 21:05:42', '2025-06-30 21:05:42', 3);
INSERT INTO public.precio VALUES (14, 201, NULL, '2025-06-30 21:25:04', '2025-06-30 21:25:04', 2);
INSERT INTO public.precio VALUES (15, 101, NULL, '2025-06-30 21:25:05', '2025-06-30 21:25:05', 1);
INSERT INTO public.precio VALUES (7, 120, NULL, '2025-06-30 21:08:07', '2025-06-30 21:08:07', 3);
INSERT INTO public.precio VALUES (27, 120, 9, '2025-07-01 05:17:09', '2025-07-01 05:17:09', 1);
INSERT INTO public.precio VALUES (28, 180, 9, '2025-07-01 05:17:10', '2025-07-01 05:17:10', 2);
INSERT INTO public.precio VALUES (29, 150, 9, '2025-07-01 05:17:11', '2025-07-01 05:17:11', 3);
INSERT INTO public.precio VALUES (30, 120, 8, '2025-07-01 05:18:34', '2025-07-01 05:18:34', 2);
INSERT INTO public.precio VALUES (31, 140, 8, '2025-07-01 05:18:34', '2025-07-01 05:18:34', 1);
INSERT INTO public.precio VALUES (32, 109, 8, '2025-07-01 05:18:35', '2025-07-01 05:18:35', 3);
INSERT INTO public.precio VALUES (34, 20, NULL, '2025-07-03 05:05:15', '2025-07-03 05:05:15', 2);
INSERT INTO public.precio VALUES (35, 300, 11, '2025-07-03 12:38:05', '2025-07-03 12:38:05', 3);
INSERT INTO public.precio VALUES (36, 350, 11, '2025-07-03 12:38:05', '2025-07-03 12:38:05', 2);
INSERT INTO public.precio VALUES (37, 400, 11, '2025-07-03 12:38:05', '2025-07-03 12:38:05', 1);


--
-- TOC entry 3603 (class 0 OID 135592)
-- Dependencies: 230
-- Data for Name: profesor; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.profesor VALUES (1, 'Jose', 'Lopez', 'Lic.', '/images/prof1.jpg', 11111111, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.profesor VALUES (2, 'Ana', 'Martinez', 'Ing.', '/images/prof2.jpg', 22222222, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.profesor VALUES (3, 'Juan', 'Perez', 'Lic. en Matemáticas', 'foto.jpg', 1234567, '2025-06-25 23:48:32', '2025-06-25 23:48:32');
INSERT INTO public.profesor VALUES (4, 'Juan', 'Perez', 'Lic. en Matematicas', 'foto.jpg', 1234567, '2025-06-26 00:21:58', '2025-06-26 00:21:58');
INSERT INTO public.profesor VALUES (6, 'Acentooooo', 'Peréz', 'Lic. en Matemáticas', 'foto.jpg', 1234567, '2025-06-26 00:28:35', '2025-06-26 00:31:42');
INSERT INTO public.profesor VALUES (7, 'gustavo', 'camargo', 'ing', 'profesores/cfU29JbOwW4TtT0uTfZ2aJDWg7Or3Kq3vJvTgS71.webp', 1232343, '2025-07-01 02:34:27', '2025-07-01 02:34:27');
INSERT INTO public.profesor VALUES (8, 'gustavo', 'camargo', 'ing', 'profesores/1WLGLlgUi5JFZarVKltZ2fvZLDxHUVkdPtgVpStS.jpg', 1232343123, '2025-07-01 02:35:20', '2025-07-01 02:35:20');
INSERT INTO public.profesor VALUES (9, 'profesor', 'nuevo', 'ing', 'profesores/UT0MIKMNW6JrqxyJskpjxe3hB3aqgp5DJZ8AUQHt.png', 140129292, '2025-07-03 11:53:58', '2025-07-03 11:53:58');
INSERT INTO public.profesor VALUES (10, 'nuevo', 'profesor', 'ing', NULL, 2323412, '2025-07-03 11:55:07', '2025-07-03 11:55:07');
INSERT INTO public.profesor VALUES (11, 'Gino', 'Barroso Viruez', 'Ing.', NULL, 11223344, '2025-07-03 12:36:45', '2025-07-03 12:36:45');
INSERT INTO public.profesor VALUES (12, 'Jose', 'Roman', 'ing', NULL, 12238233, '2025-07-03 12:36:47', '2025-07-03 12:36:47');


--
-- TOC entry 3613 (class 0 OID 150349)
-- Dependencies: 240
-- Data for Name: recurso; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.recurso VALUES (1, 'view-any', 'Ver todos los recursos');
INSERT INTO public.recurso VALUES (2, 'view', 'Ver un recurso específico');
INSERT INTO public.recurso VALUES (3, 'create', 'Crear un recurso');
INSERT INTO public.recurso VALUES (4, 'update', 'Actualizar un recurso');
INSERT INTO public.recurso VALUES (5, 'delete', 'Eliminar un recursos');
INSERT INTO public.recurso VALUES (6, 'metodo-pago', 'Realizar pago');
INSERT INTO public.recurso VALUES (7, 'ingresos', 'Ingresar productos');
INSERT INTO public.recurso VALUES (8, 'export-pdf', 'Exportar a pdf');
INSERT INTO public.recurso VALUES (9, 'export-csv', 'Exportar a csv');
INSERT INTO public.recurso VALUES (10, 'service-attention', 'Atención de servicios');
INSERT INTO public.recurso VALUES (11, 'assign-functionality', 'Asignar funcionalidad');
INSERT INTO public.recurso VALUES (12, 'history-stock', 'Mostrar historial de stock');


--
-- TOC entry 3615 (class 0 OID 150358)
-- Dependencies: 242
-- Data for Name: rol; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.rol VALUES (1, 'coordinador', 'Coordinador responsable de la CICIT');
INSERT INTO public.rol VALUES (2, 'secretario', 'Secretario que se encarga de los pagos por los cursos');
INSERT INTO public.rol VALUES (3, 'publico', 'usuario publico que solo puede ver los cursos y preinscribirse a los mismos');


--
-- TOC entry 3601 (class 0 OID 135583)
-- Dependencies: 228
-- Data for Name: tipo_estudiante; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.tipo_estudiante VALUES (2, 'Universidad', true, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.tipo_estudiante VALUES (1, 'Particular', true, '2025-06-25 23:45:59', '2025-06-25 23:45:59');
INSERT INTO public.tipo_estudiante VALUES (3, 'Facultad', true, '2025-06-26 09:56:44', '2025-06-26 09:56:44');


--
-- TOC entry 3595 (class 0 OID 135553)
-- Dependencies: 222
-- Data for Name: users; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.users VALUES (1, 'Ana López', '87654321', 'ana@example.com', NULL, '$2y$12$6oCqXnbhJIk2qQMr3kQ4vexyGEyUO0Sx0tmybBqdhZ3I96gQ4F6d6', NULL, '2025-06-25 23:03:10', '2025-06-25 23:14:23', 1);
INSERT INTO public.users VALUES (3, 'Maria Perez', '87654321', 'maria@example.com', NULL, '$2y$12$6oCqXnbhJIk2qQMr3kQ4vexyGEyUO0Sx0tmybBqdhZ3I96gQ4F6d6', NULL, '2025-06-25 23:45:59', '2025-06-25 23:45:59', 2);
INSERT INTO public.users VALUES (2, 'Luis Gomez', '12345678', 'luis@example.com', NULL, '$2y$12$6oCqXnbhJIk2qQMr3kQ4vexyGEyUO0Sx0tmybBqdhZ3I96gQ4F6d6', NULL, '2025-06-25 23:45:59', '2025-06-25 23:45:59', 1);
INSERT INTO public.users VALUES (4, 'admin', '3435453', 'admin@gmail.com', NULL, '$2y$12$6oCqXnbhJIk2qQMr3kQ4vexyGEyUO0Sx0tmybBqdhZ3I96gQ4F6d6', NULL, '2025-06-25 23:03:10', '2025-06-25 23:03:10', 1);
INSERT INTO public.users VALUES (5, 'gustavos', '140116292', 'gusta25demarzo@gmail.coma', NULL, '$2y$12$8CNPxruUyHidZuxkrYpNAObg9ESeB9SK.Xtl.RiJgLvrR/n5mao9.', NULL, NULL, NULL, 2);


--
-- TOC entry 3622 (class 0 OID 150411)
-- Dependencies: 249
-- Data for Name: visita; Type: TABLE DATA; Schema: public; Owner: grupo07sc
--

INSERT INTO public.visita VALUES (41, '/users/5', 1, 4);
INSERT INTO public.visita VALUES (14, '/', 49, 4);
INSERT INTO public.visita VALUES (1, '/home', 7, 2);
INSERT INTO public.visita VALUES (78, '/api/estudiante/454654', 1, 4);
INSERT INTO public.visita VALUES (4, '/livewire/update', 35, 2);
INSERT INTO public.visita VALUES (34, '/roles/4/functionalities', 1, 4);
INSERT INTO public.visita VALUES (3, '/forbidden', 3, 2);
INSERT INTO public.visita VALUES (6, '/login', 8, NULL);
INSERT INTO public.visita VALUES (84, '/api/estudiante/546546', 1, 4);
INSERT INTO public.visita VALUES (11, '/roles', 1, 1);
INSERT INTO public.visita VALUES (10, '/home', 2, 1);
INSERT INTO public.visita VALUES (12, '/users', 1, 1);
INSERT INTO public.visita VALUES (52, '/estudiantes/1/edit', 11, 4);
INSERT INTO public.visita VALUES (17, '/cursos', 61, 4);
INSERT INTO public.visita VALUES (28, '/cursos/editar/6', 3, 4);
INSERT INTO public.visita VALUES (94, '/api/estudiante/456443', 1, 4);
INSERT INTO public.visita VALUES (13, '/home', 4, 4);
INSERT INTO public.visita VALUES (35, '/cursos/eliminar/5', 1, 4);
INSERT INTO public.visita VALUES (8, '/ventas', 1, 3);
INSERT INTO public.visita VALUES (73, '/api/estudiante/5345', 2, 4);
INSERT INTO public.visita VALUES (29, '/cursos/editar/4', 4, 4);
INSERT INTO public.visita VALUES (30, '/cursos/actualizar/4', 3, 4);
INSERT INTO public.visita VALUES (2, '/roles', 9, 2);
INSERT INTO public.visita VALUES (36, '/cursos/eliminar/4', 1, 4);
INSERT INTO public.visita VALUES (7, '/home', 2, 3);
INSERT INTO public.visita VALUES (50, '/profesores/7/edit', 2, 4);
INSERT INTO public.visita VALUES (85, '/api/estudiante/65456', 1, 4);
INSERT INTO public.visita VALUES (18, '/api/get-cursos', 89, 4);
INSERT INTO public.visita VALUES (20, '/roles', 88, 4);
INSERT INTO public.visita VALUES (33, '/roles/create', 9, 4);
INSERT INTO public.visita VALUES (27, '/cursos/guardar', 17, 4);
INSERT INTO public.visita VALUES (108, '/cursos/inscripciones/7/formulario', 3, 3);
INSERT INTO public.visita VALUES (37, '/cursos/eliminar/3', 1, 4);
INSERT INTO public.visita VALUES (51, '/estudiantes/4/edit', 1, 4);
INSERT INTO public.visita VALUES (38, '/cursos/eliminar/6', 1, 4);
INSERT INTO public.visita VALUES (21, '/roles/1/edit', 2, 4);
INSERT INTO public.visita VALUES (22, '/roles/1', 2, 4);
INSERT INTO public.visita VALUES (79, '/cursos/9/inscripciones/export', 4, 4);
INSERT INTO public.visita VALUES (54, '/facturas/1', 1, 4);
INSERT INTO public.visita VALUES (44, '/estudiantes/6/edit', 1, 4);
INSERT INTO public.visita VALUES (86, '/api/estudiante/3454656', 1, 4);
INSERT INTO public.visita VALUES (88, '/inscripciones/7/certificado', 3, 4);
INSERT INTO public.visita VALUES (24, '/inicio', 9, 4);
INSERT INTO public.visita VALUES (45, '/estudiantes/6', 2, 4);
INSERT INTO public.visita VALUES (56, '/facturas/create', 8, 4);
INSERT INTO public.visita VALUES (59, '/facturas/4/pdf', 5, 4);
INSERT INTO public.visita VALUES (80, '/api/estudiante/4564565', 1, 4);
INSERT INTO public.visita VALUES (55, '/facturas/2', 2, 4);
INSERT INTO public.visita VALUES (23, '/roles/1/functionalities', 60, 4);
INSERT INTO public.visita VALUES (120, '/users/6', 2, 4);
INSERT INTO public.visita VALUES (49, '/profesores/8/edit', 44, 4);
INSERT INTO public.visita VALUES (9, '/dashboard', 115, 4);
INSERT INTO public.visita VALUES (95, '/api/estudiante/65464', 1, 4);
INSERT INTO public.visita VALUES (47, '/profesores/1/edit', 1, 4);
INSERT INTO public.visita VALUES (16, '/users', 76, 4);
INSERT INTO public.visita VALUES (71, '/estudiantes/export', 4, 4);
INSERT INTO public.visita VALUES (97, '/public/inscripciones/buscar', 4, 4);
INSERT INTO public.visita VALUES (53, '/facturas', 24, 4);
INSERT INTO public.visita VALUES (75, '/api/estudiante/45646', 2, 4);
INSERT INTO public.visita VALUES (101, '/api/estudiante/3333333', 1, 4);
INSERT INTO public.visita VALUES (40, '/users/5/edit', 1, 4);
INSERT INTO public.visita VALUES (57, '/facturas/4', 19, 4);
INSERT INTO public.visita VALUES (48, '/profesores/create', 9, 4);
INSERT INTO public.visita VALUES (113, '/facturas/create', 11, 3);
INSERT INTO public.visita VALUES (103, '/dashboard', 4, 3);
INSERT INTO public.visita VALUES (77, '/api/estudiante/345345', 1, 4);
INSERT INTO public.visita VALUES (61, '/api/estudiante/333333', 1, 4);
INSERT INTO public.visita VALUES (67, '/cursos/1/inscripcion', 1, 4);
INSERT INTO public.visita VALUES (65, '/cursos/', 1, 4);
INSERT INTO public.visita VALUES (76, '/facturas/export', 2, 4);
INSERT INTO public.visita VALUES (68, '/cursos/1/estudiantes', 1, 4);
INSERT INTO public.visita VALUES (39, '/users/create', 13, 4);
INSERT INTO public.visita VALUES (31, '/cursos/editar/8', 6, 4);
INSERT INTO public.visita VALUES (102, '/roles/2/functionalities', 4, 4);
INSERT INTO public.visita VALUES (81, '/api/estudiante/345435', 2, 4);
INSERT INTO public.visita VALUES (32, '/cursos/actualizar/8', 4, 4);
INSERT INTO public.visita VALUES (58, '/cursos/8/inscripcion', 17, 4);
INSERT INTO public.visita VALUES (92, '/api/estudiante/6546', 1, 4);
INSERT INTO public.visita VALUES (96, '/api/estudiante/456565645', 2, 4);
INSERT INTO public.visita VALUES (15, '/login', 10, 4);
INSERT INTO public.visita VALUES (100, '/api/estudiante/33333', 1, 4);
INSERT INTO public.visita VALUES (89, '/inscripciones/7/formulario', 2, 4);
INSERT INTO public.visita VALUES (90, '/api/estudiante/45345', 1, 4);
INSERT INTO public.visita VALUES (105, '/', 1, 3);
INSERT INTO public.visita VALUES (42, '/estudiantes', 60, 4);
INSERT INTO public.visita VALUES (72, '/profesores/export', 2, 4);
INSERT INTO public.visita VALUES (69, '/api/estudiante/44444444', 4, 4);
INSERT INTO public.visita VALUES (63, '/api/inscripcion', 18, 4);
INSERT INTO public.visita VALUES (83, '/api/estudiante/657657', 1, 4);
INSERT INTO public.visita VALUES (93, '/api/estudiante/345', 1, 4);
INSERT INTO public.visita VALUES (62, '/logout', 7, 4);
INSERT INTO public.visita VALUES (66, '/cursos/9/estudiantes', 33, 4);
INSERT INTO public.visita VALUES (64, '/cursos/9/inscripcion', 53, 4);
INSERT INTO public.visita VALUES (91, '/api/estudiante/54646', 1, 4);
INSERT INTO public.visita VALUES (98, '/cursos/inscripciones/buscar', 3, 4);
INSERT INTO public.visita VALUES (87, '/api/estudiante/7657', 2, 4);
INSERT INTO public.visita VALUES (106, '/api/get-cursos', 1, 3);
INSERT INTO public.visita VALUES (25, '/cursos/inicio', 114, 4);
INSERT INTO public.visita VALUES (99, '/cursos/inscripciones/7/formulario', 2, 4);
INSERT INTO public.visita VALUES (60, '/api/estudiante/33333333', 26, 4);
INSERT INTO public.visita VALUES (107, '/cursos/inscripciones/buscar', 1, 3);
INSERT INTO public.visita VALUES (109, '/cursos/inscripciones/7/certificado', 2, 3);
INSERT INTO public.visita VALUES (111, '/facturas/4', 2, 3);
INSERT INTO public.visita VALUES (110, '/cursos/inscripciones/6/formulario', 2, 3);
INSERT INTO public.visita VALUES (19, '/funcionalidades/search', 22, 4);
INSERT INTO public.visita VALUES (112, '/facturas/4/pdf', 2, 3);
INSERT INTO public.visita VALUES (43, '/estudiantes/create', 6, 4);
INSERT INTO public.visita VALUES (104, '/facturas', 9, 3);
INSERT INTO public.visita VALUES (114, '/facturas/5', 2, 3);
INSERT INTO public.visita VALUES (115, '/facturas/5/pdf', 1, 3);
INSERT INTO public.visita VALUES (116, '/roles/5/functionalities', 3, 4);
INSERT INTO public.visita VALUES (117, '/roles/5/edit', 1, 4);
INSERT INTO public.visita VALUES (118, '/roles/5', 2, 4);
INSERT INTO public.visita VALUES (119, '/users/6/edit', 1, 4);
INSERT INTO public.visita VALUES (70, '/users/export', 12, 4);
INSERT INTO public.visita VALUES (121, '/cursos/10/estudiantes', 1, 4);
INSERT INTO public.visita VALUES (74, '/cursos/export', 7, 4);
INSERT INTO public.visita VALUES (122, '/cursos/editar/10', 1, 4);
INSERT INTO public.visita VALUES (123, '/cursos/actualizar/10', 1, 4);
INSERT INTO public.visita VALUES (124, '/cursos/eliminar/10', 1, 4);
INSERT INTO public.visita VALUES (125, '/api/estudiante/43543', 1, 4);
INSERT INTO public.visita VALUES (26, '/cursos/crear', 37, 4);
INSERT INTO public.visita VALUES (126, '/cursos/11/inscripcion', 1, 4);
INSERT INTO public.visita VALUES (127, '/api/estudiante/12345678', 1, 4);
INSERT INTO public.visita VALUES (82, '/api/estudianteNuevo', 13, 4);
INSERT INTO public.visita VALUES (129, '/facturas/6', 2, 3);
INSERT INTO public.visita VALUES (128, '/cursos/11/estudiantes', 3, 4);
INSERT INTO public.visita VALUES (46, '/profesores', 35, 4);
INSERT INTO public.visita VALUES (130, '/profile', 2, 4);


--
-- TOC entry 3647 (class 0 OID 0)
-- Dependencies: 233
-- Name: cronograma_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.cronograma_codigo_seq', 32, true);


--
-- TOC entry 3648 (class 0 OID 0)
-- Dependencies: 231
-- Name: curso_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.curso_codigo_seq', 11, true);


--
-- TOC entry 3649 (class 0 OID 0)
-- Dependencies: 223
-- Name: estado_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.estado_codigo_seq', 5, true);


--
-- TOC entry 3650 (class 0 OID 0)
-- Dependencies: 235
-- Name: estudiante_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.estudiante_codigo_seq', 23, true);


--
-- TOC entry 3651 (class 0 OID 0)
-- Dependencies: 237
-- Name: estudiante_curso_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.estudiante_curso_codigo_seq', 16, true);


--
-- TOC entry 3652 (class 0 OID 0)
-- Dependencies: 225
-- Name: factura_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.factura_codigo_seq', 6, true);


--
-- TOC entry 3653 (class 0 OID 0)
-- Dependencies: 243
-- Name: funcionalidad_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.funcionalidad_codigo_seq', 10, true);


--
-- TOC entry 3654 (class 0 OID 0)
-- Dependencies: 245
-- Name: permiso_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.permiso_codigo_seq', 17, true);


--
-- TOC entry 3655 (class 0 OID 0)
-- Dependencies: 250
-- Name: precio_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.precio_codigo_seq', 37, true);


--
-- TOC entry 3656 (class 0 OID 0)
-- Dependencies: 229
-- Name: profesor_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.profesor_codigo_seq', 12, true);


--
-- TOC entry 3657 (class 0 OID 0)
-- Dependencies: 239
-- Name: recurso_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.recurso_codigo_seq', 12, true);


--
-- TOC entry 3658 (class 0 OID 0)
-- Dependencies: 241
-- Name: rol_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.rol_codigo_seq', 5, true);


--
-- TOC entry 3659 (class 0 OID 0)
-- Dependencies: 227
-- Name: tipo_estudiante_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.tipo_estudiante_codigo_seq', 5, true);


--
-- TOC entry 3660 (class 0 OID 0)
-- Dependencies: 221
-- Name: users_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.users_codigo_seq', 6, true);


--
-- TOC entry 3661 (class 0 OID 0)
-- Dependencies: 248
-- Name: visita_codigo_seq; Type: SEQUENCE SET; Schema: public; Owner: grupo07sc
--

SELECT pg_catalog.setval('public.visita_codigo_seq', 130, true);


--
-- TOC entry 3414 (class 2606 OID 135625)
-- Name: cronograma cronograma_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.cronograma
    ADD CONSTRAINT cronograma_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3412 (class 2606 OID 135608)
-- Name: curso curso_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.curso
    ADD CONSTRAINT curso_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3404 (class 2606 OID 135569)
-- Name: estado estado_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estado
    ADD CONSTRAINT estado_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3418 (class 2606 OID 135657)
-- Name: estudiante_curso estudiante_curso_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso
    ADD CONSTRAINT estudiante_curso_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3416 (class 2606 OID 135637)
-- Name: estudiante estudiante_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante
    ADD CONSTRAINT estudiante_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3406 (class 2606 OID 135576)
-- Name: factura factura_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.factura
    ADD CONSTRAINT factura_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3424 (class 2606 OID 150374)
-- Name: funcionalidad funcionalidad_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.funcionalidad
    ADD CONSTRAINT funcionalidad_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3426 (class 2606 OID 150381)
-- Name: permiso permiso_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso
    ADD CONSTRAINT permiso_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3428 (class 2606 OID 151029)
-- Name: permiso_recurso permiso_recurso_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso_recurso
    ADD CONSTRAINT permiso_recurso_pkey PRIMARY KEY (recurso_id, permiso_id);


--
-- TOC entry 3433 (class 2606 OID 150431)
-- Name: precio precio_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.precio
    ADD CONSTRAINT precio_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3410 (class 2606 OID 135599)
-- Name: profesor profesor_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.profesor
    ADD CONSTRAINT profesor_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3420 (class 2606 OID 150356)
-- Name: recurso recurso_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.recurso
    ADD CONSTRAINT recurso_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3422 (class 2606 OID 150365)
-- Name: rol rol_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.rol
    ADD CONSTRAINT rol_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3408 (class 2606 OID 135590)
-- Name: tipo_estudiante tipo_estudiante_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.tipo_estudiante
    ADD CONSTRAINT tipo_estudiante_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3400 (class 2606 OID 135562)
-- Name: users users_email_unique; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_email_unique UNIQUE (email);


--
-- TOC entry 3402 (class 2606 OID 135560)
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3430 (class 2606 OID 150417)
-- Name: visita visita_pkey; Type: CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.visita
    ADD CONSTRAINT visita_pkey PRIMARY KEY (codigo);


--
-- TOC entry 3431 (class 1259 OID 176109)
-- Name: fki_t; Type: INDEX; Schema: public; Owner: grupo07sc
--

CREATE INDEX fki_t ON public.precio USING btree (tipo_estudiante_id);


--
-- TOC entry 3398 (class 1259 OID 176115)
-- Name: fki_users_rol; Type: INDEX; Schema: public; Owner: grupo07sc
--

CREATE INDEX fki_users_rol ON public.users USING btree (rol_id);


--
-- TOC entry 3438 (class 2606 OID 135626)
-- Name: cronograma cronograma_curso_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.cronograma
    ADD CONSTRAINT cronograma_curso_id_foreign FOREIGN KEY (curso_id) REFERENCES public.curso(codigo) ON DELETE SET NULL;


--
-- TOC entry 3436 (class 2606 OID 135609)
-- Name: curso curso_profesor_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.curso
    ADD CONSTRAINT curso_profesor_id_foreign FOREIGN KEY (profesor_id) REFERENCES public.profesor(codigo) ON DELETE SET NULL;


--
-- TOC entry 3437 (class 2606 OID 135614)
-- Name: curso curso_users_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.curso
    ADD CONSTRAINT curso_users_id_foreign FOREIGN KEY (users_id) REFERENCES public.users(codigo) ON DELETE SET NULL;


--
-- TOC entry 3440 (class 2606 OID 135663)
-- Name: estudiante_curso estudiante_curso_curso_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso
    ADD CONSTRAINT estudiante_curso_curso_id_foreign FOREIGN KEY (curso_id) REFERENCES public.curso(codigo) ON DELETE CASCADE;


--
-- TOC entry 3441 (class 2606 OID 135668)
-- Name: estudiante_curso estudiante_curso_estado_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso
    ADD CONSTRAINT estudiante_curso_estado_id_foreign FOREIGN KEY (estado_id) REFERENCES public.estado(codigo) ON DELETE SET NULL;


--
-- TOC entry 3442 (class 2606 OID 135658)
-- Name: estudiante_curso estudiante_curso_estudiante_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso
    ADD CONSTRAINT estudiante_curso_estudiante_id_foreign FOREIGN KEY (estudiante_id) REFERENCES public.estudiante(codigo) ON DELETE SET NULL;


--
-- TOC entry 3443 (class 2606 OID 135673)
-- Name: estudiante_curso estudiante_curso_factura_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante_curso
    ADD CONSTRAINT estudiante_curso_factura_id_foreign FOREIGN KEY (factura_id) REFERENCES public.factura(codigo) ON DELETE SET NULL;


--
-- TOC entry 3439 (class 2606 OID 135639)
-- Name: estudiante estudiante_tipo_estudiante_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.estudiante
    ADD CONSTRAINT estudiante_tipo_estudiante_id_foreign FOREIGN KEY (tipo_estudiante_id) REFERENCES public.tipo_estudiante(codigo) ON DELETE SET NULL;


--
-- TOC entry 3435 (class 2606 OID 135577)
-- Name: factura factura_users_id_foreign; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.factura
    ADD CONSTRAINT factura_users_id_foreign FOREIGN KEY (users_id) REFERENCES public.users(codigo) ON DELETE SET NULL;


--
-- TOC entry 3444 (class 2606 OID 150387)
-- Name: permiso fk_permiso_funcionalidad; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso
    ADD CONSTRAINT fk_permiso_funcionalidad FOREIGN KEY (funcionalidad_id) REFERENCES public.funcionalidad(codigo) ON DELETE CASCADE;


--
-- TOC entry 3446 (class 2606 OID 150400)
-- Name: permiso_recurso fk_permiso_recurso_permiso; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso_recurso
    ADD CONSTRAINT fk_permiso_recurso_permiso FOREIGN KEY (permiso_id) REFERENCES public.permiso(codigo) ON DELETE CASCADE;


--
-- TOC entry 3447 (class 2606 OID 150405)
-- Name: permiso_recurso fk_permiso_recurso_recurso; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso_recurso
    ADD CONSTRAINT fk_permiso_recurso_recurso FOREIGN KEY (recurso_id) REFERENCES public.recurso(codigo) ON DELETE CASCADE;


--
-- TOC entry 3445 (class 2606 OID 150382)
-- Name: permiso fk_permiso_rol; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.permiso
    ADD CONSTRAINT fk_permiso_rol FOREIGN KEY (rol_id) REFERENCES public.rol(codigo) ON DELETE CASCADE;


--
-- TOC entry 3449 (class 2606 OID 150432)
-- Name: precio fk_precio_curso; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.precio
    ADD CONSTRAINT fk_precio_curso FOREIGN KEY (curso_id) REFERENCES public.curso(codigo) ON DELETE SET NULL;


--
-- TOC entry 3450 (class 2606 OID 176104)
-- Name: precio fk_tipo_estudiante; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.precio
    ADD CONSTRAINT fk_tipo_estudiante FOREIGN KEY (tipo_estudiante_id) REFERENCES public.tipo_estudiante(codigo) ON UPDATE CASCADE ON DELETE SET NULL NOT VALID;


--
-- TOC entry 3448 (class 2606 OID 150418)
-- Name: visita fk_visita_user; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.visita
    ADD CONSTRAINT fk_visita_user FOREIGN KEY (users_id) REFERENCES public.users(codigo) ON DELETE SET NULL;


--
-- TOC entry 3434 (class 2606 OID 176110)
-- Name: users users_rol; Type: FK CONSTRAINT; Schema: public; Owner: grupo07sc
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_rol FOREIGN KEY (rol_id) REFERENCES public.rol(codigo) ON UPDATE CASCADE ON DELETE SET NULL NOT VALID;


--
-- TOC entry 3631 (class 0 OID 0)
-- Dependencies: 9
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: grupo07sc
--

REVOKE USAGE ON SCHEMA public FROM PUBLIC;


-- Completed on 2025-09-14 17:43:38

--
-- PostgreSQL database dump complete
--

\unrestrict cBR7ipK0mmvSgk7DHwhaFedZ0ZLYjNpVZM59aNbHOcImyDROhqcMePcvQj457Bt


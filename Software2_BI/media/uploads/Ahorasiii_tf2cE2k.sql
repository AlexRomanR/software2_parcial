--
-- PostgreSQL database dump
--

\restrict 2iv7kOFj04hIgDwiBuyVpH48TfMiOTWaj9IaPAGDnpnTZlredtGpEwjsGD0hQon

-- Dumped from database version 17.6
-- Dumped by pg_dump version 17.6

-- Started on 2025-09-14 17:30:38

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
-- TOC entry 13 (class 2615 OID 17891)
-- Name: user_1_file_00a1f910; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_00a1f910;


ALTER SCHEMA user_1_file_00a1f910 OWNER TO postgres;

--
-- TOC entry 9 (class 2615 OID 17096)
-- Name: user_1_file_08bbf7e0; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_08bbf7e0;


ALTER SCHEMA user_1_file_08bbf7e0 OWNER TO postgres;

--
-- TOC entry 30 (class 2615 OID 18014)
-- Name: user_1_file_0ab7d650; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_0ab7d650;


ALTER SCHEMA user_1_file_0ab7d650 OWNER TO postgres;

--
-- TOC entry 19 (class 2615 OID 17972)
-- Name: user_1_file_0aff13a0; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_0aff13a0;


ALTER SCHEMA user_1_file_0aff13a0 OWNER TO postgres;

--
-- TOC entry 28 (class 2615 OID 18012)
-- Name: user_1_file_0f166322; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_0f166322;


ALTER SCHEMA user_1_file_0f166322 OWNER TO postgres;

--
-- TOC entry 26 (class 2615 OID 18010)
-- Name: user_1_file_1c8551b4; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_1c8551b4;


ALTER SCHEMA user_1_file_1c8551b4 OWNER TO postgres;

--
-- TOC entry 36 (class 2615 OID 18020)
-- Name: user_1_file_1d7c681b; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_1d7c681b;


ALTER SCHEMA user_1_file_1d7c681b OWNER TO postgres;

--
-- TOC entry 31 (class 2615 OID 18015)
-- Name: user_1_file_2bcfbb24; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_2bcfbb24;


ALTER SCHEMA user_1_file_2bcfbb24 OWNER TO postgres;

--
-- TOC entry 34 (class 2615 OID 18018)
-- Name: user_1_file_314393bd; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_314393bd;


ALTER SCHEMA user_1_file_314393bd OWNER TO postgres;

--
-- TOC entry 18 (class 2615 OID 17971)
-- Name: user_1_file_33f586fe; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_33f586fe;


ALTER SCHEMA user_1_file_33f586fe OWNER TO postgres;

--
-- TOC entry 22 (class 2615 OID 17975)
-- Name: user_1_file_361deb5e; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_361deb5e;


ALTER SCHEMA user_1_file_361deb5e OWNER TO postgres;

--
-- TOC entry 27 (class 2615 OID 18011)
-- Name: user_1_file_51e1937a; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_51e1937a;


ALTER SCHEMA user_1_file_51e1937a OWNER TO postgres;

--
-- TOC entry 21 (class 2615 OID 17974)
-- Name: user_1_file_69c52904; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_69c52904;


ALTER SCHEMA user_1_file_69c52904 OWNER TO postgres;

--
-- TOC entry 25 (class 2615 OID 18009)
-- Name: user_1_file_6d3ff0be; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_6d3ff0be;


ALTER SCHEMA user_1_file_6d3ff0be OWNER TO postgres;

--
-- TOC entry 20 (class 2615 OID 17973)
-- Name: user_1_file_749e8c7f; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_749e8c7f;


ALTER SCHEMA user_1_file_749e8c7f OWNER TO postgres;

--
-- TOC entry 6 (class 2615 OID 17081)
-- Name: user_1_file_7639c196; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_7639c196;


ALTER SCHEMA user_1_file_7639c196 OWNER TO postgres;

--
-- TOC entry 12 (class 2615 OID 17164)
-- Name: user_1_file_7fad03c9; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_7fad03c9;


ALTER SCHEMA user_1_file_7fad03c9 OWNER TO postgres;

--
-- TOC entry 35 (class 2615 OID 18019)
-- Name: user_1_file_80f5aa96; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_80f5aa96;


ALTER SCHEMA user_1_file_80f5aa96 OWNER TO postgres;

--
-- TOC entry 14 (class 2615 OID 17967)
-- Name: user_1_file_93c7c2b7; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_93c7c2b7;


ALTER SCHEMA user_1_file_93c7c2b7 OWNER TO postgres;

--
-- TOC entry 32 (class 2615 OID 18016)
-- Name: user_1_file_9fef015f; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_9fef015f;


ALTER SCHEMA user_1_file_9fef015f OWNER TO postgres;

--
-- TOC entry 33 (class 2615 OID 18017)
-- Name: user_1_file_a7b43904; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_a7b43904;


ALTER SCHEMA user_1_file_a7b43904 OWNER TO postgres;

--
-- TOC entry 29 (class 2615 OID 18013)
-- Name: user_1_file_aa5e966d; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_aa5e966d;


ALTER SCHEMA user_1_file_aa5e966d OWNER TO postgres;

--
-- TOC entry 38 (class 2615 OID 18096)
-- Name: user_1_file_ad5c2d2d; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_ad5c2d2d;


ALTER SCHEMA user_1_file_ad5c2d2d OWNER TO postgres;

--
-- TOC entry 23 (class 2615 OID 17976)
-- Name: user_1_file_b39d309b; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_b39d309b;


ALTER SCHEMA user_1_file_b39d309b OWNER TO postgres;

--
-- TOC entry 17 (class 2615 OID 17970)
-- Name: user_1_file_c31e2c7a; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_c31e2c7a;


ALTER SCHEMA user_1_file_c31e2c7a OWNER TO postgres;

--
-- TOC entry 24 (class 2615 OID 18008)
-- Name: user_1_file_cab7469b; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_cab7469b;


ALTER SCHEMA user_1_file_cab7469b OWNER TO postgres;

--
-- TOC entry 11 (class 2615 OID 17098)
-- Name: user_1_file_d0663889; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_d0663889;


ALTER SCHEMA user_1_file_d0663889 OWNER TO postgres;

--
-- TOC entry 7 (class 2615 OID 17088)
-- Name: user_1_file_d4a80680; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_d4a80680;


ALTER SCHEMA user_1_file_d4a80680 OWNER TO postgres;

--
-- TOC entry 8 (class 2615 OID 17095)
-- Name: user_1_file_d803d64d; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_d803d64d;


ALTER SCHEMA user_1_file_d803d64d OWNER TO postgres;

--
-- TOC entry 10 (class 2615 OID 17097)
-- Name: user_1_file_d8ea9e61; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_d8ea9e61;


ALTER SCHEMA user_1_file_d8ea9e61 OWNER TO postgres;

--
-- TOC entry 16 (class 2615 OID 17969)
-- Name: user_1_file_ebae10c3; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_ebae10c3;


ALTER SCHEMA user_1_file_ebae10c3 OWNER TO postgres;

--
-- TOC entry 15 (class 2615 OID 17968)
-- Name: user_1_file_f2243067; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_f2243067;


ALTER SCHEMA user_1_file_f2243067 OWNER TO postgres;

--
-- TOC entry 37 (class 2615 OID 18021)
-- Name: user_1_file_f31fa970; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA user_1_file_f31fa970;


ALTER SCHEMA user_1_file_f31fa970 OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 257 (class 1259 OID 16413)
-- Name: auth_group; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_group (
    id integer NOT NULL,
    name character varying(150) NOT NULL
);


ALTER TABLE public.auth_group OWNER TO postgres;

--
-- TOC entry 256 (class 1259 OID 16412)
-- Name: auth_group_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_group ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_group_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 259 (class 1259 OID 16421)
-- Name: auth_group_permissions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_group_permissions (
    id bigint NOT NULL,
    group_id integer NOT NULL,
    permission_id integer NOT NULL
);


ALTER TABLE public.auth_group_permissions OWNER TO postgres;

--
-- TOC entry 258 (class 1259 OID 16420)
-- Name: auth_group_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_group_permissions ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_group_permissions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 255 (class 1259 OID 16407)
-- Name: auth_permission; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_permission (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    content_type_id integer NOT NULL,
    codename character varying(100) NOT NULL
);


ALTER TABLE public.auth_permission OWNER TO postgres;

--
-- TOC entry 254 (class 1259 OID 16406)
-- Name: auth_permission_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_permission ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_permission_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 261 (class 1259 OID 16427)
-- Name: auth_user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_user (
    id integer NOT NULL,
    password character varying(128) NOT NULL,
    last_login timestamp with time zone,
    is_superuser boolean NOT NULL,
    username character varying(150) NOT NULL,
    first_name character varying(150) NOT NULL,
    last_name character varying(150) NOT NULL,
    email character varying(254) NOT NULL,
    is_staff boolean NOT NULL,
    is_active boolean NOT NULL,
    date_joined timestamp with time zone NOT NULL
);


ALTER TABLE public.auth_user OWNER TO postgres;

--
-- TOC entry 263 (class 1259 OID 16435)
-- Name: auth_user_groups; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_user_groups (
    id bigint NOT NULL,
    user_id integer NOT NULL,
    group_id integer NOT NULL
);


ALTER TABLE public.auth_user_groups OWNER TO postgres;

--
-- TOC entry 262 (class 1259 OID 16434)
-- Name: auth_user_groups_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_user_groups ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_user_groups_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 260 (class 1259 OID 16426)
-- Name: auth_user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_user ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_user_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 265 (class 1259 OID 16441)
-- Name: auth_user_user_permissions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auth_user_user_permissions (
    id bigint NOT NULL,
    user_id integer NOT NULL,
    permission_id integer NOT NULL
);


ALTER TABLE public.auth_user_user_permissions OWNER TO postgres;

--
-- TOC entry 264 (class 1259 OID 16440)
-- Name: auth_user_user_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.auth_user_user_permissions ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.auth_user_user_permissions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 267 (class 1259 OID 16499)
-- Name: django_admin_log; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.django_admin_log (
    id integer NOT NULL,
    action_time timestamp with time zone NOT NULL,
    object_id text,
    object_repr character varying(200) NOT NULL,
    action_flag smallint NOT NULL,
    change_message text NOT NULL,
    content_type_id integer,
    user_id integer NOT NULL,
    CONSTRAINT django_admin_log_action_flag_check CHECK ((action_flag >= 0))
);


ALTER TABLE public.django_admin_log OWNER TO postgres;

--
-- TOC entry 266 (class 1259 OID 16498)
-- Name: django_admin_log_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.django_admin_log ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.django_admin_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 253 (class 1259 OID 16399)
-- Name: django_content_type; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.django_content_type (
    id integer NOT NULL,
    app_label character varying(100) NOT NULL,
    model character varying(100) NOT NULL
);


ALTER TABLE public.django_content_type OWNER TO postgres;

--
-- TOC entry 252 (class 1259 OID 16398)
-- Name: django_content_type_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.django_content_type ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.django_content_type_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 251 (class 1259 OID 16391)
-- Name: django_migrations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.django_migrations (
    id bigint NOT NULL,
    app character varying(255) NOT NULL,
    name character varying(255) NOT NULL,
    applied timestamp with time zone NOT NULL
);


ALTER TABLE public.django_migrations OWNER TO postgres;

--
-- TOC entry 250 (class 1259 OID 16390)
-- Name: django_migrations_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.django_migrations ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.django_migrations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 268 (class 1259 OID 16527)
-- Name: django_session; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.django_session (
    session_key character varying(40) NOT NULL,
    session_data text NOT NULL,
    expire_date timestamp with time zone NOT NULL
);


ALTER TABLE public.django_session OWNER TO postgres;

--
-- TOC entry 270 (class 1259 OID 16537)
-- Name: ingestion_datasource; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ingestion_datasource (
    id bigint NOT NULL,
    name character varying(120) NOT NULL,
    kind character varying(10) NOT NULL,
    created_at timestamp with time zone NOT NULL,
    internal_schema character varying(120) NOT NULL,
    internal_table character varying(120) NOT NULL,
    owner_id integer NOT NULL
);


ALTER TABLE public.ingestion_datasource OWNER TO postgres;

--
-- TOC entry 269 (class 1259 OID 16536)
-- Name: ingestion_datasource_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.ingestion_datasource ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.ingestion_datasource_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 272 (class 1259 OID 16543)
-- Name: ingestion_externalconnection; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ingestion_externalconnection (
    id bigint NOT NULL,
    db_type character varying(20) NOT NULL,
    host character varying(255) NOT NULL,
    port integer NOT NULL,
    database character varying(255) NOT NULL,
    username bytea NOT NULL,
    password bytea NOT NULL,
    extras jsonb NOT NULL,
    created_at timestamp with time zone NOT NULL,
    source_id bigint NOT NULL
);


ALTER TABLE public.ingestion_externalconnection OWNER TO postgres;

--
-- TOC entry 271 (class 1259 OID 16542)
-- Name: ingestion_externalconnection_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.ingestion_externalconnection ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.ingestion_externalconnection_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 274 (class 1259 OID 16553)
-- Name: ingestion_uploadeddataset; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ingestion_uploadeddataset (
    id bigint NOT NULL,
    file character varying(100) NOT NULL,
    file_type character varying(10) NOT NULL,
    uploaded_at timestamp with time zone NOT NULL,
    rows_ingested integer NOT NULL,
    columns jsonb NOT NULL,
    source_id bigint NOT NULL,
    CONSTRAINT ingestion_uploadeddataset_rows_ingested_check CHECK ((rows_ingested >= 0))
);


ALTER TABLE public.ingestion_uploadeddataset OWNER TO postgres;

--
-- TOC entry 273 (class 1259 OID 16552)
-- Name: ingestion_uploadeddataset_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.ingestion_uploadeddataset ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY (
    SEQUENCE NAME public.ingestion_uploadeddataset_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 277 (class 1259 OID 17892)
-- Name: actividades_tarea; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.actividades_tarea (
    id integer NOT NULL,
    curso_id integer NOT NULL,
    titulo text NOT NULL,
    descripcion text,
    archivo character varying(255) DEFAULT NULL::character varying,
    fecha_entrega date
);


ALTER TABLE user_1_file_00a1f910.actividades_tarea OWNER TO postgres;

--
-- TOC entry 278 (class 1259 OID 17898)
-- Name: administradores; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.administradores (
    id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.administradores OWNER TO postgres;

--
-- TOC entry 279 (class 1259 OID 17901)
-- Name: categoria; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.categoria (
    id integer NOT NULL,
    nombre text NOT NULL
);


ALTER TABLE user_1_file_00a1f910.categoria OWNER TO postgres;

--
-- TOC entry 280 (class 1259 OID 17906)
-- Name: cursos; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.cursos (
    id integer NOT NULL,
    nombre text NOT NULL,
    descripcion text,
    fecha_inicio date NOT NULL,
    fecha_fin date NOT NULL
);


ALTER TABLE user_1_file_00a1f910.cursos OWNER TO postgres;

--
-- TOC entry 281 (class 1259 OID 17911)
-- Name: eventos; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.eventos (
    id integer NOT NULL,
    titulo character varying(100) NOT NULL,
    descripcion text NOT NULL,
    fecha date NOT NULL,
    ministerio_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.eventos OWNER TO postgres;

--
-- TOC entry 282 (class 1259 OID 17916)
-- Name: membresias; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.membresias (
    id integer NOT NULL,
    user_id integer NOT NULL,
    fecha_inicio date NOT NULL,
    fecha_fin date,
    descripcion text
);


ALTER TABLE user_1_file_00a1f910.membresias OWNER TO postgres;

--
-- TOC entry 284 (class 1259 OID 17924)
-- Name: miembro_curso; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.miembro_curso (
    id integer NOT NULL,
    user_id integer NOT NULL,
    curso_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.miembro_curso OWNER TO postgres;

--
-- TOC entry 285 (class 1259 OID 17927)
-- Name: miembro_curso_actividad; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.miembro_curso_actividad (
    id integer NOT NULL,
    miembro_curso_id integer NOT NULL,
    actividad_tarea_id integer NOT NULL,
    nota integer,
    archivo character varying(255) DEFAULT NULL::character varying
);


ALTER TABLE user_1_file_00a1f910.miembro_curso_actividad OWNER TO postgres;

--
-- TOC entry 283 (class 1259 OID 17921)
-- Name: miembros; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.miembros (
    id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.miembros OWNER TO postgres;

--
-- TOC entry 286 (class 1259 OID 17931)
-- Name: ministerio; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.ministerio (
    id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.ministerio OWNER TO postgres;

--
-- TOC entry 287 (class 1259 OID 17934)
-- Name: ministerio_user; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.ministerio_user (
    id integer NOT NULL,
    user_id integer NOT NULL,
    ministerio_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.ministerio_user OWNER TO postgres;

--
-- TOC entry 288 (class 1259 OID 17937)
-- Name: model_has_role; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.model_has_role (
    user_id integer NOT NULL,
    role_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.model_has_role OWNER TO postgres;

--
-- TOC entry 289 (class 1259 OID 17940)
-- Name: permissions; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.permissions (
    id integer NOT NULL,
    name character varying(50) NOT NULL
);


ALTER TABLE user_1_file_00a1f910.permissions OWNER TO postgres;

--
-- TOC entry 291 (class 1259 OID 17946)
-- Name: role_has_permission; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.role_has_permission (
    role_id integer NOT NULL,
    permission_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.role_has_permission OWNER TO postgres;

--
-- TOC entry 290 (class 1259 OID 17943)
-- Name: roles; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.roles (
    id integer NOT NULL,
    name character varying(50) NOT NULL
);


ALTER TABLE user_1_file_00a1f910.roles OWNER TO postgres;

--
-- TOC entry 293 (class 1259 OID 17954)
-- Name: sacramento_user; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.sacramento_user (
    id integer NOT NULL,
    sacramento_id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.sacramento_user OWNER TO postgres;

--
-- TOC entry 292 (class 1259 OID 17949)
-- Name: sacramentos; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.sacramentos (
    id integer NOT NULL,
    fecha date NOT NULL,
    lugar text NOT NULL,
    ministro text NOT NULL,
    categoria_id integer NOT NULL
);


ALTER TABLE user_1_file_00a1f910.sacramentos OWNER TO postgres;

--
-- TOC entry 294 (class 1259 OID 17957)
-- Name: users; Type: TABLE; Schema: user_1_file_00a1f910; Owner: postgres
--

CREATE TABLE user_1_file_00a1f910.users (
    id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    password character varying(255) NOT NULL,
    email character varying(100) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    direccion character varying(255) DEFAULT NULL::character varying,
    telefono bigint,
    ci integer,
    foto character varying(255) DEFAULT NULL::character varying,
    fecha_nacimiento date,
    genero character varying(20) DEFAULT NULL::character varying
);


ALTER TABLE user_1_file_00a1f910.users OWNER TO postgres;

--
-- TOC entry 275 (class 1259 OID 17082)
-- Name: actividades_tarea; Type: TABLE; Schema: user_1_file_7639c196; Owner: postgres
--

CREATE TABLE user_1_file_7639c196.actividades_tarea (
    id integer NOT NULL,
    curso_id integer NOT NULL,
    titulo text NOT NULL,
    descripcion text,
    archivo character varying(255) DEFAULT NULL::character varying,
    fecha_entrega date
);


ALTER TABLE user_1_file_7639c196.actividades_tarea OWNER TO postgres;

--
-- TOC entry 296 (class 1259 OID 17978)
-- Name: departamentos; Type: TABLE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE TABLE user_1_file_b39d309b.departamentos (
    id integer NOT NULL,
    nombre character varying(50),
    ubicacion character varying(50)
);


ALTER TABLE user_1_file_b39d309b.departamentos OWNER TO postgres;

--
-- TOC entry 295 (class 1259 OID 17977)
-- Name: departamentos_id_seq; Type: SEQUENCE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE SEQUENCE user_1_file_b39d309b.departamentos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE user_1_file_b39d309b.departamentos_id_seq OWNER TO postgres;

--
-- TOC entry 5317 (class 0 OID 0)
-- Dependencies: 295
-- Name: departamentos_id_seq; Type: SEQUENCE OWNED BY; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER SEQUENCE user_1_file_b39d309b.departamentos_id_seq OWNED BY user_1_file_b39d309b.departamentos.id;


--
-- TOC entry 298 (class 1259 OID 17985)
-- Name: empleados; Type: TABLE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE TABLE user_1_file_b39d309b.empleados (
    id integer NOT NULL,
    nombre character varying(50),
    puesto character varying(50),
    salario numeric(10,2),
    departamento_id integer
);


ALTER TABLE user_1_file_b39d309b.empleados OWNER TO postgres;

--
-- TOC entry 297 (class 1259 OID 17984)
-- Name: empleados_id_seq; Type: SEQUENCE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE SEQUENCE user_1_file_b39d309b.empleados_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE user_1_file_b39d309b.empleados_id_seq OWNER TO postgres;

--
-- TOC entry 5318 (class 0 OID 0)
-- Dependencies: 297
-- Name: empleados_id_seq; Type: SEQUENCE OWNED BY; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER SEQUENCE user_1_file_b39d309b.empleados_id_seq OWNED BY user_1_file_b39d309b.empleados.id;


--
-- TOC entry 300 (class 1259 OID 17997)
-- Name: proyectos; Type: TABLE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE TABLE user_1_file_b39d309b.proyectos (
    id integer NOT NULL,
    nombre character varying(50),
    presupuesto numeric(10,2),
    departamento_id integer
);


ALTER TABLE user_1_file_b39d309b.proyectos OWNER TO postgres;

--
-- TOC entry 299 (class 1259 OID 17996)
-- Name: proyectos_id_seq; Type: SEQUENCE; Schema: user_1_file_b39d309b; Owner: postgres
--

CREATE SEQUENCE user_1_file_b39d309b.proyectos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE user_1_file_b39d309b.proyectos_id_seq OWNER TO postgres;

--
-- TOC entry 5319 (class 0 OID 0)
-- Dependencies: 299
-- Name: proyectos_id_seq; Type: SEQUENCE OWNED BY; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER SEQUENCE user_1_file_b39d309b.proyectos_id_seq OWNED BY user_1_file_b39d309b.proyectos.id;


--
-- TOC entry 276 (class 1259 OID 17089)
-- Name: actividades_tarea; Type: TABLE; Schema: user_1_file_d4a80680; Owner: postgres
--

CREATE TABLE user_1_file_d4a80680.actividades_tarea (
    id integer NOT NULL,
    curso_id integer NOT NULL,
    titulo text NOT NULL,
    descripcion text,
    archivo character varying(255) DEFAULT NULL::character varying,
    fecha_entrega date
);


ALTER TABLE user_1_file_d4a80680.actividades_tarea OWNER TO postgres;

--
-- TOC entry 301 (class 1259 OID 18022)
-- Name: actividades_tarea; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.actividades_tarea (
    id integer NOT NULL,
    curso_id integer NOT NULL,
    titulo text NOT NULL,
    descripcion text,
    archivo character varying(255) DEFAULT NULL::character varying,
    fecha_entrega date
);


ALTER TABLE user_1_file_f31fa970.actividades_tarea OWNER TO postgres;

--
-- TOC entry 302 (class 1259 OID 18028)
-- Name: administradores; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.administradores (
    id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.administradores OWNER TO postgres;

--
-- TOC entry 303 (class 1259 OID 18031)
-- Name: categoria; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.categoria (
    id integer NOT NULL,
    nombre text NOT NULL
);


ALTER TABLE user_1_file_f31fa970.categoria OWNER TO postgres;

--
-- TOC entry 304 (class 1259 OID 18036)
-- Name: cursos; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.cursos (
    id integer NOT NULL,
    nombre text NOT NULL,
    descripcion text,
    fecha_inicio date NOT NULL,
    fecha_fin date NOT NULL
);


ALTER TABLE user_1_file_f31fa970.cursos OWNER TO postgres;

--
-- TOC entry 305 (class 1259 OID 18041)
-- Name: eventos; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.eventos (
    id integer NOT NULL,
    titulo character varying(100) NOT NULL,
    descripcion text NOT NULL,
    fecha date NOT NULL,
    ministerio_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.eventos OWNER TO postgres;

--
-- TOC entry 306 (class 1259 OID 18046)
-- Name: membresias; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.membresias (
    id integer NOT NULL,
    user_id integer NOT NULL,
    fecha_inicio date NOT NULL,
    fecha_fin date,
    descripcion text
);


ALTER TABLE user_1_file_f31fa970.membresias OWNER TO postgres;

--
-- TOC entry 308 (class 1259 OID 18054)
-- Name: miembro_curso; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.miembro_curso (
    id integer NOT NULL,
    user_id integer NOT NULL,
    curso_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.miembro_curso OWNER TO postgres;

--
-- TOC entry 309 (class 1259 OID 18057)
-- Name: miembro_curso_actividad; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.miembro_curso_actividad (
    id integer NOT NULL,
    miembro_curso_id integer NOT NULL,
    actividad_tarea_id integer NOT NULL,
    nota integer,
    archivo character varying(255) DEFAULT NULL::character varying
);


ALTER TABLE user_1_file_f31fa970.miembro_curso_actividad OWNER TO postgres;

--
-- TOC entry 307 (class 1259 OID 18051)
-- Name: miembros; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.miembros (
    id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.miembros OWNER TO postgres;

--
-- TOC entry 310 (class 1259 OID 18061)
-- Name: ministerio; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.ministerio (
    id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.ministerio OWNER TO postgres;

--
-- TOC entry 311 (class 1259 OID 18064)
-- Name: ministerio_user; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.ministerio_user (
    id integer NOT NULL,
    user_id integer NOT NULL,
    ministerio_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.ministerio_user OWNER TO postgres;

--
-- TOC entry 312 (class 1259 OID 18067)
-- Name: model_has_role; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.model_has_role (
    user_id integer NOT NULL,
    role_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.model_has_role OWNER TO postgres;

--
-- TOC entry 313 (class 1259 OID 18070)
-- Name: permissions; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.permissions (
    id integer NOT NULL,
    name character varying(50) NOT NULL
);


ALTER TABLE user_1_file_f31fa970.permissions OWNER TO postgres;

--
-- TOC entry 315 (class 1259 OID 18076)
-- Name: role_has_permission; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.role_has_permission (
    role_id integer NOT NULL,
    permission_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.role_has_permission OWNER TO postgres;

--
-- TOC entry 314 (class 1259 OID 18073)
-- Name: roles; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.roles (
    id integer NOT NULL,
    name character varying(50) NOT NULL
);


ALTER TABLE user_1_file_f31fa970.roles OWNER TO postgres;

--
-- TOC entry 317 (class 1259 OID 18084)
-- Name: sacramento_user; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.sacramento_user (
    id integer NOT NULL,
    sacramento_id integer NOT NULL,
    user_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.sacramento_user OWNER TO postgres;

--
-- TOC entry 316 (class 1259 OID 18079)
-- Name: sacramentos; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.sacramentos (
    id integer NOT NULL,
    fecha date NOT NULL,
    lugar text NOT NULL,
    ministro text NOT NULL,
    categoria_id integer NOT NULL
);


ALTER TABLE user_1_file_f31fa970.sacramentos OWNER TO postgres;

--
-- TOC entry 318 (class 1259 OID 18087)
-- Name: users; Type: TABLE; Schema: user_1_file_f31fa970; Owner: postgres
--

CREATE TABLE user_1_file_f31fa970.users (
    id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    password character varying(255) NOT NULL,
    email character varying(100) NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
    direccion character varying(255) DEFAULT NULL::character varying,
    telefono bigint,
    ci integer,
    foto character varying(255) DEFAULT NULL::character varying,
    fecha_nacimiento date,
    genero character varying(20) DEFAULT NULL::character varying
);


ALTER TABLE user_1_file_f31fa970.users OWNER TO postgres;

--
-- TOC entry 5009 (class 2604 OID 17981)
-- Name: departamentos id; Type: DEFAULT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.departamentos ALTER COLUMN id SET DEFAULT nextval('user_1_file_b39d309b.departamentos_id_seq'::regclass);


--
-- TOC entry 5010 (class 2604 OID 17988)
-- Name: empleados id; Type: DEFAULT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.empleados ALTER COLUMN id SET DEFAULT nextval('user_1_file_b39d309b.empleados_id_seq'::regclass);


--
-- TOC entry 5011 (class 2604 OID 18000)
-- Name: proyectos id; Type: DEFAULT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.proyectos ALTER COLUMN id SET DEFAULT nextval('user_1_file_b39d309b.proyectos_id_seq'::regclass);


--
-- TOC entry 5250 (class 0 OID 16413)
-- Dependencies: 257
-- Data for Name: auth_group; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5252 (class 0 OID 16421)
-- Dependencies: 259
-- Data for Name: auth_group_permissions; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5248 (class 0 OID 16407)
-- Dependencies: 255
-- Data for Name: auth_permission; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.auth_permission VALUES (1, 'Can add log entry', 1, 'add_logentry');
INSERT INTO public.auth_permission VALUES (2, 'Can change log entry', 1, 'change_logentry');
INSERT INTO public.auth_permission VALUES (3, 'Can delete log entry', 1, 'delete_logentry');
INSERT INTO public.auth_permission VALUES (4, 'Can view log entry', 1, 'view_logentry');
INSERT INTO public.auth_permission VALUES (5, 'Can add permission', 2, 'add_permission');
INSERT INTO public.auth_permission VALUES (6, 'Can change permission', 2, 'change_permission');
INSERT INTO public.auth_permission VALUES (7, 'Can delete permission', 2, 'delete_permission');
INSERT INTO public.auth_permission VALUES (8, 'Can view permission', 2, 'view_permission');
INSERT INTO public.auth_permission VALUES (9, 'Can add group', 3, 'add_group');
INSERT INTO public.auth_permission VALUES (10, 'Can change group', 3, 'change_group');
INSERT INTO public.auth_permission VALUES (11, 'Can delete group', 3, 'delete_group');
INSERT INTO public.auth_permission VALUES (12, 'Can view group', 3, 'view_group');
INSERT INTO public.auth_permission VALUES (13, 'Can add user', 4, 'add_user');
INSERT INTO public.auth_permission VALUES (14, 'Can change user', 4, 'change_user');
INSERT INTO public.auth_permission VALUES (15, 'Can delete user', 4, 'delete_user');
INSERT INTO public.auth_permission VALUES (16, 'Can view user', 4, 'view_user');
INSERT INTO public.auth_permission VALUES (17, 'Can add content type', 5, 'add_contenttype');
INSERT INTO public.auth_permission VALUES (18, 'Can change content type', 5, 'change_contenttype');
INSERT INTO public.auth_permission VALUES (19, 'Can delete content type', 5, 'delete_contenttype');
INSERT INTO public.auth_permission VALUES (20, 'Can view content type', 5, 'view_contenttype');
INSERT INTO public.auth_permission VALUES (21, 'Can add session', 6, 'add_session');
INSERT INTO public.auth_permission VALUES (22, 'Can change session', 6, 'change_session');
INSERT INTO public.auth_permission VALUES (23, 'Can delete session', 6, 'delete_session');
INSERT INTO public.auth_permission VALUES (24, 'Can view session', 6, 'view_session');
INSERT INTO public.auth_permission VALUES (25, 'Can add uploaded dataset', 7, 'add_uploadeddataset');
INSERT INTO public.auth_permission VALUES (26, 'Can change uploaded dataset', 7, 'change_uploadeddataset');
INSERT INTO public.auth_permission VALUES (27, 'Can delete uploaded dataset', 7, 'delete_uploadeddataset');
INSERT INTO public.auth_permission VALUES (28, 'Can view uploaded dataset', 7, 'view_uploadeddataset');
INSERT INTO public.auth_permission VALUES (29, 'Can add data source', 8, 'add_datasource');
INSERT INTO public.auth_permission VALUES (30, 'Can change data source', 8, 'change_datasource');
INSERT INTO public.auth_permission VALUES (31, 'Can delete data source', 8, 'delete_datasource');
INSERT INTO public.auth_permission VALUES (32, 'Can view data source', 8, 'view_datasource');
INSERT INTO public.auth_permission VALUES (33, 'Can add external connection', 9, 'add_externalconnection');
INSERT INTO public.auth_permission VALUES (34, 'Can change external connection', 9, 'change_externalconnection');
INSERT INTO public.auth_permission VALUES (35, 'Can delete external connection', 9, 'delete_externalconnection');
INSERT INTO public.auth_permission VALUES (36, 'Can view external connection', 9, 'view_externalconnection');


--
-- TOC entry 5254 (class 0 OID 16427)
-- Dependencies: 261
-- Data for Name: auth_user; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.auth_user VALUES (1, 'pbkdf2_sha256$1000000$ctrbL0A44QdA8fyq9ZV6To$WzP1e9yvdxX4lI2c66b/nCz94CZdnsWR1UGR7bAoBIQ=', '2025-09-10 21:57:11.776806-04', true, 'admin', '', '', 'admin@gmail.com', true, true, '2025-09-10 21:56:22.177035-04');


--
-- TOC entry 5256 (class 0 OID 16435)
-- Dependencies: 263
-- Data for Name: auth_user_groups; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5258 (class 0 OID 16441)
-- Dependencies: 265
-- Data for Name: auth_user_user_permissions; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5260 (class 0 OID 16499)
-- Dependencies: 267
-- Data for Name: django_admin_log; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5246 (class 0 OID 16399)
-- Dependencies: 253
-- Data for Name: django_content_type; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.django_content_type VALUES (1, 'admin', 'logentry');
INSERT INTO public.django_content_type VALUES (2, 'auth', 'permission');
INSERT INTO public.django_content_type VALUES (3, 'auth', 'group');
INSERT INTO public.django_content_type VALUES (4, 'auth', 'user');
INSERT INTO public.django_content_type VALUES (5, 'contenttypes', 'contenttype');
INSERT INTO public.django_content_type VALUES (6, 'sessions', 'session');
INSERT INTO public.django_content_type VALUES (7, 'ingestion', 'uploadeddataset');
INSERT INTO public.django_content_type VALUES (8, 'ingestion', 'datasource');
INSERT INTO public.django_content_type VALUES (9, 'ingestion', 'externalconnection');


--
-- TOC entry 5244 (class 0 OID 16391)
-- Dependencies: 251
-- Data for Name: django_migrations; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.django_migrations VALUES (1, 'contenttypes', '0001_initial', '2025-09-10 21:46:59.017171-04');
INSERT INTO public.django_migrations VALUES (2, 'auth', '0001_initial', '2025-09-10 21:46:59.055302-04');
INSERT INTO public.django_migrations VALUES (3, 'admin', '0001_initial', '2025-09-10 21:46:59.063649-04');
INSERT INTO public.django_migrations VALUES (4, 'admin', '0002_logentry_remove_auto_add', '2025-09-10 21:46:59.067158-04');
INSERT INTO public.django_migrations VALUES (5, 'admin', '0003_logentry_add_action_flag_choices', '2025-09-10 21:46:59.076667-04');
INSERT INTO public.django_migrations VALUES (6, 'contenttypes', '0002_remove_content_type_name', '2025-09-10 21:46:59.081863-04');
INSERT INTO public.django_migrations VALUES (7, 'auth', '0002_alter_permission_name_max_length', '2025-09-10 21:46:59.08537-04');
INSERT INTO public.django_migrations VALUES (8, 'auth', '0003_alter_user_email_max_length', '2025-09-10 21:46:59.088854-04');
INSERT INTO public.django_migrations VALUES (9, 'auth', '0004_alter_user_username_opts', '2025-09-10 21:46:59.091852-04');
INSERT INTO public.django_migrations VALUES (10, 'auth', '0005_alter_user_last_login_null', '2025-09-10 21:46:59.094361-04');
INSERT INTO public.django_migrations VALUES (11, 'auth', '0006_require_contenttypes_0002', '2025-09-10 21:46:59.095361-04');
INSERT INTO public.django_migrations VALUES (12, 'auth', '0007_alter_validators_add_error_messages', '2025-09-10 21:46:59.097865-04');
INSERT INTO public.django_migrations VALUES (13, 'auth', '0008_alter_user_username_max_length', '2025-09-10 21:46:59.105401-04');
INSERT INTO public.django_migrations VALUES (14, 'auth', '0009_alter_user_last_name_max_length', '2025-09-10 21:46:59.109087-04');
INSERT INTO public.django_migrations VALUES (15, 'auth', '0010_alter_group_name_max_length', '2025-09-10 21:46:59.112091-04');
INSERT INTO public.django_migrations VALUES (16, 'auth', '0011_update_proxy_permissions', '2025-09-10 21:46:59.114597-04');
INSERT INTO public.django_migrations VALUES (17, 'auth', '0012_alter_user_first_name_max_length', '2025-09-10 21:46:59.117102-04');
INSERT INTO public.django_migrations VALUES (18, 'sessions', '0001_initial', '2025-09-10 21:46:59.121611-04');
INSERT INTO public.django_migrations VALUES (19, 'ingestion', '0001_initial', '2025-09-14 12:45:34.5293-04');


--
-- TOC entry 5261 (class 0 OID 16527)
-- Dependencies: 268
-- Data for Name: django_session; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.django_session VALUES ('wfi7dr0hstpizsb1eom6xxzval2rq3jy', '.eJxVjMsOwiAQRf-FtSE8hgIu3fsNZIBBqgaS0q6M_65NutDtPefcFwu4rTVsg5YwZ3Zmkp1-t4jpQW0H-Y7t1nnqbV3myHeFH3Twa8_0vBzu30HFUb81kNU4YQYjhNMiSvKAJlshpQUqEZ1xCkqyOIFMCp3PRSWvRNTWm0Ts_QHRwTeb:1uwWYh:_teXck_VLieu3SSxLh5PgDiNFxMEfuSkXvk5-KCtH9U', '2025-09-24 21:57:11.777806-04');


--
-- TOC entry 5263 (class 0 OID 16537)
-- Dependencies: 270
-- Data for Name: ingestion_datasource; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.ingestion_datasource VALUES (1, 'AAA', 'FILE', '2025-09-14 16:03:16.680127-04', 'user_1_file_7639c196', 'actividades_tarea', 1);
INSERT INTO public.ingestion_datasource VALUES (2, 'ffff', 'FILE', '2025-09-14 16:03:46.897199-04', 'user_1_file_d4a80680', 'actividades_tarea', 1);
INSERT INTO public.ingestion_datasource VALUES (3, 'ffff', 'FILE', '2025-09-14 16:05:32.375999-04', 'user_1_file_d803d64d', '', 1);
INSERT INTO public.ingestion_datasource VALUES (4, 'ffffAAAA', 'FILE', '2025-09-14 16:07:21.380337-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (5, 'ffffAAAA', 'FILE', '2025-09-14 16:08:23.402539-04', 'user_1_file_d8ea9e61', '', 1);
INSERT INTO public.ingestion_datasource VALUES (6, 'DDDDDDDDDDDDDDD', 'FILE', '2025-09-14 16:11:22.856031-04', 'user_1_file_d0663889', '', 1);
INSERT INTO public.ingestion_datasource VALUES (7, 'DSFDSFDSF', 'FILE', '2025-09-14 16:11:40.582355-04', 'user_1_file_7fad03c9', '', 1);
INSERT INTO public.ingestion_datasource VALUES (8, 'ABER', 'FILE', '2025-09-14 16:16:18.470692-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (9, 'ABER', 'FILE', '2025-09-14 16:17:58.118716-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (10, 'ABER', 'FILE', '2025-09-14 16:19:10.028586-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (11, 'ABER', 'FILE', '2025-09-14 16:20:40.466324-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (12, 'ABER', 'FILE', '2025-09-14 16:20:42.641621-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (13, 'ABER', 'FILE', '2025-09-14 16:23:33.348759-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (14, 'ABERff', 'FILE', '2025-09-14 16:23:36.308442-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (15, 'ABERff', 'FILE', '2025-09-14 16:25:01.870707-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (16, 'ABERff', 'FILE', '2025-09-14 16:25:03.797483-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (17, 'ABERff', 'FILE', '2025-09-14 16:25:21.881749-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (18, 'ABERff', 'FILE', '2025-09-14 16:26:15.989789-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (19, 'ABERff', 'FILE', '2025-09-14 16:28:38.332387-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (20, 'A', 'FILE', '2025-09-14 16:37:46.718587-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (21, 'A', 'FILE', '2025-09-14 16:40:20.244446-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (22, 'A', 'FILE', '2025-09-14 16:42:10.290128-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (23, 'A', 'FILE', '2025-09-14 16:42:12.257245-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (24, 'A', 'FILE', '2025-09-14 16:42:54.435374-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (25, 'A', 'FILE', '2025-09-14 16:43:54.050647-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (26, 'A', 'FILE', '2025-09-14 16:43:56.684658-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (27, 'A', 'FILE', '2025-09-14 16:44:14.367655-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (28, 'A', 'FILE', '2025-09-14 16:45:00.953243-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (29, 'DASDSDSDSDSDSD', 'FILE', '2025-09-14 16:46:47.779836-04', 'user_1_file_b39d309b', 'departamentos', 1);
INSERT INTO public.ingestion_datasource VALUES (30, 'MYSQL', 'FILE', '2025-09-14 16:53:11.106242-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (31, 'MYSQL', 'FILE', '2025-09-14 16:56:43.010273-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (32, 'MYSQL', 'FILE', '2025-09-14 16:56:48.298477-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (33, 'MYSQL', 'FILE', '2025-09-14 16:57:28.785587-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (34, 'MYSQL', 'FILE', '2025-09-14 16:57:31.65956-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (35, 'MYSQL', 'FILE', '2025-09-14 17:00:35.036136-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (36, 'MYSQL', 'FILE', '2025-09-14 17:01:17.683735-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (37, 'MYSQL', 'FILE', '2025-09-14 17:03:44.524775-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (38, 'MYSQL', 'FILE', '2025-09-14 17:03:53.869375-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (39, 'FFFFFSADS', 'FILE', '2025-09-14 17:04:04.047768-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (40, 'FFFFFSADS', 'FILE', '2025-09-14 17:05:11.498619-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (41, 'FFFFFSADS', 'FILE', '2025-09-14 17:05:24.539286-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (42, 'FFFFFSADS', 'FILE', '2025-09-14 17:05:26.877243-04', '', '', 1);
INSERT INTO public.ingestion_datasource VALUES (43, 'FFFFFSADS', 'FILE', '2025-09-14 17:07:37.492557-04', 'user_1_file_f31fa970', 'actividades_tarea', 1);
INSERT INTO public.ingestion_datasource VALUES (44, 'Prueaaa', 'FILE', '2025-09-14 17:26:09.610317-04', 'user_1_file_ad5c2d2d', '', 1);


--
-- TOC entry 5265 (class 0 OID 16543)
-- Dependencies: 272
-- Data for Name: ingestion_externalconnection; Type: TABLE DATA; Schema: public; Owner: postgres
--



--
-- TOC entry 5267 (class 0 OID 16553)
-- Dependencies: 274
-- Data for Name: ingestion_uploadeddataset; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.ingestion_uploadeddataset VALUES (1, 'uploads/iglesiaXd_eL2zmCc.sql', 'sql', '2025-09-14 16:03:16.682632-04', 0, '["id", "curso_id", "titulo", "descripcion", "archivo", "fecha_entrega"]', 1);
INSERT INTO public.ingestion_uploadeddataset VALUES (2, 'uploads/iglesiaXd_bn71LNL.sql', 'sql', '2025-09-14 16:03:46.899199-04', 0, '["id", "curso_id", "titulo", "descripcion", "archivo", "fecha_entrega"]', 2);
INSERT INTO public.ingestion_uploadeddataset VALUES (3, 'uploads/iglesiaXd_SVmRpYt.sql', 'sql', '2025-09-14 16:05:32.376999-04', 0, '[]', 3);
INSERT INTO public.ingestion_uploadeddataset VALUES (4, 'uploads/iglesiaXd_uqfMCHp.sql', 'sql', '2025-09-14 16:07:21.382341-04', 0, '[]', 4);
INSERT INTO public.ingestion_uploadeddataset VALUES (5, 'uploads/iglesiaXd_7tVugYi.sql', 'sql', '2025-09-14 16:08:23.40454-04', 0, '[]', 5);
INSERT INTO public.ingestion_uploadeddataset VALUES (6, 'uploads/iglesiaXd_z111vMi.sql', 'sql', '2025-09-14 16:11:22.859536-04', 0, '[]', 6);
INSERT INTO public.ingestion_uploadeddataset VALUES (7, 'uploads/iglesiaXd_TrcdRGh.sql', 'sql', '2025-09-14 16:11:40.585355-04', 0, '[]', 7);
INSERT INTO public.ingestion_uploadeddataset VALUES (8, 'uploads/iglesiaXd_L8lBTt8.sql', 'sql', '2025-09-14 16:16:18.472692-04', 0, '[]', 8);
INSERT INTO public.ingestion_uploadeddataset VALUES (9, 'uploads/iglesiaXd_knB6lV0.sql', 'sql', '2025-09-14 16:17:58.121715-04', 0, '[]', 9);
INSERT INTO public.ingestion_uploadeddataset VALUES (10, 'uploads/iglesiaXd_Uo48JVS.sql', 'sql', '2025-09-14 16:19:10.031586-04', 0, '[]', 10);
INSERT INTO public.ingestion_uploadeddataset VALUES (11, 'uploads/iglesiaXd_F4mZ5P5.sql', 'sql', '2025-09-14 16:20:40.468414-04', 0, '[]', 11);
INSERT INTO public.ingestion_uploadeddataset VALUES (12, 'uploads/iglesiaXd_1B8iJSA.sql', 'sql', '2025-09-14 16:20:42.642621-04', 0, '[]', 12);
INSERT INTO public.ingestion_uploadeddataset VALUES (13, 'uploads/iglesiaXd_pAyE4j3.sql', 'sql', '2025-09-14 16:23:33.353778-04', 0, '[]', 13);
INSERT INTO public.ingestion_uploadeddataset VALUES (14, 'uploads/iglesiaXd_D6L5kCe.sql', 'sql', '2025-09-14 16:23:36.327068-04', 0, '[]', 14);
INSERT INTO public.ingestion_uploadeddataset VALUES (15, 'uploads/iglesiaXd_yK6vtxL.sql', 'sql', '2025-09-14 16:25:01.873731-04', 0, '[]', 15);
INSERT INTO public.ingestion_uploadeddataset VALUES (16, 'uploads/iglesiaXd_Hwxs3eq.sql', 'sql', '2025-09-14 16:25:03.799483-04', 0, '[]', 16);
INSERT INTO public.ingestion_uploadeddataset VALUES (17, 'uploads/iglesiaXd_vtUalRg.sql', 'sql', '2025-09-14 16:25:21.884769-04', 0, '[]', 17);
INSERT INTO public.ingestion_uploadeddataset VALUES (18, 'uploads/iglesiaXd_dDrpmdG.sql', 'sql', '2025-09-14 16:26:15.991789-04', 0, '[]', 18);
INSERT INTO public.ingestion_uploadeddataset VALUES (19, 'uploads/iglesiaXd_0XuCTlD.sql', 'sql', '2025-09-14 16:28:38.334388-04', 0, '[]', 19);
INSERT INTO public.ingestion_uploadeddataset VALUES (20, 'uploads/empresa_k37lU4w.sql', 'sql', '2025-09-14 16:37:46.721176-04', 0, '[]', 20);
INSERT INTO public.ingestion_uploadeddataset VALUES (21, 'uploads/empresa_lRkQQgn.sql', 'sql', '2025-09-14 16:40:20.245445-04', 0, '[]', 21);
INSERT INTO public.ingestion_uploadeddataset VALUES (22, 'uploads/empresa_JE8DItq.sql', 'sql', '2025-09-14 16:42:10.293131-04', 0, '[]', 22);
INSERT INTO public.ingestion_uploadeddataset VALUES (23, 'uploads/empresa_UAfdJVo.sql', 'sql', '2025-09-14 16:42:12.260273-04', 0, '[]', 23);
INSERT INTO public.ingestion_uploadeddataset VALUES (24, 'uploads/empresa_buSaqFK.sql', 'sql', '2025-09-14 16:42:54.437374-04', 0, '[]', 24);
INSERT INTO public.ingestion_uploadeddataset VALUES (25, 'uploads/empresa_rDKaVNY.sql', 'sql', '2025-09-14 16:43:54.053646-04', 0, '[]', 25);
INSERT INTO public.ingestion_uploadeddataset VALUES (26, 'uploads/empresa_4VWYGSX.sql', 'sql', '2025-09-14 16:43:56.686657-04', 0, '[]', 26);
INSERT INTO public.ingestion_uploadeddataset VALUES (27, 'uploads/empresa_YaCEjmT.sql', 'sql', '2025-09-14 16:44:14.370655-04', 0, '[]', 27);
INSERT INTO public.ingestion_uploadeddataset VALUES (28, 'uploads/empresa_KhvefTF.sql', 'sql', '2025-09-14 16:45:00.95525-04', 0, '[]', 28);
INSERT INTO public.ingestion_uploadeddataset VALUES (29, 'uploads/empresa_Up3czBC.sql', 'sql', '2025-09-14 16:46:47.781836-04', 3, '["id", "nombre", "ubicacion"]', 29);
INSERT INTO public.ingestion_uploadeddataset VALUES (30, 'uploads/iglesiaXd_MWqvre7.sql', 'sql', '2025-09-14 16:53:11.108386-04', 0, '[]', 30);
INSERT INTO public.ingestion_uploadeddataset VALUES (31, 'uploads/iglesiaXd_4JhxEVR.sql', 'sql', '2025-09-14 16:56:43.012273-04', 0, '[]', 31);
INSERT INTO public.ingestion_uploadeddataset VALUES (32, 'uploads/iglesiaXd_Kmnrjqn.sql', 'sql', '2025-09-14 16:56:48.301475-04', 0, '[]', 32);
INSERT INTO public.ingestion_uploadeddataset VALUES (33, 'uploads/iglesiaXd_K4XHjkV.sql', 'sql', '2025-09-14 16:57:28.787783-04', 0, '[]', 33);
INSERT INTO public.ingestion_uploadeddataset VALUES (34, 'uploads/iglesiaXd_GzTYVDL.sql', 'sql', '2025-09-14 16:57:31.661559-04', 0, '[]', 34);
INSERT INTO public.ingestion_uploadeddataset VALUES (35, 'uploads/iglesiaXd_YIYpGgK.sql', 'sql', '2025-09-14 17:00:35.03764-04', 0, '[]', 35);
INSERT INTO public.ingestion_uploadeddataset VALUES (36, 'uploads/iglesiaXd_0mqoPHn.sql', 'sql', '2025-09-14 17:01:17.68724-04', 0, '[]', 36);
INSERT INTO public.ingestion_uploadeddataset VALUES (37, 'uploads/iglesiaXd_N6kXzD2.sql', 'sql', '2025-09-14 17:03:44.527899-04', 0, '[]', 37);
INSERT INTO public.ingestion_uploadeddataset VALUES (38, 'uploads/iglesiaXd_6a7hyed.sql', 'sql', '2025-09-14 17:03:53.871375-04', 0, '[]', 38);
INSERT INTO public.ingestion_uploadeddataset VALUES (39, 'uploads/iglesiaXd_CkY08SC.sql', 'sql', '2025-09-14 17:04:04.049768-04', 0, '[]', 39);
INSERT INTO public.ingestion_uploadeddataset VALUES (40, 'uploads/iglesiaXd_L8kiRiO.sql', 'sql', '2025-09-14 17:05:11.500619-04', 0, '[]', 40);
INSERT INTO public.ingestion_uploadeddataset VALUES (41, 'uploads/iglesiaXd_iCjaxx0.sql', 'sql', '2025-09-14 17:05:24.542798-04', 0, '[]', 41);
INSERT INTO public.ingestion_uploadeddataset VALUES (42, 'uploads/iglesiaXd_swztSmj.sql', 'sql', '2025-09-14 17:05:26.878243-04', 0, '[]', 42);
INSERT INTO public.ingestion_uploadeddataset VALUES (43, 'uploads/iglesiaXd_wpD4Df2.sql', 'sql', '2025-09-14 17:07:37.494557-04', 2, '["id", "curso_id", "titulo", "descripcion", "archivo", "fecha_entrega"]', 43);
INSERT INTO public.ingestion_uploadeddataset VALUES (44, 'uploads/PruebadePostgre.sql', 'sql', '2025-09-14 17:26:09.612654-04', 0, '[]', 44);


--
-- TOC entry 5270 (class 0 OID 17892)
-- Dependencies: 277
-- Data for Name: actividades_tarea; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.actividades_tarea VALUES (1, 2, 'ddd', 'dddd', '433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-22');
INSERT INTO user_1_file_00a1f910.actividades_tarea VALUES (2, 2, 'aa', 'assssdasdsad', 'Conectar el DHCP de un router a un switch a maquinas.docx', '2025-03-28');
INSERT INTO user_1_file_00a1f910.actividades_tarea VALUES (1, 2, 'ddd', 'dddd', '433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-22');
INSERT INTO user_1_file_00a1f910.actividades_tarea VALUES (2, 2, 'aa', 'assssdasdsad', 'Conectar el DHCP de un router a un switch a maquinas.docx', '2025-03-28');


--
-- TOC entry 5271 (class 0 OID 17898)
-- Dependencies: 278
-- Data for Name: administradores; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.administradores VALUES (2, 3);
INSERT INTO user_1_file_00a1f910.administradores VALUES (3, 4);
INSERT INTO user_1_file_00a1f910.administradores VALUES (5, 7);
INSERT INTO user_1_file_00a1f910.administradores VALUES (7, 14);
INSERT INTO user_1_file_00a1f910.administradores VALUES (11, 18);
INSERT INTO user_1_file_00a1f910.administradores VALUES (10, 16);
INSERT INTO user_1_file_00a1f910.administradores VALUES (2, 3);
INSERT INTO user_1_file_00a1f910.administradores VALUES (3, 4);
INSERT INTO user_1_file_00a1f910.administradores VALUES (5, 7);
INSERT INTO user_1_file_00a1f910.administradores VALUES (7, 14);
INSERT INTO user_1_file_00a1f910.administradores VALUES (11, 18);
INSERT INTO user_1_file_00a1f910.administradores VALUES (10, 16);


--
-- TOC entry 5272 (class 0 OID 17901)
-- Dependencies: 279
-- Data for Name: categoria; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.categoria VALUES (1, 'Matrimonio');
INSERT INTO user_1_file_00a1f910.categoria VALUES (2, 'Bautizo');
INSERT INTO user_1_file_00a1f910.categoria VALUES (3, 'Comunion');
INSERT INTO user_1_file_00a1f910.categoria VALUES (4, 'Confirmacion');
INSERT INTO user_1_file_00a1f910.categoria VALUES (1, 'Matrimonio');
INSERT INTO user_1_file_00a1f910.categoria VALUES (2, 'Bautizo');
INSERT INTO user_1_file_00a1f910.categoria VALUES (3, 'Comunion');
INSERT INTO user_1_file_00a1f910.categoria VALUES (4, 'Confirmacion');


--
-- TOC entry 5273 (class 0 OID 17906)
-- Dependencies: 280
-- Data for Name: cursos; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.cursos VALUES (2, 'AAAAD', 'aaaa', '2025-03-07', '2025-03-29');
INSERT INTO user_1_file_00a1f910.cursos VALUES (2, 'AAAAD', 'aaaa', '2025-03-07', '2025-03-29');


--
-- TOC entry 5274 (class 0 OID 17911)
-- Dependencies: 281
-- Data for Name: eventos; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.eventos VALUES (2, 'xddd', 'aaa', '2025-03-28', 1);
INSERT INTO user_1_file_00a1f910.eventos VALUES (2, 'xddd', 'aaa', '2025-03-28', 1);


--
-- TOC entry 5275 (class 0 OID 17916)
-- Dependencies: 282
-- Data for Name: membresias; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.membresias VALUES (2, 1, '2025-03-24', '2025-06-20', 'aasdasd');
INSERT INTO user_1_file_00a1f910.membresias VALUES (2, 1, '2025-03-24', '2025-06-20', 'aasdasd');


--
-- TOC entry 5277 (class 0 OID 17924)
-- Dependencies: 284
-- Data for Name: miembro_curso; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (6, 3, 2);
INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (9, 4, 2);
INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (10, 1, 2);
INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (6, 3, 2);
INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (9, 4, 2);
INSERT INTO user_1_file_00a1f910.miembro_curso VALUES (10, 1, 2);


--
-- TOC entry 5278 (class 0 OID 17927)
-- Dependencies: 285
-- Data for Name: miembro_curso_actividad; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.miembro_curso_actividad VALUES (1, 6, 2, 70, '1743234376_1486472-rammus-5-article_image_bd-1.jpg');
INSERT INTO user_1_file_00a1f910.miembro_curso_actividad VALUES (1, 6, 2, 70, '1743234376_1486472-rammus-5-article_image_bd-1.jpg');


--
-- TOC entry 5276 (class 0 OID 17921)
-- Dependencies: 283
-- Data for Name: miembros; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.miembros VALUES (2, 8);
INSERT INTO user_1_file_00a1f910.miembros VALUES (6, 13);
INSERT INTO user_1_file_00a1f910.miembros VALUES (2, 8);
INSERT INTO user_1_file_00a1f910.miembros VALUES (6, 13);


--
-- TOC entry 5279 (class 0 OID 17931)
-- Dependencies: 286
-- Data for Name: ministerio; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.ministerio VALUES (1, 'ministerio del amor', 3);
INSERT INTO user_1_file_00a1f910.ministerio VALUES (1, 'ministerio del amor', 3);


--
-- TOC entry 5280 (class 0 OID 17934)
-- Dependencies: 287
-- Data for Name: ministerio_user; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.ministerio_user VALUES (3, 1, 1);
INSERT INTO user_1_file_00a1f910.ministerio_user VALUES (5, 4, 1);
INSERT INTO user_1_file_00a1f910.ministerio_user VALUES (3, 1, 1);
INSERT INTO user_1_file_00a1f910.ministerio_user VALUES (5, 4, 1);


--
-- TOC entry 5281 (class 0 OID 17937)
-- Dependencies: 288
-- Data for Name: model_has_role; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.model_has_role VALUES (7, 1);
INSERT INTO user_1_file_00a1f910.model_has_role VALUES (8, 4);
INSERT INTO user_1_file_00a1f910.model_has_role VALUES (7, 1);
INSERT INTO user_1_file_00a1f910.model_has_role VALUES (8, 4);


--
-- TOC entry 5282 (class 0 OID 17940)
-- Dependencies: 289
-- Data for Name: permissions; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.permissions VALUES (1, 'crear');
INSERT INTO user_1_file_00a1f910.permissions VALUES (2, 'editar');
INSERT INTO user_1_file_00a1f910.permissions VALUES (3, 'eliminar');
INSERT INTO user_1_file_00a1f910.permissions VALUES (4, 'subir');
INSERT INTO user_1_file_00a1f910.permissions VALUES (5, 'calificar');
INSERT INTO user_1_file_00a1f910.permissions VALUES (1, 'crear');
INSERT INTO user_1_file_00a1f910.permissions VALUES (2, 'editar');
INSERT INTO user_1_file_00a1f910.permissions VALUES (3, 'eliminar');
INSERT INTO user_1_file_00a1f910.permissions VALUES (4, 'subir');
INSERT INTO user_1_file_00a1f910.permissions VALUES (5, 'calificar');


--
-- TOC entry 5284 (class 0 OID 17946)
-- Dependencies: 291
-- Data for Name: role_has_permission; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 1);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 2);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 3);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (4, 4);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 1);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 2);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (1, 3);
INSERT INTO user_1_file_00a1f910.role_has_permission VALUES (4, 4);


--
-- TOC entry 5283 (class 0 OID 17943)
-- Dependencies: 290
-- Data for Name: roles; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.roles VALUES (1, 'Administrador');
INSERT INTO user_1_file_00a1f910.roles VALUES (4, 'Miembro');
INSERT INTO user_1_file_00a1f910.roles VALUES (1, 'Administrador');
INSERT INTO user_1_file_00a1f910.roles VALUES (4, 'Miembro');


--
-- TOC entry 5286 (class 0 OID 17954)
-- Dependencies: 293
-- Data for Name: sacramento_user; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (1, 1, 1);
INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (7, 2, 4);
INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (6, 2, 3);
INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (1, 1, 1);
INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (7, 2, 4);
INSERT INTO user_1_file_00a1f910.sacramento_user VALUES (6, 2, 3);


--
-- TOC entry 5285 (class 0 OID 17949)
-- Dependencies: 292
-- Data for Name: sacramentos; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.sacramentos VALUES (1, '2025-03-07', 'asdsad', 'asdasd', 2);
INSERT INTO user_1_file_00a1f910.sacramentos VALUES (2, '2025-03-25', 'sss', 'ss', 1);
INSERT INTO user_1_file_00a1f910.sacramentos VALUES (1, '2025-03-07', 'asdsad', 'asdasd', 2);
INSERT INTO user_1_file_00a1f910.sacramentos VALUES (2, '2025-03-25', 'sss', 'ss', 1);


--
-- TOC entry 5287 (class 0 OID 17957)
-- Dependencies: 294
-- Data for Name: users; Type: TABLE DATA; Schema: user_1_file_00a1f910; Owner: postgres
--

INSERT INTO user_1_file_00a1f910.users VALUES (1, 'usuario1', '$2y$10$IOraQxcAm51IkWWSq6ncGODuIkmIPdhftzRabRUPgy9sUqi7r8MXi', 'user@gmail.com', '2025-03-22 03:33:48', NULL, NULL, NULL, NULL, NULL, NULL);
INSERT INTO user_1_file_00a1f910.users VALUES (3, 'admin', '12345678', 'admin@gmail.com', '2025-03-23 02:03:13', 'Santa Cruz', 7844521, 457456856, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-14', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (4, 'asdxdd123', '12345678', 'asdasd@gmail.com', '2025-03-23 02:44:00', 'Santa Cruz', 78403, 4523, '', '2025-03-14', 'Femenino');
INSERT INTO user_1_file_00a1f910.users VALUES (8, 'user1', '12345678', 'user1@gmail.com', '2025-03-29 19:29:32', 'Santa Cruz', 78408033, 12311, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-13', 'Femenino');
INSERT INTO user_1_file_00a1f910.users VALUES (7, 'adm@gmail.com', '12345678', 'adm@gmail.com', '2025-03-29 19:25:28', 'Santa Cruz', 78408033, 4213, '/uploads/1486472-rammus-5-article_image_bd-1.jpg', '2025-03-28', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (13, 'Usersdfsdfsdf', '12345678', 'fgdf@das.dcom', '2025-06-13 01:33:52', 'asdsad', 784, 1234, '', '2025-06-26', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (14, 'Addmminn', '12345678', 'asdasd@mgia.com', '2025-06-13 01:41:20', 'gfhfgh', 22222, 4563, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (16, 'usssrrr', '12345678', 'dsfsdf@sdf.co', '2025-06-13 04:19:30', 'ffff', 545, 32, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (18, 'addddmdd', '12345678', 'sdfdsf@dsfds.co', '2025-06-13 04:33:21', 'defdf', 87876, 56876, '', '2025-06-14', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (1, 'usuario1', '$2y$10$IOraQxcAm51IkWWSq6ncGODuIkmIPdhftzRabRUPgy9sUqi7r8MXi', 'user@gmail.com', '2025-03-22 03:33:48', NULL, NULL, NULL, NULL, NULL, NULL);
INSERT INTO user_1_file_00a1f910.users VALUES (3, 'admin', '12345678', 'admin@gmail.com', '2025-03-23 02:03:13', 'Santa Cruz', 7844521, 457456856, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-14', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (4, 'asdxdd123', '12345678', 'asdasd@gmail.com', '2025-03-23 02:44:00', 'Santa Cruz', 78403, 4523, '', '2025-03-14', 'Femenino');
INSERT INTO user_1_file_00a1f910.users VALUES (8, 'user1', '12345678', 'user1@gmail.com', '2025-03-29 19:29:32', 'Santa Cruz', 78408033, 12311, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-13', 'Femenino');
INSERT INTO user_1_file_00a1f910.users VALUES (7, 'adm@gmail.com', '12345678', 'adm@gmail.com', '2025-03-29 19:25:28', 'Santa Cruz', 78408033, 4213, '/uploads/1486472-rammus-5-article_image_bd-1.jpg', '2025-03-28', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (13, 'Usersdfsdfsdf', '12345678', 'fgdf@das.dcom', '2025-06-13 01:33:52', 'asdsad', 784, 1234, '', '2025-06-26', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (14, 'Addmminn', '12345678', 'asdasd@mgia.com', '2025-06-13 01:41:20', 'gfhfgh', 22222, 4563, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (16, 'usssrrr', '12345678', 'dsfsdf@sdf.co', '2025-06-13 04:19:30', 'ffff', 545, 32, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_00a1f910.users VALUES (18, 'addddmdd', '12345678', 'sdfdsf@dsfds.co', '2025-06-13 04:33:21', 'defdf', 87876, 56876, '', '2025-06-14', 'Masculino');


--
-- TOC entry 5268 (class 0 OID 17082)
-- Dependencies: 275
-- Data for Name: actividades_tarea; Type: TABLE DATA; Schema: user_1_file_7639c196; Owner: postgres
--



--
-- TOC entry 5289 (class 0 OID 17978)
-- Dependencies: 296
-- Data for Name: departamentos; Type: TABLE DATA; Schema: user_1_file_b39d309b; Owner: postgres
--

INSERT INTO user_1_file_b39d309b.departamentos VALUES (1, 'Recursos Humanos', 'Santa Cruz');
INSERT INTO user_1_file_b39d309b.departamentos VALUES (2, 'Desarrollo', 'La Paz');
INSERT INTO user_1_file_b39d309b.departamentos VALUES (3, 'Ventas', 'Cochabamba');


--
-- TOC entry 5291 (class 0 OID 17985)
-- Dependencies: 298
-- Data for Name: empleados; Type: TABLE DATA; Schema: user_1_file_b39d309b; Owner: postgres
--

INSERT INTO user_1_file_b39d309b.empleados VALUES (1, 'Ana López', 'Desarrolladora', 5000.00, 2);
INSERT INTO user_1_file_b39d309b.empleados VALUES (2, 'Carlos Pérez', 'Vendedor', 3000.00, 3);
INSERT INTO user_1_file_b39d309b.empleados VALUES (3, 'Lucía Torres', 'Recursos Humanos', 3500.00, 1);
INSERT INTO user_1_file_b39d309b.empleados VALUES (4, 'Mario Gómez', 'Desarrollador', 5200.00, 2);


--
-- TOC entry 5293 (class 0 OID 17997)
-- Dependencies: 300
-- Data for Name: proyectos; Type: TABLE DATA; Schema: user_1_file_b39d309b; Owner: postgres
--

INSERT INTO user_1_file_b39d309b.proyectos VALUES (1, 'Sistema Web', 10000.00, 2);
INSERT INTO user_1_file_b39d309b.proyectos VALUES (2, 'Campaña Ventas', 7000.00, 3);
INSERT INTO user_1_file_b39d309b.proyectos VALUES (3, 'Capacitación Interna', 5000.00, 1);


--
-- TOC entry 5269 (class 0 OID 17089)
-- Dependencies: 276
-- Data for Name: actividades_tarea; Type: TABLE DATA; Schema: user_1_file_d4a80680; Owner: postgres
--



--
-- TOC entry 5294 (class 0 OID 18022)
-- Dependencies: 301
-- Data for Name: actividades_tarea; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.actividades_tarea VALUES (1, 2, 'ddd', 'dddd', '433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-22');
INSERT INTO user_1_file_f31fa970.actividades_tarea VALUES (2, 2, 'aa', 'assssdasdsad', 'Conectar el DHCP de un router a un switch a maquinas.docx', '2025-03-28');


--
-- TOC entry 5295 (class 0 OID 18028)
-- Dependencies: 302
-- Data for Name: administradores; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.administradores VALUES (2, 3);
INSERT INTO user_1_file_f31fa970.administradores VALUES (3, 4);
INSERT INTO user_1_file_f31fa970.administradores VALUES (5, 7);
INSERT INTO user_1_file_f31fa970.administradores VALUES (7, 14);
INSERT INTO user_1_file_f31fa970.administradores VALUES (11, 18);
INSERT INTO user_1_file_f31fa970.administradores VALUES (10, 16);


--
-- TOC entry 5296 (class 0 OID 18031)
-- Dependencies: 303
-- Data for Name: categoria; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.categoria VALUES (1, 'Matrimonio');
INSERT INTO user_1_file_f31fa970.categoria VALUES (2, 'Bautizo');
INSERT INTO user_1_file_f31fa970.categoria VALUES (3, 'Comunion');
INSERT INTO user_1_file_f31fa970.categoria VALUES (4, 'Confirmacion');


--
-- TOC entry 5297 (class 0 OID 18036)
-- Dependencies: 304
-- Data for Name: cursos; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.cursos VALUES (2, 'AAAAD', 'aaaa', '2025-03-07', '2025-03-29');


--
-- TOC entry 5298 (class 0 OID 18041)
-- Dependencies: 305
-- Data for Name: eventos; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.eventos VALUES (2, 'xddd', 'aaa', '2025-03-28', 1);


--
-- TOC entry 5299 (class 0 OID 18046)
-- Dependencies: 306
-- Data for Name: membresias; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.membresias VALUES (2, 1, '2025-03-24', '2025-06-20', 'aasdasd');


--
-- TOC entry 5301 (class 0 OID 18054)
-- Dependencies: 308
-- Data for Name: miembro_curso; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.miembro_curso VALUES (6, 3, 2);
INSERT INTO user_1_file_f31fa970.miembro_curso VALUES (9, 4, 2);
INSERT INTO user_1_file_f31fa970.miembro_curso VALUES (10, 1, 2);


--
-- TOC entry 5302 (class 0 OID 18057)
-- Dependencies: 309
-- Data for Name: miembro_curso_actividad; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.miembro_curso_actividad VALUES (1, 6, 2, 70, '1743234376_1486472-rammus-5-article_image_bd-1.jpg');


--
-- TOC entry 5300 (class 0 OID 18051)
-- Dependencies: 307
-- Data for Name: miembros; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.miembros VALUES (2, 8);
INSERT INTO user_1_file_f31fa970.miembros VALUES (6, 13);


--
-- TOC entry 5303 (class 0 OID 18061)
-- Dependencies: 310
-- Data for Name: ministerio; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.ministerio VALUES (1, 'ministerio del amor', 3);


--
-- TOC entry 5304 (class 0 OID 18064)
-- Dependencies: 311
-- Data for Name: ministerio_user; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.ministerio_user VALUES (3, 1, 1);
INSERT INTO user_1_file_f31fa970.ministerio_user VALUES (5, 4, 1);


--
-- TOC entry 5305 (class 0 OID 18067)
-- Dependencies: 312
-- Data for Name: model_has_role; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.model_has_role VALUES (7, 1);
INSERT INTO user_1_file_f31fa970.model_has_role VALUES (8, 4);


--
-- TOC entry 5306 (class 0 OID 18070)
-- Dependencies: 313
-- Data for Name: permissions; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.permissions VALUES (1, 'crear');
INSERT INTO user_1_file_f31fa970.permissions VALUES (2, 'editar');
INSERT INTO user_1_file_f31fa970.permissions VALUES (3, 'eliminar');
INSERT INTO user_1_file_f31fa970.permissions VALUES (4, 'subir');
INSERT INTO user_1_file_f31fa970.permissions VALUES (5, 'calificar');


--
-- TOC entry 5308 (class 0 OID 18076)
-- Dependencies: 315
-- Data for Name: role_has_permission; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.role_has_permission VALUES (1, 1);
INSERT INTO user_1_file_f31fa970.role_has_permission VALUES (1, 2);
INSERT INTO user_1_file_f31fa970.role_has_permission VALUES (1, 3);
INSERT INTO user_1_file_f31fa970.role_has_permission VALUES (4, 4);


--
-- TOC entry 5307 (class 0 OID 18073)
-- Dependencies: 314
-- Data for Name: roles; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.roles VALUES (1, 'Administrador');
INSERT INTO user_1_file_f31fa970.roles VALUES (4, 'Miembro');


--
-- TOC entry 5310 (class 0 OID 18084)
-- Dependencies: 317
-- Data for Name: sacramento_user; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.sacramento_user VALUES (1, 1, 1);
INSERT INTO user_1_file_f31fa970.sacramento_user VALUES (7, 2, 4);
INSERT INTO user_1_file_f31fa970.sacramento_user VALUES (6, 2, 3);


--
-- TOC entry 5309 (class 0 OID 18079)
-- Dependencies: 316
-- Data for Name: sacramentos; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.sacramentos VALUES (1, '2025-03-07', 'asdsad', 'asdasd', 2);
INSERT INTO user_1_file_f31fa970.sacramentos VALUES (2, '2025-03-25', 'sss', 'ss', 1);


--
-- TOC entry 5311 (class 0 OID 18087)
-- Dependencies: 318
-- Data for Name: users; Type: TABLE DATA; Schema: user_1_file_f31fa970; Owner: postgres
--

INSERT INTO user_1_file_f31fa970.users VALUES (1, 'usuario1', '$2y$10$IOraQxcAm51IkWWSq6ncGODuIkmIPdhftzRabRUPgy9sUqi7r8MXi', 'user@gmail.com', '2025-03-22 03:33:48-04', NULL, NULL, NULL, NULL, NULL, NULL);
INSERT INTO user_1_file_f31fa970.users VALUES (3, 'admin', '12345678', 'admin@gmail.com', '2025-03-23 02:03:13-04', 'Santa Cruz', 7844521, 457456856, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-14', 'Masculino');
INSERT INTO user_1_file_f31fa970.users VALUES (4, 'asdxdd123', '12345678', 'asdasd@gmail.com', '2025-03-23 02:44:00-04', 'Santa Cruz', 78403, 4523, '', '2025-03-14', 'Femenino');
INSERT INTO user_1_file_f31fa970.users VALUES (8, 'user1', '12345678', 'user1@gmail.com', '2025-03-29 19:29:32-04', 'Santa Cruz', 78408033, 12311, '/uploads/433622785_3658678264380282_5049959931368625275_n.jpg', '2025-03-13', 'Femenino');
INSERT INTO user_1_file_f31fa970.users VALUES (7, 'adm@gmail.com', '12345678', 'adm@gmail.com', '2025-03-29 19:25:28-04', 'Santa Cruz', 78408033, 4213, '/uploads/1486472-rammus-5-article_image_bd-1.jpg', '2025-03-28', 'Masculino');
INSERT INTO user_1_file_f31fa970.users VALUES (13, 'Usersdfsdfsdf', '12345678', 'fgdf@das.dcom', '2025-06-13 01:33:52-04', 'asdsad', 784, 1234, '', '2025-06-26', 'Masculino');
INSERT INTO user_1_file_f31fa970.users VALUES (14, 'Addmminn', '12345678', 'asdasd@mgia.com', '2025-06-13 01:41:20-04', 'gfhfgh', 22222, 4563, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_f31fa970.users VALUES (16, 'usssrrr', '12345678', 'dsfsdf@sdf.co', '2025-06-13 04:19:30-04', 'ffff', 545, 32, '', '2025-06-12', 'Masculino');
INSERT INTO user_1_file_f31fa970.users VALUES (18, 'addddmdd', '12345678', 'sdfdsf@dsfds.co', '2025-06-13 04:33:21-04', 'defdf', 87876, 56876, '', '2025-06-14', 'Masculino');


--
-- TOC entry 5320 (class 0 OID 0)
-- Dependencies: 256
-- Name: auth_group_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_group_id_seq', 1, false);


--
-- TOC entry 5321 (class 0 OID 0)
-- Dependencies: 258
-- Name: auth_group_permissions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_group_permissions_id_seq', 1, false);


--
-- TOC entry 5322 (class 0 OID 0)
-- Dependencies: 254
-- Name: auth_permission_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_permission_id_seq', 36, true);


--
-- TOC entry 5323 (class 0 OID 0)
-- Dependencies: 262
-- Name: auth_user_groups_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_user_groups_id_seq', 1, false);


--
-- TOC entry 5324 (class 0 OID 0)
-- Dependencies: 260
-- Name: auth_user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_user_id_seq', 1, true);


--
-- TOC entry 5325 (class 0 OID 0)
-- Dependencies: 264
-- Name: auth_user_user_permissions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.auth_user_user_permissions_id_seq', 1, false);


--
-- TOC entry 5326 (class 0 OID 0)
-- Dependencies: 266
-- Name: django_admin_log_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.django_admin_log_id_seq', 1, false);


--
-- TOC entry 5327 (class 0 OID 0)
-- Dependencies: 252
-- Name: django_content_type_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.django_content_type_id_seq', 9, true);


--
-- TOC entry 5328 (class 0 OID 0)
-- Dependencies: 250
-- Name: django_migrations_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.django_migrations_id_seq', 19, true);


--
-- TOC entry 5329 (class 0 OID 0)
-- Dependencies: 269
-- Name: ingestion_datasource_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ingestion_datasource_id_seq', 44, true);


--
-- TOC entry 5330 (class 0 OID 0)
-- Dependencies: 271
-- Name: ingestion_externalconnection_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ingestion_externalconnection_id_seq', 1, false);


--
-- TOC entry 5331 (class 0 OID 0)
-- Dependencies: 273
-- Name: ingestion_uploadeddataset_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.ingestion_uploadeddataset_id_seq', 44, true);


--
-- TOC entry 5332 (class 0 OID 0)
-- Dependencies: 295
-- Name: departamentos_id_seq; Type: SEQUENCE SET; Schema: user_1_file_b39d309b; Owner: postgres
--

SELECT pg_catalog.setval('user_1_file_b39d309b.departamentos_id_seq', 3, true);


--
-- TOC entry 5333 (class 0 OID 0)
-- Dependencies: 297
-- Name: empleados_id_seq; Type: SEQUENCE SET; Schema: user_1_file_b39d309b; Owner: postgres
--

SELECT pg_catalog.setval('user_1_file_b39d309b.empleados_id_seq', 4, true);


--
-- TOC entry 5334 (class 0 OID 0)
-- Dependencies: 299
-- Name: proyectos_id_seq; Type: SEQUENCE SET; Schema: user_1_file_b39d309b; Owner: postgres
--

SELECT pg_catalog.setval('user_1_file_b39d309b.proyectos_id_seq', 3, true);


--
-- TOC entry 5033 (class 2606 OID 16525)
-- Name: auth_group auth_group_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group
    ADD CONSTRAINT auth_group_name_key UNIQUE (name);


--
-- TOC entry 5038 (class 2606 OID 16456)
-- Name: auth_group_permissions auth_group_permissions_group_id_permission_id_0cd325b0_uniq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group_permissions
    ADD CONSTRAINT auth_group_permissions_group_id_permission_id_0cd325b0_uniq UNIQUE (group_id, permission_id);


--
-- TOC entry 5041 (class 2606 OID 16425)
-- Name: auth_group_permissions auth_group_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group_permissions
    ADD CONSTRAINT auth_group_permissions_pkey PRIMARY KEY (id);


--
-- TOC entry 5035 (class 2606 OID 16417)
-- Name: auth_group auth_group_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group
    ADD CONSTRAINT auth_group_pkey PRIMARY KEY (id);


--
-- TOC entry 5028 (class 2606 OID 16447)
-- Name: auth_permission auth_permission_content_type_id_codename_01ab375a_uniq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_permission
    ADD CONSTRAINT auth_permission_content_type_id_codename_01ab375a_uniq UNIQUE (content_type_id, codename);


--
-- TOC entry 5030 (class 2606 OID 16411)
-- Name: auth_permission auth_permission_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_permission
    ADD CONSTRAINT auth_permission_pkey PRIMARY KEY (id);


--
-- TOC entry 5049 (class 2606 OID 16439)
-- Name: auth_user_groups auth_user_groups_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_groups
    ADD CONSTRAINT auth_user_groups_pkey PRIMARY KEY (id);


--
-- TOC entry 5052 (class 2606 OID 16471)
-- Name: auth_user_groups auth_user_groups_user_id_group_id_94350c0c_uniq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_groups
    ADD CONSTRAINT auth_user_groups_user_id_group_id_94350c0c_uniq UNIQUE (user_id, group_id);


--
-- TOC entry 5043 (class 2606 OID 16431)
-- Name: auth_user auth_user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user
    ADD CONSTRAINT auth_user_pkey PRIMARY KEY (id);


--
-- TOC entry 5055 (class 2606 OID 16445)
-- Name: auth_user_user_permissions auth_user_user_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_user_permissions
    ADD CONSTRAINT auth_user_user_permissions_pkey PRIMARY KEY (id);


--
-- TOC entry 5058 (class 2606 OID 16485)
-- Name: auth_user_user_permissions auth_user_user_permissions_user_id_permission_id_14a6b632_uniq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_user_permissions
    ADD CONSTRAINT auth_user_user_permissions_user_id_permission_id_14a6b632_uniq UNIQUE (user_id, permission_id);


--
-- TOC entry 5046 (class 2606 OID 16520)
-- Name: auth_user auth_user_username_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user
    ADD CONSTRAINT auth_user_username_key UNIQUE (username);


--
-- TOC entry 5061 (class 2606 OID 16506)
-- Name: django_admin_log django_admin_log_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_admin_log
    ADD CONSTRAINT django_admin_log_pkey PRIMARY KEY (id);


--
-- TOC entry 5023 (class 2606 OID 16405)
-- Name: django_content_type django_content_type_app_label_model_76bd3d3b_uniq; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_content_type
    ADD CONSTRAINT django_content_type_app_label_model_76bd3d3b_uniq UNIQUE (app_label, model);


--
-- TOC entry 5025 (class 2606 OID 16403)
-- Name: django_content_type django_content_type_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_content_type
    ADD CONSTRAINT django_content_type_pkey PRIMARY KEY (id);


--
-- TOC entry 5021 (class 2606 OID 16397)
-- Name: django_migrations django_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_migrations
    ADD CONSTRAINT django_migrations_pkey PRIMARY KEY (id);


--
-- TOC entry 5065 (class 2606 OID 16533)
-- Name: django_session django_session_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_session
    ADD CONSTRAINT django_session_pkey PRIMARY KEY (session_key);


--
-- TOC entry 5069 (class 2606 OID 16541)
-- Name: ingestion_datasource ingestion_datasource_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_datasource
    ADD CONSTRAINT ingestion_datasource_pkey PRIMARY KEY (id);


--
-- TOC entry 5071 (class 2606 OID 16549)
-- Name: ingestion_externalconnection ingestion_externalconnection_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_externalconnection
    ADD CONSTRAINT ingestion_externalconnection_pkey PRIMARY KEY (id);


--
-- TOC entry 5073 (class 2606 OID 16551)
-- Name: ingestion_externalconnection ingestion_externalconnection_source_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_externalconnection
    ADD CONSTRAINT ingestion_externalconnection_source_id_key UNIQUE (source_id);


--
-- TOC entry 5075 (class 2606 OID 16560)
-- Name: ingestion_uploadeddataset ingestion_uploadeddataset_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_uploadeddataset
    ADD CONSTRAINT ingestion_uploadeddataset_pkey PRIMARY KEY (id);


--
-- TOC entry 5077 (class 2606 OID 16562)
-- Name: ingestion_uploadeddataset ingestion_uploadeddataset_source_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_uploadeddataset
    ADD CONSTRAINT ingestion_uploadeddataset_source_id_key UNIQUE (source_id);


--
-- TOC entry 5079 (class 2606 OID 17983)
-- Name: departamentos departamentos_pkey; Type: CONSTRAINT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.departamentos
    ADD CONSTRAINT departamentos_pkey PRIMARY KEY (id);


--
-- TOC entry 5081 (class 2606 OID 17990)
-- Name: empleados empleados_pkey; Type: CONSTRAINT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.empleados
    ADD CONSTRAINT empleados_pkey PRIMARY KEY (id);


--
-- TOC entry 5083 (class 2606 OID 18002)
-- Name: proyectos proyectos_pkey; Type: CONSTRAINT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.proyectos
    ADD CONSTRAINT proyectos_pkey PRIMARY KEY (id);


--
-- TOC entry 5031 (class 1259 OID 16526)
-- Name: auth_group_name_a6ea08ec_like; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_group_name_a6ea08ec_like ON public.auth_group USING btree (name varchar_pattern_ops);


--
-- TOC entry 5036 (class 1259 OID 16467)
-- Name: auth_group_permissions_group_id_b120cbf9; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_group_permissions_group_id_b120cbf9 ON public.auth_group_permissions USING btree (group_id);


--
-- TOC entry 5039 (class 1259 OID 16468)
-- Name: auth_group_permissions_permission_id_84c5c92e; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_group_permissions_permission_id_84c5c92e ON public.auth_group_permissions USING btree (permission_id);


--
-- TOC entry 5026 (class 1259 OID 16453)
-- Name: auth_permission_content_type_id_2f476e4b; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_permission_content_type_id_2f476e4b ON public.auth_permission USING btree (content_type_id);


--
-- TOC entry 5047 (class 1259 OID 16483)
-- Name: auth_user_groups_group_id_97559544; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_user_groups_group_id_97559544 ON public.auth_user_groups USING btree (group_id);


--
-- TOC entry 5050 (class 1259 OID 16482)
-- Name: auth_user_groups_user_id_6a12ed8b; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_user_groups_user_id_6a12ed8b ON public.auth_user_groups USING btree (user_id);


--
-- TOC entry 5053 (class 1259 OID 16497)
-- Name: auth_user_user_permissions_permission_id_1fbb5f2c; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_user_user_permissions_permission_id_1fbb5f2c ON public.auth_user_user_permissions USING btree (permission_id);


--
-- TOC entry 5056 (class 1259 OID 16496)
-- Name: auth_user_user_permissions_user_id_a95ead1b; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_user_user_permissions_user_id_a95ead1b ON public.auth_user_user_permissions USING btree (user_id);


--
-- TOC entry 5044 (class 1259 OID 16521)
-- Name: auth_user_username_6821ab7c_like; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX auth_user_username_6821ab7c_like ON public.auth_user USING btree (username varchar_pattern_ops);


--
-- TOC entry 5059 (class 1259 OID 16517)
-- Name: django_admin_log_content_type_id_c4bce8eb; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX django_admin_log_content_type_id_c4bce8eb ON public.django_admin_log USING btree (content_type_id);


--
-- TOC entry 5062 (class 1259 OID 16518)
-- Name: django_admin_log_user_id_c564eba6; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX django_admin_log_user_id_c564eba6 ON public.django_admin_log USING btree (user_id);


--
-- TOC entry 5063 (class 1259 OID 16535)
-- Name: django_session_expire_date_a5c62663; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX django_session_expire_date_a5c62663 ON public.django_session USING btree (expire_date);


--
-- TOC entry 5066 (class 1259 OID 16534)
-- Name: django_session_session_key_c0390e0f_like; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX django_session_session_key_c0390e0f_like ON public.django_session USING btree (session_key varchar_pattern_ops);


--
-- TOC entry 5067 (class 1259 OID 16568)
-- Name: ingestion_datasource_owner_id_ae61d6d6; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ingestion_datasource_owner_id_ae61d6d6 ON public.ingestion_datasource USING btree (owner_id);


--
-- TOC entry 5085 (class 2606 OID 16462)
-- Name: auth_group_permissions auth_group_permissio_permission_id_84c5c92e_fk_auth_perm; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group_permissions
    ADD CONSTRAINT auth_group_permissio_permission_id_84c5c92e_fk_auth_perm FOREIGN KEY (permission_id) REFERENCES public.auth_permission(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5086 (class 2606 OID 16457)
-- Name: auth_group_permissions auth_group_permissions_group_id_b120cbf9_fk_auth_group_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_group_permissions
    ADD CONSTRAINT auth_group_permissions_group_id_b120cbf9_fk_auth_group_id FOREIGN KEY (group_id) REFERENCES public.auth_group(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5084 (class 2606 OID 16448)
-- Name: auth_permission auth_permission_content_type_id_2f476e4b_fk_django_co; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_permission
    ADD CONSTRAINT auth_permission_content_type_id_2f476e4b_fk_django_co FOREIGN KEY (content_type_id) REFERENCES public.django_content_type(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5087 (class 2606 OID 16477)
-- Name: auth_user_groups auth_user_groups_group_id_97559544_fk_auth_group_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_groups
    ADD CONSTRAINT auth_user_groups_group_id_97559544_fk_auth_group_id FOREIGN KEY (group_id) REFERENCES public.auth_group(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5088 (class 2606 OID 16472)
-- Name: auth_user_groups auth_user_groups_user_id_6a12ed8b_fk_auth_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_groups
    ADD CONSTRAINT auth_user_groups_user_id_6a12ed8b_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES public.auth_user(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5089 (class 2606 OID 16491)
-- Name: auth_user_user_permissions auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_user_permissions
    ADD CONSTRAINT auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm FOREIGN KEY (permission_id) REFERENCES public.auth_permission(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5090 (class 2606 OID 16486)
-- Name: auth_user_user_permissions auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auth_user_user_permissions
    ADD CONSTRAINT auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES public.auth_user(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5091 (class 2606 OID 16507)
-- Name: django_admin_log django_admin_log_content_type_id_c4bce8eb_fk_django_co; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_admin_log
    ADD CONSTRAINT django_admin_log_content_type_id_c4bce8eb_fk_django_co FOREIGN KEY (content_type_id) REFERENCES public.django_content_type(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5092 (class 2606 OID 16512)
-- Name: django_admin_log django_admin_log_user_id_c564eba6_fk_auth_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.django_admin_log
    ADD CONSTRAINT django_admin_log_user_id_c564eba6_fk_auth_user_id FOREIGN KEY (user_id) REFERENCES public.auth_user(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5093 (class 2606 OID 16563)
-- Name: ingestion_datasource ingestion_datasource_owner_id_ae61d6d6_fk_auth_user_id; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_datasource
    ADD CONSTRAINT ingestion_datasource_owner_id_ae61d6d6_fk_auth_user_id FOREIGN KEY (owner_id) REFERENCES public.auth_user(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5094 (class 2606 OID 16569)
-- Name: ingestion_externalconnection ingestion_externalco_source_id_186f64b7_fk_ingestion; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_externalconnection
    ADD CONSTRAINT ingestion_externalco_source_id_186f64b7_fk_ingestion FOREIGN KEY (source_id) REFERENCES public.ingestion_datasource(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5095 (class 2606 OID 16574)
-- Name: ingestion_uploadeddataset ingestion_uploadedda_source_id_948cfada_fk_ingestion; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ingestion_uploadeddataset
    ADD CONSTRAINT ingestion_uploadedda_source_id_948cfada_fk_ingestion FOREIGN KEY (source_id) REFERENCES public.ingestion_datasource(id) DEFERRABLE INITIALLY DEFERRED;


--
-- TOC entry 5096 (class 2606 OID 17991)
-- Name: empleados empleados_departamento_id_fkey; Type: FK CONSTRAINT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.empleados
    ADD CONSTRAINT empleados_departamento_id_fkey FOREIGN KEY (departamento_id) REFERENCES user_1_file_b39d309b.departamentos(id);


--
-- TOC entry 5097 (class 2606 OID 18003)
-- Name: proyectos proyectos_departamento_id_fkey; Type: FK CONSTRAINT; Schema: user_1_file_b39d309b; Owner: postgres
--

ALTER TABLE ONLY user_1_file_b39d309b.proyectos
    ADD CONSTRAINT proyectos_departamento_id_fkey FOREIGN KEY (departamento_id) REFERENCES user_1_file_b39d309b.departamentos(id);


-- Completed on 2025-09-14 17:30:38

--
-- PostgreSQL database dump complete
--

\unrestrict 2iv7kOFj04hIgDwiBuyVpH48TfMiOTWaj9IaPAGDnpnTZlredtGpEwjsGD0hQon


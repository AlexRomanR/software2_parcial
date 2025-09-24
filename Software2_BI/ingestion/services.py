import re
import sqlparse
import pandas as pd
from pathlib import Path
from django.conf import settings
from sqlalchemy import create_engine, text
import chardet
import sqlglot
import os
import google.generativeai as genai
from django.conf import settings
import json


# Construye engine SQLAlchemy desde la BD por defecto de Django (PostgreSQL)
def get_engine():
    db = settings.DATABASES["default"]
    if db["ENGINE"].endswith("postgresql") or "psycopg2" in db["ENGINE"]:
        url = f"postgresql+psycopg2://{db['USER']}:{db['PASSWORD']}@{db['HOST']}:{db['PORT']}/{db['NAME']}"
    else:
        # agrega otros si quisieras, pero recomendamos Postgres
        raise RuntimeError("Usa PostgreSQL para ingestas con schema.")
    return create_engine(url, future=True)

# Sanitiza nombre de esquema/tabla (solo letras, números y _)
def sanitize_identifier(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", name).lower().strip("_")
    return clean[:60] or "ds"

def ensure_schema(engine, schema: str):
    with engine.begin() as conn:
        conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

def import_csv_or_excel(file_path: str, schema: str, table: str):
    engine = get_engine()
    ensure_schema(engine, schema)

    # Detecta por extensión
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        df = read_csv_with_auto_encoding(path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    else:
        df = pd.read_excel(path)


    # Volcar datos (replace para MVP)
    df.to_sql(table, engine, schema=schema, if_exists="replace", index=False)
    return {"rows": len(df), "columns": list(df.columns)}

FORBIDDEN = re.compile(
    r"\b(ALTER\s+SYSTEM|CREATE\s+USER|GRANT|REVOKE|TRUNCATE|ALTER\s+ROLE)\b",
    re.I
)

# --------- Filtros de seguridad ----------
FORBIDDEN = re.compile(
    r"\b(ALTER\s+SYSTEM|CREATE\s+USER|GRANT|REVOKE|TRUNCATE|ALTER\s+ROLE)\b",
    re.I
)

# --------- Detección y limpieza MySQL ----------
def _looks_like_mysql(sql_text: str) -> bool:
    patterns = [
        r'`', r'\bENGINE\s*=', r'\bAUTO_INCREMENT\b', r'\bUNSIGNED\b',
        r'\bLOCK\s+TABLES\b', r'\bUNLOCK\s+TABLES\b', r'\bDELIMITER\b',
        r'\bCHARSET\b', r'\bCOLLATE\b', r'\bENUM\s*\(',
        r'\bint\s*\(\s*\d+\s*\)',
        r'\b(UNIQUE\s+)?KEY\b\s+`?',  # índices dentro del CREATE
    ]
    return any(re.search(p, sql_text, re.I) for p in patterns)

def _strip_db_qualifier(sql_text: str) -> str:
    # CREATE TABLE db.tbl -> CREATE TABLE tbl
    x = re.sub(
        r'(?i)(\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?)'
        r'((?:"[^"]+"|`[^`]+`|[a-zA-Z_]\w*)\.)'
        r'(?P<tbl>"[^"]+"|`[^`]+`|[a-zA-Z_]\w*)',
        r'\1\g<tbl>', sql_text,
    )
    # INSERT INTO db.tbl -> INSERT INTO tbl
    x = re.sub(
        r'(?i)(\bINSERT\s+INTO\s+)'
        r'((?:"[^"]+"|`[^`]+`|[a-zA-Z_]\w*)\.)'
        r'(?P<tbl>"[^"]+"|`[^`]+`|[a-zA-Z_]\w*)',
        r'\1\g<tbl>', x,
    )
    return x

def _remove_mysql_global_noise(sql_text: str) -> str:
    x = sql_text
    x = re.sub(r'(?im)^\s*DELIMITER\s+.+$', '', x)
    x = re.sub(r'(?im)^\s*(LOCK|UNLOCK)\s+TABLES.*?;$', '', x)
    x = re.sub(r'(?im)^\s*SET\s+[^;]+;$', '', x)
    x = re.sub(r'(?im)^\s*USE\s+["`]?[\w-]+["`]?\s*;$', '', x)
    return x

def _remove_mysql_trailers(stmt: str) -> str:
    """Quita colas típicas al final de CREATE TABLE (ENGINE, CHARSET, COMMENT...)."""
    x = re.sub(r'\)\s*ENGINE\s*=\s*[^;]+;', ');', stmt, flags=re.I|re.S)
    x = re.sub(r'\)\s*DEFAULT\s+CHARSET\s*=\s*\w+(\s+COLLATE\s*=\s*\w+)?\s*;', ');', x, flags=re.I|re.S)
    x = re.sub(r'\)\s*CHARSET\s*=\s*\w+(\s+COLLATE\s*=\s*\w+)?\s*;', ');', x, flags=re.I|re.S)
    return x

def _strip_keys_inside_create(stmt: str) -> str:
    """
    Elimina líneas de KEY / UNIQUE KEY dentro del CREATE TABLE (no PRIMARY KEY).
    Deja PRIMARY KEY, que sí es válido.
    """
    if not re.match(r'(?is)^\s*CREATE\s+TABLE\b', stmt):
        return stmt

    # separa cabecera, cuerpo (paréntesis) y cola
    m = re.search(r'(?is)^\s*(CREATE\s+TABLE[^(]+)\((.*)\)(.*)$', stmt)
    if not m:
        return stmt
    head, body, tail = m.group(1), m.group(2), m.group(3)

    # quita líneas KEY...
    lines = [ln for ln in re.split(r',\s*\n|,\n|\n', body)]
    keep = []
    for ln in lines:
        ln_clean = ln.strip()
        if re.match(r'(?i)^(UNIQUE\s+)?KEY\b', ln_clean):
            continue
        keep.append(ln)
    # recompón con comas
    new_body = ',\n'.join([l.strip() for l in keep if l.strip()])
    return f"{head}(\n{new_body}\n){tail}"

def _normalize_post_transpile(stmt: str) -> str:
    x = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S)  # quita comentarios C-style
    x = x.replace('`', '"')

    # 👇 SIN \b al final (clave para INT(11) y similares)
    x = re.sub(r'(?i)\bTINYINT\s*\(\s*1\s*\)', 'BOOLEAN', x)
    x = re.sub(r'(?i)\bTINYINT\s*\(\s*\d+\s*\)', 'SMALLINT', x)
    x = re.sub(r'(?i)\bSMALLINT\s*\(\s*\d+\s*\)', 'SMALLINT', x)
    x = re.sub(r'(?i)\bINT\s*\(\s*\d+\s*\)', 'INTEGER', x)
    x = re.sub(r'(?i)\bBIGINT\s*\(\s*\d+\s*\)', 'BIGINT', x)

    x = re.sub(
        r'(?i)\b("?[a-zA-Z_]\w*"?\s+)(INTEGER|BIGINT|SMALLINT)[^,]*?\bAUTO_INCREMENT\b',
        r'\1\2 GENERATED BY DEFAULT AS IDENTITY',
        x,
    )

    x = re.sub(r'(?i)\bSMALLINT\s+UNSIGNED', 'INTEGER', x)
    x = re.sub(r'(?i)\bINTEGER\s+UNSIGNED', 'BIGINT', x)
    x = re.sub(r'(?i)\bBIGINT\s+UNSIGNED', 'NUMERIC(20,0)', x)
    x = re.sub(r'(?i)\bUNSIGNED\b', '', x)

    x = re.sub(r'(?is)\bENUM\s*\([^)]*\)', 'TEXT', x)
    x = re.sub(r'(?i)\bDATETIME\s*\(\s*\d+\s*\)', 'TIMESTAMP', x)
    x = re.sub(r'(?i)\bDATETIME\b', 'TIMESTAMP', x)
    x = re.sub(r'(?i)\bON\s+UPDATE\s+CURRENT_TIMESTAMP\b', '', x)

    x = re.sub(r'\)\s*ENGINE\s*=\s*[^;]+;', ');', x, flags=re.I|re.S)
    x = re.sub(r'\)\s*DEFAULT\s+CHARSET\s*=\s*\w+(\s+COLLATE\s*=\s*\w+)?\s*;', ');', x, flags=re.I|re.S)
    x = re.sub(r'\)\s*CHARSET\s*=\s*\w+(\s+COLLATE\s*=\s*\w+)?\s*;', ');', x, flags=re.I|re.S)
    return x

def _final_pg_cleanup(stmt: str) -> str:
    """
    Guardia final antes de ejecutar:
    - Quita comentarios /* ... */
    - Normaliza INT(n)/SMALLINT(n)/BIGINT(n)/TINYINT(n)
    - Quita espacios raros
    """
    y = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S)
    y = re.sub(r'(?i)\bTINYINT\s*\(\s*1\s*\)', 'BOOLEAN', y)
    y = re.sub(r'(?i)\bTINYINT\s*\(\s*\d+\s*\)', 'SMALLINT', y)
    y = re.sub(r'(?i)\bSMALLINT\s*\(\s*\d+\s*\)', 'SMALLINT', y)
    y = re.sub(r'(?i)\bINT\s*\(\s*\d+\s*\)', 'INTEGER', y)
    y = re.sub(r'(?i)\bBIGINT\s*\(\s*\d+\s*\)', 'BIGINT', y)
    # Por si quedó algún AUTO_INCREMENT suelto
    y = re.sub(
        r'(?i)\b("?[a-zA-Z_]\w*"?\s+)(INTEGER|BIGINT|SMALLINT)[^,]*?\bAUTO_INCREMENT\b',
        r'\1\2 GENERATED BY DEFAULT AS IDENTITY', y,
    )
    # Limpieza final
    y = re.sub(r'\s+', ' ', y).strip()
    return y

def _is_allowed_stmt(stmt: str) -> bool:
    s = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S).strip()
    s = re.sub(r'^\s*(--|#).*$','', s, flags=re.M).strip()
    return bool(re.match(r'(?is)^\s*(CREATE\s+TABLE|INSERT\s+INTO)\b', s))

def _extract_table_from_create(stmt: str) -> str | None:
    """
    Extrae el nombre de la tabla de un CREATE TABLE, tolerando comentarios y db.tbl.
    """
    s = re.sub(r'/\*.*?\*/', '', stmt, flags=re.S)  # quita comentarios
    m = re.search(
        r'(?is)\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        r'(?P<name>(?:"[^"]+"|`[^`]+`|[a-zA-Z_]\w*)(?:\.(?:"[^"]+"|`[^`]+`|[a-zA-Z_]\w*))?)',
        s
    )
    if not m:
        return None
    name = m.group('name')
    if '.' in name:
        name = name.split('.')[-1]
    if (name.startswith('`') and name.endswith('`')) or (name.startswith('"') and name.endswith('"')):
        name = name[1:-1]
    return name

def _transpile_mysql_to_pg(stmt: str) -> str:
    """
    Usa sqlglot para convertir MySQL->PostgreSQL, con limpieza previa y ajustes posteriores.
    """
    pre = _strip_keys_inside_create(_remove_mysql_trailers(stmt))
    try:
        # sqlglot intenta parsear en MySQL y escribir en Postgres
        pg_stmt = sqlglot.transpile(pre, read="mysql", write="postgres")[0]
    except Exception:
        # Si sqlglot no puede con algo muy particular, usamos el original "pre"
        pg_stmt = pre
    return _normalize_post_transpile(pg_stmt)

def _remove_privileges_and_roles(sql_text: str) -> str:
    # Quita GRANT, REVOKE, ALTER ROLE, OWNER TO
    sql_text = re.sub(r'(?im)^\s*GRANT\b.*?;$', '', sql_text)
    sql_text = re.sub(r'(?im)^\s*REVOKE\b.*?;$', '', sql_text)
    sql_text = re.sub(r'(?im)^\s*ALTER\s+ROLE\b.*?;$', '', sql_text)
    sql_text = re.sub(r'(?im)^\s*ALTER\s+SYSTEM\b.*?;$', '', sql_text)
    sql_text = re.sub(r'(?im)^\s*COMMENT\s+ON\b.*?;$', '', sql_text)
    sql_text = re.sub(r'(?im)^\s*OWNER\s+TO\b.*?;$', '', sql_text)
    return sql_text


# --------- Función principal ----------
def import_sql_script(file_path: str, schema: str):
    """
    Sube un .sql (MySQL o PostgreSQL) y ejecuta SOLO CREATE TABLE e INSERT.
    Si parece MySQL, transpila con sqlglot a Postgres.
    """
    engine = get_engine()
    ensure_schema(engine, schema)

    raw_sql = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    raw_sql = _remove_privileges_and_roles(raw_sql)
    if FORBIDDEN.search(raw_sql):
        raise ValueError("El script SQL contiene instrucciones no permitidas.")

    # Limpieza general (MySQL)
    raw_sql = _remove_mysql_global_noise(raw_sql)

    # Quita db.table para que caiga en el search_path
    raw_sql = _strip_db_qualifier(raw_sql)

    # Partimos el archivo en sentencias; filtramos permitidas
    stmts = [s.strip() for s in sqlparse.split(raw_sql) if s and s.strip()]
    stmts = [s for s in stmts if _is_allowed_stmt(s)]

    # ¿Parece MySQL?
    mysqlish = _looks_like_mysql(raw_sql)

    # Transpilación (solo si MySQL); si no, aplicamos un normalizador suave
    converted = []
    for s in stmts:
        if mysqlish:
            converted.append(_transpile_mysql_to_pg(s))
        else:
            # Aunque no parezca MySQL, aplica un pequeño normalizador por si se coló int(11)
            converted.append(_normalize_post_transpile(s))

    # Ejecutar en Postgres, dentro del schema indicado
    with engine.begin() as conn:
        conn.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        conn.exec_driver_sql(f'SET search_path TO "{schema}"')

        for stmt in converted:
            stmt = _final_pg_cleanup(stmt)  # 👈 guardia final

            if re.match(r'(?is)^\s*CREATE\s+TABLE\b', stmt):
                tbl = _extract_table_from_create(stmt)
                if tbl:
                    conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{schema}"."{tbl}" CASCADE')

            conn.exec_driver_sql(stmt)

    # -------- Meta info resultante --------
    with engine.begin() as conn:
        res = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = :schema AND table_type='BASE TABLE'
            ORDER BY table_name
        """), {"schema": schema})
        tables = [r[0] for r in res.fetchall()]

    meta_info = {}
    if tables:
        with engine.begin() as conn:
            for tbl in tables:
                cols = [c[0] for c in conn.execute(text("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema=:schema AND table_name=:table
                    ORDER BY ordinal_position
                """), {"schema": schema, "table": tbl}).fetchall()]
                rows = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{tbl}"')).scalar()
                meta_info[tbl] = {"columns": cols, "rows": rows}

    main_table = next((t for t, info in meta_info.items() if info["rows"] > 0), tables[0] if tables else None)
    return tables, meta_info, main_table

def get_dataset(schema, table):
    engine = get_engine()
    query = f'SELECT * FROM "{schema}"."{table}"'
    df = pd.read_sql(query, engine)
    return df



def read_csv_with_auto_encoding(path):
    # Detectar la codificación con chardet
    with open(path, 'rb') as f:
        rawdata = f.read(10000)
    result = chardet.detect(rawdata)
    encoding = result['encoding'] or 'utf-8'

    # Leer con pandas sin el parámetro errors
    try:
        return pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        # Si falla, intenta con latin1
        return pd.read_csv(path, encoding="latin1")
        
def get_schema_info(schema: str, preview_rows: int = 5):
    engine = get_engine()
    info = {}

    with engine.begin() as conn:
        # Obtener todas las tablas
        res = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = :schema AND table_type='BASE TABLE'
            ORDER BY table_name
        """), {"schema": schema})
        tables = [r[0] for r in res.fetchall()]

        for tbl in tables:
            # Columnas
            cols_res = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema=:schema AND table_name=:table
                ORDER BY ordinal_position
            """), {"schema": schema, "table": tbl})
            columns = [c[0] for c in cols_res.fetchall()]

            # Cantidad de filas
            count_res = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{tbl}"'))
            rows = count_res.scalar()
            # Primeras filas
            preview_data = []
            if rows > 0:
                data_res = conn.execute(text(f'SELECT * FROM "{schema}"."{tbl}" LIMIT {preview_rows}'))
                preview_data = [
                    [row._mapping[c] for c in columns] for row in data_res.fetchall()
                ]


            info[tbl] = {
                "columns": columns,
                "rows": rows,
                "preview": preview_data
            }

    return info


# Configuración de Gemini se hace en cada función que la usa


def reduce_schema(esquema_full: dict) -> dict[str, list[str]]:
    """
    Recibe el dict de get_schema_info({table: {columns, rows, preview}}) y
    devuelve solo {table: [col1, col2, ...]} con strings.
    """
    reducido = {}
    for tbl, info in (esquema_full or {}).items():
        cols = []
        if isinstance(info, dict) and "columns" in info:
            cols = info["columns"]
        elif isinstance(info, (list, tuple)):  # por si ya viene reducido
            cols = info
        reducido[str(tbl)] = [str(c) for c in (cols or [])]
    return reducido

def _extract_json_block(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    s = text.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s:i+1].strip()
    return None

def generar_consulta_y_grafico(esquema_reducido: dict, mensaje_usuario: str):
    # 🔒 Defensa: asegurar que lo que mandamos es reducido y serializable
    esquema_reducido = reduce_schema(esquema_reducido)

    if not mensaje_usuario or mensaje_usuario.strip().lower() in {"hola", "buenas", "hello", "hey"}:
        ejemplos = []
        for tabla, columnas in esquema_reducido.items():
            if len(columnas) >= 2:
                ejemplos.append(f"Total de {columnas[1]} por {columnas[0]} en la tabla {tabla}")
            elif len(columnas) == 1:
                ejemplos.append(f"Conteo de {columnas[0]} en la tabla {tabla}")
            # solo 2 ejemplos para no saturar
            if len(ejemplos) >= 2:
                break

        ejemplos_texto = " o ".join([f"'{e}'" for e in ejemplos]) if ejemplos else "'Conteo de registros por categoría'"        
        return (None, None, f"¡Hola! Dime qué quieres ver. Por ejemplo: {ejemplos_texto}.")

    prompt = f"""
Eres un asistente que genera SQL para PostgreSQL y sugiere un tipo de gráfico.

ESQUEMA (resumido: {{tabla: [columnas...]}}):
{json.dumps(esquema_reducido, ensure_ascii=False)}

INSTRUCCIÓN DEL USUARIO:
\"\"\"{mensaje_usuario}\"\"\"

REGLAS IMPORTANTES:
- Solo usa tablas y columnas que existan en el esquema. Si el nombre pedido no existe, elige la más similar.
- Si usas GROUP BY, todas las demás columnas deben usar agregación (SUM, COUNT, AVG, etc.).
- La consulta debe ser válida en PostgreSQL.
- Si NO tienes suficiente información (falta métrica, dimensión o periodo), NO inventes SQL: devuelve una pregunta de seguimiento en el campo "ask".
- Siempre responde SOLO en JSON válido:

1) Completo:
{{ "sql": "SELECT ...", "grafico": "bar|line|pie|doughnut|scatter", "respuesta": "..." }}

2) Falta info:
{{ "ask": "Pregunta concreta para obtener lo que falta." }}
"""

    # Configurar Gemini con la API key
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    raw_text = getattr(resp, "text", str(resp))

    print("===== RESPUESTA CRUDA DE GEMINI =====")
    print(raw_text)
    print("====================================")

    json_block = _extract_json_block(raw_text)
    if not json_block:
        return (None, None,
                "No tengo suficiente info. Indícame métrica (SUM/COUNT/AVG), dimensión (por mes/categoría) y periodo.")

    try:
        data = json.loads(json_block)
    except Exception:
        return (None, None,
                "La respuesta no fue clara. Dime métrica, dimensión y periodo (p. ej., SUM(total) por mes 2024).")

    if "ask" in data and not data.get("sql"):
        return (None, None, data.get("ask") or "¿Qué métrica, dimensión y periodo necesitas?")

    sql = data.get("sql")
    grafico = data.get("grafico") or "bar"
    respuesta = data.get("respuesta") or "Aquí tienes el gráfico."
    if not sql:
        return (None, None, "Necesito un poco más de detalle: métrica, dimensión y periodo.")
    return (sql, grafico, respuesta)

# ========================================
# NUEVAS FUNCIONES PARA DIAGRAMAS AUTOMÁTICOS
# ========================================

def analizar_datos_archivo(data_source):
    """
    Analiza los datos reales del archivo para determinar qué gráficos son más útiles.
    """
    engine = get_engine()
    schema = data_source.internal_schema
    tabla = data_source.internal_table
    
    analisis = {
        "total_registros": 0,
        "columnas": [],
        "columnas_numericas": [],
        "columnas_categoricas": [],
        "columnas_fecha": [],
        "columnas_unicas": [],
        "patrones_detectados": [],
        "recomendaciones_graficos": []
    }
    
    try:
        with engine.begin() as conn:
            # 1. Información básica
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{tabla}"'))
            analisis["total_registros"] = result.scalar()
            
            # 2. Análisis de columnas
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = :schema AND table_name = :tabla
                ORDER BY ordinal_position
            """), {"schema": schema, "tabla": tabla})
            
            columnas_info = [(row[0], row[1]) for row in result.fetchall()]
            analisis["columnas"] = [col[0] for col in columnas_info]
            
            # 3. Análisis detallado por columna
            for col_name, data_type in columnas_info:
                # Contar valores únicos
                result = conn.execute(text(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{schema}"."{tabla}"'))
                valores_unicos = result.scalar()
                
                col_info = {
                    "nombre": col_name,
                    "tipo": data_type,
                    "valores_unicos": valores_unicos,
                    "es_categorica": valores_unicos <= 20 and valores_unicos > 1,
                    "es_identificador": valores_unicos == analisis["total_registros"],
                    "es_numerica": data_type in ['integer', 'bigint', 'numeric', 'real', 'double precision'],
                    "es_fecha": 'date' in data_type.lower() or 'timestamp' in data_type.lower()
                }
                
                # Detectar patrones de negocio
                col_lower = col_name.lower()
                if any(palabra in col_lower for palabra in ['precio', 'monto', 'costo', 'valor', 'total', 'venta', 'ingreso']):
                    col_info["es_monetaria"] = True
                    analisis["patrones_detectados"].append(f"Columna monetaria detectada: {col_name}")
                
                if col_info["es_categorica"]:
                    analisis["columnas_categoricas"].append(col_info)
                elif col_info["es_numerica"] and not col_info["es_identificador"]:
                    analisis["columnas_numericas"].append(col_info)
                elif col_info["es_fecha"]:
                    analisis["columnas_fecha"].append(col_info)
                elif col_info["es_identificador"]:
                    analisis["columnas_unicas"].append(col_info)
        
        return analisis
        
    except Exception as e:
        print(f"Error analizando datos: {e}")
        return analisis

def analizar_datos_archivo(data_source):
    """
    Analiza inteligentemente los datos para identificar patrones y oportunidades de visualización.
    """
    engine = get_engine()
    schema = data_source.internal_schema
    tabla = data_source.internal_table
    
    analisis = {
        "total_registros": 0,
        "columnas_metricas": [],  # Columnas numéricas útiles para análisis
        "columnas_dimensiones": [],  # Columnas categóricas para agrupar
        "columnas_temporales": [],  # Fechas y timestamps
        "columnas_ignorar": [],  # IDs y columnas sin valor analítico
        "dominio_detectado": None,  # Tipo de dataset detectado
        "metricas_principales": []  # Las métricas más relevantes
    }
    
    try:
        with engine.begin() as conn:
            # Obtener total de registros
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."{tabla}"'))
            analisis["total_registros"] = result.scalar()
            
            if analisis["total_registros"] == 0:
                return analisis
            
            # Obtener información de columnas
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = :schema AND table_name = :tabla
                ORDER BY ordinal_position
            """), {"schema": schema, "tabla": tabla})
            
            columnas_info = [(row[0], row[1]) for row in result.fetchall()]
            
            # Analizar cada columna con inteligencia contextual
            for col_name, data_type in columnas_info:
                col_lower = col_name.lower()
                
                # Obtener estadísticas básicas
                result = conn.execute(text(f'''
                    SELECT 
                        COUNT(DISTINCT "{col_name}") as valores_unicos,
                        COUNT("{col_name}") as valores_no_nulos
                    FROM "{schema}"."{tabla}"
                '''))
                stats = result.fetchone()
                valores_unicos = stats[0]
                valores_no_nulos = stats[1]
                
                # Ratio de unicidad
                ratio_unicidad = valores_unicos / analisis["total_registros"] if analisis["total_registros"] > 0 else 0
                
                # DETECTAR COLUMNAS A IGNORAR
                es_id = any(palabra in col_lower for palabra in ['id', 'codigo', 'key', 'index', 'pk', 'uuid'])
                es_secuencial = False
                
                # Verificar si es una secuencia (1,2,3,4...)
                if data_type in ['integer', 'bigint'] and valores_unicos > 10:
                    result = conn.execute(text(f'''
                        SELECT MIN("{col_name}"), MAX("{col_name}") 
                        FROM "{schema}"."{tabla}"
                        WHERE "{col_name}" IS NOT NULL
                    '''))
                    min_val, max_val = result.fetchone()
                    if min_val is not None and max_val is not None:
                        rango_esperado = max_val - min_val + 1
                        es_secuencial = abs(valores_unicos - rango_esperado) < rango_esperado * 0.1
                
                if es_id or es_secuencial or ratio_unicidad > 0.95:
                    analisis["columnas_ignorar"].append(col_name)
                    continue
                
                # CLASIFICAR COLUMNAS ÚTILES
                
                # 1. TEMPORALES - Máxima prioridad
                if 'date' in data_type.lower() or 'timestamp' in data_type.lower():
                    analisis["columnas_temporales"].append({
                        "nombre": col_name,
                        "tipo": data_type
                    })
                elif any(palabra in col_lower for palabra in ['fecha', 'date', 'año', 'year', 'mes', 'month', 'dia', 'day', 'periodo', 'period']):
                    analisis["columnas_temporales"].append({
                        "nombre": col_name,
                        "tipo": data_type
                    })
                
                # 2. MÉTRICAS - Columnas numéricas con significado
                elif data_type in ['integer', 'bigint', 'numeric', 'real', 'double precision', 'decimal', 'float']:
                    # Detectar métricas de negocio por nomenclatura
                    es_metrica_negocio = any(palabra in col_lower for palabra in [
                        'cantidad', 'amount', 'total', 'suma', 'sum',
                        'precio', 'price', 'costo', 'cost', 'valor', 'value',
                        'venta', 'sale', 'ingreso', 'revenue', 'ganancia', 'profit',
                        'score', 'punto', 'point', 'nota', 'grade', 'calificacion',
                        'duracion', 'duration', 'tiempo', 'time', 'hora', 'hour',
                        'peso', 'weight', 'altura', 'height', 'edad', 'age',
                        'distancia', 'distance', 'velocidad', 'speed', 'temperatura',
                        'porcentaje', 'percentage', 'ratio', 'rate', 'promedio', 'average'
                    ])
                    
                    # Si no es una métrica obvia, verificar si tiene variación significativa
                    if not es_metrica_negocio and valores_unicos > 5:
                        result = conn.execute(text(f'''
                            SELECT STDDEV("{col_name}"), AVG("{col_name}")
                            FROM "{schema}"."{tabla}"
                            WHERE "{col_name}" IS NOT NULL
                        '''))
                        stddev, avg = result.fetchone()
                        if stddev and avg and avg != 0:
                            coef_variacion = abs(stddev / avg)
                            es_metrica_negocio = coef_variacion > 0.1  # Variación significativa
                    
                    if es_metrica_negocio:
                        analisis["columnas_metricas"].append({
                            "nombre": col_name,
                            "tipo": "monetaria" if any(p in col_lower for p in ['precio', 'costo', 'venta', 'ingreso']) else "general",
                            "prioridad": 1 if es_metrica_negocio else 2
                        })
                
                # 3. DIMENSIONES - Columnas categóricas útiles para agrupar
                elif valores_unicos >= 2 and valores_unicos <= 100:
                    # Obtener muestra de valores para análisis semántico
                    result = conn.execute(text(f'''
                        SELECT DISTINCT "{col_name}" 
                        FROM "{schema}"."{tabla}"
                        WHERE "{col_name}" IS NOT NULL
                        LIMIT 5
                    '''))
                    muestra = [str(row[0]) for row in result.fetchall()]
                    
                    # Verificar que no sean valores sin sentido
                    valores_sospechosos = all(
                        len(str(v)) > 20 or  # Valores muy largos
                        str(v).count('-') > 3 or  # Muchos guiones (UUIDs)
                        str(v).count('_') > 3  # Muchos underscores (códigos)
                        for v in muestra if v
                    )
                    
                    if not valores_sospechosos:
                        analisis["columnas_dimensiones"].append({
                            "nombre": col_name,
                            "valores_unicos": valores_unicos,
                            "es_principal": valores_unicos <= 10,  # Dimensiones principales tienen pocos valores
                            "muestra": muestra[:3]
                        })
            
            # DETECTAR DOMINIO DEL DATASET
            analisis["dominio_detectado"] = _detectar_dominio_dataset(columnas_info, analisis)
            
            # PRIORIZAR MÉTRICAS SEGÚN DOMINIO
            analisis["metricas_principales"] = _priorizar_metricas(analisis)
        
        return analisis
        
    except Exception as e:
        print(f"Error analizando datos: {e}")
        return analisis


def _detectar_dominio_dataset(columnas_info, analisis):
    """
    Detecta el tipo de dataset basado en las columnas.
    """
    columnas_lower = [col[0].lower() for col in columnas_info]
    
    # Patrones por dominio
    dominios = {
        "ventas": ['venta', 'sale', 'cliente', 'customer', 'producto', 'product', 'precio', 'price'],
        "academico": ['estudiante', 'student', 'curso', 'course', 'nota', 'grade', 'calificacion', 'score'],
        "rrhh": ['empleado', 'employee', 'salario', 'salary', 'departamento', 'department', 'cargo', 'position'],
        "inventario": ['stock', 'inventory', 'producto', 'item', 'cantidad', 'quantity', 'almacen', 'warehouse'],
        "financiero": ['cuenta', 'account', 'transaccion', 'transaction', 'balance', 'monto', 'amount'],
        "salud": ['paciente', 'patient', 'diagnostico', 'diagnosis', 'tratamiento', 'treatment', 'medico', 'doctor'],
        "marketing": ['campaña', 'campaign', 'conversion', 'click', 'impresion', 'impression', 'ctr', 'roi'],
        "logistica": ['envio', 'shipment', 'pedido', 'order', 'entrega', 'delivery', 'ruta', 'route'],
        "deportes": ['jugador', 'player', 'equipo', 'team', 'partido', 'game', 'gol', 'goal', 'punto', 'score'],
        "investigacion": ['experimento', 'experiment', 'muestra', 'sample', 'resultado', 'result', 'medicion', 'measurement']
    }
    
    mejor_dominio = "general"
    mejor_coincidencias = 0
    
    for dominio, palabras in dominios.items():
        coincidencias = sum(1 for palabra in palabras if any(palabra in col for col in columnas_lower))
        if coincidencias > mejor_coincidencias:
            mejor_coincidencias = coincidencias
            mejor_dominio = dominio
    
    return mejor_dominio


def _priorizar_metricas(analisis):
    """
    Prioriza las métricas más relevantes según el dominio detectado.
    """
    dominio = analisis["dominio_detectado"]
    metricas = analisis["columnas_metricas"]
    
    if not metricas:
        return []
    
    # Prioridades por dominio
    prioridades_dominio = {
        "ventas": ['total', 'venta', 'ingreso', 'cantidad', 'precio'],
        "academico": ['nota', 'calificacion', 'score', 'promedio', 'puntos'],
        "financiero": ['monto', 'balance', 'total', 'valor', 'cantidad'],
        "salud": ['resultado', 'medicion', 'valor', 'duracion', 'cantidad'],
        "deportes": ['puntos', 'goles', 'score', 'tiempo', 'distancia']
    }
    
    palabras_prioritarias = prioridades_dominio.get(dominio, ['total', 'cantidad', 'valor'])
    
    # Ordenar métricas por relevancia
    metricas_ordenadas = []
    for metrica in metricas:
        nombre_lower = metrica["nombre"].lower()
        prioridad = 10  # Prioridad base
        
        for idx, palabra in enumerate(palabras_prioritarias):
            if palabra in nombre_lower:
                prioridad = idx
                break
        
        metricas_ordenadas.append({**metrica, "prioridad_final": prioridad})
    
    metricas_ordenadas.sort(key=lambda x: x["prioridad_final"])
    return metricas_ordenadas[:3]  # Top 3 métricas


def generar_diagramas_automaticos(data_source):
    """
    Genera 3 diagramas diversos y útiles basados en análisis inteligente.
    """
    from .models import Diagrama
    
    print(f"🔍 Analizando: {data_source.name}")
    
    # 1. Análisis inteligente
    analisis = analizar_datos_archivo(data_source)
    
    if analisis["total_registros"] == 0:
        print("❌ Sin datos para analizar")
        return []
    
    print(f"📊 Dataset tipo: {analisis['dominio_detectado']}")
    print(f"📈 {len(analisis['columnas_metricas'])} métricas útiles encontradas")
    print(f"📂 {len(analisis['columnas_dimensiones'])} dimensiones para análisis")
    
    # 2. Generar estrategia de visualización diversa
    graficos_plan = _planificar_graficos_diversos(analisis, data_source)
    
    if not graficos_plan:
        print("⚠️ No se pudieron planificar gráficos útiles")
        return []
    
    # 3. Crear los diagramas
    diagramas_creados = []
    
    for i, plan in enumerate(graficos_plan[:3]):  # Máximo 3 gráficos
        try:
            # Ejecutar SQL
            chart_data = ejecutar_sql_para_chart(plan["sql"])
            
            if not chart_data or not chart_data.get("labels"):
                print(f"⚠️ Sin datos para: {plan['titulo']}")
                continue
            
            # Limitar datos si es necesario
            if len(chart_data["labels"]) > 15 and plan["chart_type"] in ["bar", "line"]:
                chart_data["labels"] = chart_data["labels"][:15]
                chart_data["datasets"][0]["data"] = chart_data["datasets"][0]["data"][:15]
            
            # Crear descripción concisa con hallazgo principal
            descripcion = _generar_descripcion_concisa(chart_data, plan, analisis)
            
            diagrama = Diagrama.objects.create(
                data_source=data_source,
                owner=data_source.owner,
                title=plan["titulo"],
                description=descripcion,
                chart_type=plan["chart_type"],
                source_type=Diagrama.AUTO,
                sql_query=plan["sql"],
                chart_data=chart_data,
                order=i
            )
            
            diagramas_creados.append(diagrama)
            print(f"✅ Creado: {plan['titulo']}")
            
        except Exception as e:
            print(f"❌ Error en {plan.get('titulo', 'gráfico')}: {e}")
            continue
    
    print(f"✨ {len(diagramas_creados)} diagramas útiles creados")
    return diagramas_creados


def _planificar_graficos_diversos(analisis, data_source):
    """
    Planifica 3 gráficos diversos que aporten diferentes perspectivas.
    """
    schema = data_source.internal_schema
    tabla = data_source.internal_table
    graficos = []
    tipos_usados = set()
    
    # GRÁFICO 1: Distribución principal (si hay dimensiones buenas)
    if analisis["columnas_dimensiones"]:
        dim = min(analisis["columnas_dimensiones"], key=lambda x: x["valores_unicos"])
        
        if dim["valores_unicos"] <= 12:
            sql = f'''
                SELECT "{dim['nombre']}" as label, 
                       COUNT(*) as value 
                FROM "{schema}"."{tabla}" 
                WHERE "{dim['nombre']}" IS NOT NULL
                GROUP BY "{dim['nombre']}" 
                ORDER BY value DESC
            '''
            
            graficos.append({
                "titulo": f"Distribución por {_humanizar_nombre(dim['nombre'])}",
                "sql": sql,
                "chart_type": "pie" if dim["valores_unicos"] <= 6 else "bar",
                "tipo_analisis": "distribucion"
            })
            tipos_usados.add("distribucion")
    
    # GRÁFICO 2: Tendencia temporal (si hay fechas)
    if analisis["columnas_temporales"] and "tendencia" not in tipos_usados:
        temp = analisis["columnas_temporales"][0]
        
        # Si hay métricas, usar la principal
        if analisis["metricas_principales"]:
            metrica = analisis["metricas_principales"][0]
            sql = f'''
                SELECT DATE_TRUNC('month', "{temp['nombre']}"::date) as label,
                       SUM("{metrica['nombre']}") as value
                FROM "{schema}"."{tabla}"
                WHERE "{temp['nombre']}" IS NOT NULL
                GROUP BY DATE_TRUNC('month', "{temp['nombre']}"::date)
                ORDER BY label
            '''
            titulo = f"Evolución de {_humanizar_nombre(metrica['nombre'])} por Mes"
        else:
            sql = f'''
                SELECT DATE_TRUNC('month', "{temp['nombre']}"::date) as label,
                       COUNT(*) as value
                FROM "{schema}"."{tabla}"
                WHERE "{temp['nombre']}" IS NOT NULL
                GROUP BY DATE_TRUNC('month', "{temp['nombre']}"::date)
                ORDER BY label
            '''
            titulo = "Actividad por Mes"
        
        graficos.append({
            "titulo": titulo,
            "sql": sql,
            "chart_type": "line",
            "tipo_analisis": "tendencia"
        })
        tipos_usados.add("tendencia")
    
    # GRÁFICO 3: Comparación o Ranking
    if len(graficos) < 3:
        if analisis["columnas_metricas"] and analisis["columnas_dimensiones"]:
            # Comparación métrica por dimensión
            metrica = analisis["metricas_principales"][0] if analisis["metricas_principales"] else analisis["columnas_metricas"][0]
            
            # Buscar dimensión diferente a la ya usada
            dim_disponibles = [d for d in analisis["columnas_dimensiones"] 
                              if not any(d["nombre"] in g["sql"] for g in graficos)]
            
            if dim_disponibles:
                dim = min(dim_disponibles, key=lambda x: x["valores_unicos"])
                
                sql = f'''
                    SELECT "{dim['nombre']}" as label,
                           AVG("{metrica['nombre']}") as value
                    FROM "{schema}"."{tabla}"
                    WHERE "{dim['nombre']}" IS NOT NULL 
                      AND "{metrica['nombre']}" IS NOT NULL
                    GROUP BY "{dim['nombre']}"
                    ORDER BY value DESC
                    LIMIT 10
                '''
                
                graficos.append({
                    "titulo": f"Promedio de {_humanizar_nombre(metrica['nombre'])} por {_humanizar_nombre(dim['nombre'])}",
                    "sql": sql,
                    "chart_type": "bar",
                    "tipo_analisis": "comparacion"
                })
                tipos_usados.add("comparacion")
        
        elif analisis["columnas_dimensiones"] and len(analisis["columnas_dimensiones"]) > 1:
            # Top valores de una dimensión no usada
            dims_no_usadas = [d for d in analisis["columnas_dimensiones"] 
                             if not any(d["nombre"] in g["sql"] for g in graficos)]
            
            if dims_no_usadas:
                dim = dims_no_usadas[0]
                sql = f'''
                    SELECT "{dim['nombre']}" as label,
                           COUNT(*) as value
                    FROM "{schema}"."{tabla}"
                    WHERE "{dim['nombre']}" IS NOT NULL
                    GROUP BY "{dim['nombre']}"
                    ORDER BY value DESC
                    LIMIT 10
                '''
                
                graficos.append({
                    "titulo": f"Top 10 {_humanizar_nombre(dim['nombre'])}",
                    "sql": sql,
                    "chart_type": "bar",
                    "tipo_analisis": "ranking"
                })
    
    # Si aún faltan gráficos y hay métricas, crear análisis estadístico
    if len(graficos) < 3 and analisis["columnas_metricas"]:
        for metrica in analisis["columnas_metricas"]:
            if not any(metrica["nombre"] in g["sql"] for g in graficos):
                sql = f'''
                    SELECT 
                        'Mínimo' as label, MIN("{metrica['nombre']}") as value FROM "{schema}"."{tabla}"
                    UNION ALL
                    SELECT 
                        'Promedio' as label, AVG("{metrica['nombre']}") as value FROM "{schema}"."{tabla}"
                    UNION ALL
                    SELECT 
                        'Máximo' as label, MAX("{metrica['nombre']}") as value FROM "{schema}"."{tabla}"
                '''
                
                graficos.append({
                    "titulo": f"Estadísticas de {_humanizar_nombre(metrica['nombre'])}",
                    "sql": sql,
                    "chart_type": "bar",
                    "tipo_analisis": "estadistica"
                })
                break
    
    return graficos


def _generar_descripcion_concisa(chart_data, plan, analisis):
    """
    Genera una descripción concisa con el hallazgo principal del gráfico.
    """
    labels = chart_data.get("labels", [])
    values = chart_data["datasets"][0].get("data", [])
    
    if not labels or not values:
        return "Visualización de datos"
    
    tipo_analisis = plan.get("tipo_analisis", "")
    
    if tipo_analisis == "distribucion":
        # Encontrar el valor dominante
        max_idx = values.index(max(values))
        total = sum(values)
        porcentaje = (values[max_idx] / total * 100) if total > 0 else 0
        return f"{labels[max_idx]} representa el {porcentaje:.0f}% del total ({values[max_idx]:,} de {total:,} registros)"
    
    elif tipo_analisis == "tendencia":
        # Calcular cambio
        if len(values) > 1:
            cambio = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            direccion = "aumentó" if cambio > 0 else "disminuyó"
            return f"La métrica {direccion} {abs(cambio):.0f}% desde {labels[0]} hasta {labels[-1]}"
        else:
            return f"Valor actual: {values[0]:,}"
    
    elif tipo_analisis == "comparacion" or tipo_analisis == "ranking":
        # Mostrar top 3
        top_items = []
        for i in range(min(3, len(labels))):
            top_items.append(f"{labels[i]}: {values[i]:,.0f}")
        return f"Líderes: {', '.join(top_items)}"
    
    elif tipo_analisis == "estadistica":
        # Mostrar rango
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values) if len(values) > 0 else 0
        return f"Rango: {min_val:,.0f} - {max_val:,.0f} (promedio: {avg_val:,.0f})"
    
    else:
        # Descripción genérica
        return f"Análisis de {analisis['total_registros']:,} registros"


def _humanizar_nombre(nombre_columna):
    """
    Convierte nombre_de_columna en Nombre De Columna.
    """
    # Reemplazar _ y - con espacios
    nombre = nombre_columna.replace('_', ' ').replace('-', ' ')
    
    # Capitalizar cada palabra
    palabras = nombre.split()
    palabras_capitalizadas = []
    
    for palabra in palabras:
        # Mantener acrónimos en mayúsculas
        if palabra.isupper() and len(palabra) <= 4:
            palabras_capitalizadas.append(palabra)
        else:
            palabras_capitalizadas.append(palabra.capitalize())
    
    return ' '.join(palabras_capitalizadas)
    
def generar_diagrama_chat(data_source, mensaje_usuario):
    """
    Genera un diagrama usando el chat, específico para un archivo.
    CORREGIDO: Usa esquema completo y columnas reales.
    Retorna (diagrama_obj, error_msg)
    """
    from .models import Diagrama
    
    if not data_source.internal_schema or not data_source.internal_table:
        return None, "El archivo no tiene datos válidos"
    
    try:
        # Obtener esquema específico del archivo
        esquema_info = get_schema_info(data_source.internal_schema)
        tabla = data_source.internal_table
        info_tabla = esquema_info.get(tabla, {})
        columnas = info_tabla.get("columns", [])
        
        if not columnas:
            return None, "No se pudo analizar la estructura del archivo"
        
        print(f"DEBUG Chat - Esquema: {data_source.internal_schema}")
        print(f"DEBUG Chat - Tabla: {tabla}")
        print(f"DEBUG Chat - Columnas: {columnas}")
        
        # PROMPT CORREGIDO que fuerza uso de esquema completo
        prompt_corregido = f"""
Eres un asistente experto en SQL para PostgreSQL.

INFORMACIÓN OBLIGATORIA:
- Esquema: {data_source.internal_schema}
- Tabla: {tabla}
- Columnas disponibles: {columnas}

REGLAS ESTRICTAS DE SQL:
1. SIEMPRE usar: FROM "{data_source.internal_schema}"."{tabla}"
2. SIEMPRE usar comillas dobles en nombres de columnas: "Category", "Price", "Name"
3. El resultado DEBE tener exactamente: SELECT columna as label, valor as value
4. Solo usar columnas que existen en la lista: {columnas}

MENSAJE DEL USUARIO:
"{mensaje_usuario}"

IMPORTANTE: 
- Si pregunta por "categoría" o "category", usar la columna "Category"
- Si pregunta por "precio" o "price", usar la columna "Price" 
- Si pregunta por "marca" o "brand", usar la columna "Brand"
- Si pregunta por "stock" o "inventario", usar la columna "Stock"

FORMATO DE RESPUESTA (solo JSON válido):
{{
  "sql": "SELECT \"columna_exacta\" as label, COUNT(*) as value FROM \"{data_source.internal_schema}\".\"{tabla}\" WHERE \"columna_exacta\" IS NOT NULL GROUP BY \"columna_exacta\" ORDER BY value DESC LIMIT 15",
  "grafico": "bar|line|pie",
  "respuesta": "Descripción breve del análisis"
}}

Ejemplo correcto:
{{
  "sql": "SELECT \"Category\" as label, COUNT(*) as value FROM \"{data_source.internal_schema}\".\"{tabla}\" GROUP BY \"Category\" ORDER BY value DESC",
  "grafico": "pie",
  "respuesta": "Distribución de productos por categoría"
}}
"""
        
        # Llamar a Gemini con prompt corregido
        import google.generativeai as genai
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt_corregido)
        raw_text = getattr(resp, "text", str(resp))
        
        print("===== RESPUESTA GEMINI CHAT =====")
        print(raw_text)
        print("==================================")
        
        # Extraer JSON
        json_block = _extract_json_block(raw_text)
        if not json_block:
            return None, "No se pudo generar una consulta válida"
        
        try:
            data = json.loads(json_block)
        except json.JSONDecodeError:
            return None, "Respuesta inválida de la IA"
        
        sql = data.get("sql")
        chart_type = data.get("grafico", "bar")
        respuesta = data.get("respuesta", "Análisis generado")
        
        if not sql:
            return None, "No se pudo generar consulta SQL"
        
        print(f"SQL generado: {sql}")
        
        # VALIDAR que el SQL use el esquema correcto
        if f'"{data_source.internal_schema}"."{tabla}"' not in sql:
            # CORREGIR automáticamente si no usa esquema completo
            sql = sql.replace(f' FROM {tabla} ', f' FROM "{data_source.internal_schema}"."{tabla}" ')
            sql = sql.replace(f' FROM ds_', f' FROM "{data_source.internal_schema}"."{tabla}" ')
            print(f"SQL corregido: {sql}")
        
        # Ejecutar SQL y obtener datos
        chart_data = ejecutar_sql_para_chart(sql)
        
        if not chart_data or not chart_data.get("labels"):
            return None, "La consulta no devolvió datos válidos para graficar"
        
        # Generar título basado en el mensaje del usuario
        title = generar_titulo_diagrama(mensaje_usuario, data_source.name)
        
        # Crear diagrama temporal (no guardado aún)
        diagrama = Diagrama(
            data_source=data_source,
            owner=data_source.owner,
            title=title,
            description=respuesta,
            chart_type=chart_type,
            source_type=Diagrama.CHAT,
            sql_query=sql,
            chart_data=chart_data,
            order=data_source.diagramas_count if hasattr(data_source, 'diagramas_count') else 0
        )
        
        return diagrama, None
        
    except Exception as e:
        print(f"Error en generar_diagrama_chat: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error generando diagrama: {str(e)}"

def guardar_diagrama(diagrama):
    """
    Guarda un diagrama temporal en la base de datos.
    """
    try:
        diagrama.save()
        return diagrama
    except Exception as e:
        print(f"Error guardando diagrama: {e}")
        return None

def ejecutar_sql_para_chart(sql_query):
    """
    Ejecuta una consulta SQL y retorna datos en formato Chart.js.
    """
    try:
        engine = get_engine()
        df = pd.read_sql(sql_query, engine)
        
        if df.empty or len(df.columns) < 2:
            return None
        
        # Tomar primeras dos columnas como label y value
        labels_col = df.columns[0]
        values_col = df.columns[1]
        
        labels = df[labels_col].astype(str).tolist()
        values = pd.to_numeric(df[values_col], errors='coerce').fillna(0).tolist()
        
        # Formato Chart.js
        chart_data = {
            "labels": labels,
            "datasets": [{
                "label": values_col,
                "data": values,
                "backgroundColor": [
                    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
                    "#06b6d4", "#f97316", "#84cc16", "#ec4899", "#6366f1"
                ][:len(labels)],
                "borderColor": [
                    "#1e40af", "#dc2626", "#059669", "#d97706", "#7c3aed",
                    "#0891b2", "#ea580c", "#65a30d", "#db2777", "#4f46e5"
                ][:len(labels)],
                "borderWidth": 2
            }]
        }
        
        return chart_data
        
    except Exception as e:
        print(f"Error ejecutando SQL para chart: {e}")
        return None





def generar_titulo_diagrama(mensaje_usuario, archivo_nombre):
    """
    Genera un título descriptivo para el diagrama basado en el mensaje del usuario.
    """
    # Limpiar y truncar mensaje
    mensaje_limpio = re.sub(r'[^\w\s]', '', mensaje_usuario)[:50]
    
    if not mensaje_limpio.strip():
        return f"Análisis de {archivo_nombre}"
    
    # Capitalizar primera letra
    titulo = mensaje_limpio.strip().capitalize()
    
    # Si es muy corto, agregar contexto
    if len(titulo) < 10:
        titulo = f"{titulo} - {archivo_nombre}"
    
    return titulo

def obtener_diagramas_por_archivo(data_source):
    """
    Obtiene todos los diagramas activos de un archivo, ordenados.
    """
    from .models import Diagrama
    
    return Diagrama.objects.filter(
        data_source=data_source,
        is_active=True
    ).order_by('order', 'created_at')

def duplicar_diagrama(diagrama_original, nuevo_titulo=None):
    """
    Crea una copia de un diagrama existente.
    """
    from .models import Diagrama
    
    try:
        nuevo_diagrama = Diagrama.objects.create(
            data_source=diagrama_original.data_source,
            owner=diagrama_original.owner,
            title=nuevo_titulo or f"{diagrama_original.title} (Copia)",
            description=diagrama_original.description,
            chart_type=diagrama_original.chart_type,
            source_type=Diagrama.CHAT,  # Las copias se marcan como del chat
            sql_query=diagrama_original.sql_query,
            chart_data=diagrama_original.chart_data.copy(),
            chart_config=diagrama_original.chart_config.copy(),
            order=diagrama_original.data_source.diagramas_count
        )
        return nuevo_diagrama
    except Exception as e:
        print(f"Error duplicando diagrama: {e}")
        return None

def actualizar_datos_diagrama(diagrama):
    """
    Re-ejecuta la consulta SQL de un diagrama y actualiza sus datos.
    """
    try:
        chart_data = ejecutar_sql_para_chart(diagrama.sql_query)
        
        if chart_data:
            diagrama.chart_data = chart_data
            diagrama.save(update_fields=['chart_data', 'updated_at'])
            return True
        
        return False
        
    except Exception as e:
        print(f"Error actualizando datos de diagrama: {e}")
        return False

def validar_sql_segura(sql_query):
    """
    Valida que la consulta SQL sea segura (solo SELECT).
    """
    sql_limpio = sql_query.strip().upper()
    
    # Debe empezar con SELECT
    if not sql_limpio.startswith('SELECT'):
        return False
    
    # No debe contener comandos peligrosos
    comandos_prohibidos = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
        'GRANT', 'REVOKE', 'TRUNCATE', 'EXECUTE', 'EXEC'
    ]
    
    for comando in comandos_prohibidos:
        if comando in sql_limpio:
            return False
    
    return True

# ---------- Análisis de imagen de gráficas con Gemini ----------
def analyze_chart_image(image_path: str) -> dict:
    """
    Envía una imagen de una gráfica a Gemini y retorna un dict estructurado:
    {
      summary: str,
      insights: [str],
      metrics: [{name: str, value: number, unit: str|null}],
      recommended_charts: [{type: str, reason: str}]
    }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError("No se encontró la imagen a analizar")

    # Configurar Gemini con la API key
    api_key = settings.GEMINI_API_KEY
    print(f"🔑 API Key configurada: {api_key[:10]}..." if api_key else "❌ API Key vacía")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = (
        "Analiza la gráfica de la imagen. Devuelve SOLO JSON con las claves: "
        "summary (string), insights (array de strings), metrics (array de objetos con name, value, unit opcional), "
        "recommended_charts (array de objetos con type y reason). No añadas texto fuera del JSON."
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    resp = model.generate_content([
        {"mime_type": "image/png", "data": image_bytes},
        prompt,
    ])
    raw = getattr(resp, "text", "")

    block = _extract_json_block(raw) or raw
    try:
        data = json.loads(block)
    except Exception:
        data = {"summary": raw.strip()[:500], "insights": [], "metrics": [], "recommended_charts": []}

    data.setdefault("summary", "")
    data.setdefault("insights", [])
    data.setdefault("metrics", [])
    data.setdefault("recommended_charts", [])
    return data

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

        ejemplos_texto = " o ".join([f"“{e}”" for e in ejemplos]) if ejemplos else "“Conteo de registros por categoría”"
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
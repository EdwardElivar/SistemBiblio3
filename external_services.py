import os
import re
import json
import base64
import requests
from openai import OpenAI # pyright: ignore[reportMissingImports]

# Cliente OpenAI usando variable de entorno OPENAI_API_KEY
# client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def get_openai_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    raise ValueError("OPENAI_API_KEY no configurada.")

OPENAI_API_KEY = get_openai_api_key()
client = OpenAI(api_key=OPENAI_API_KEY)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

def _call_openai_for_cover(image_bytes: bytes):
    """Usa OpenAI para intentar extraer titulo / autor / isbn desde la portada."""
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = """
    Eres un asistente para un sistema bibliotecario.
    Tu tarea es leer la portada de un libro (si existe) y devolver datos estructurados.
    IMPORTANTE:
    - Si no estás seguro, deja el campo vacío.
    - NO inventes datos.
    - Solo responde con un JSON valido, sin texto extra.

    Estructura exacta:
    {
      "titulo": string,
      "autor": string,
      "isbn": string
    }
    """

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analiza esta imagen y devuelve los datos del libro si es posible."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                ],
            },
        ],
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    return {
        "titulo": (data.get("titulo") or "").strip(),
        "autor": (data.get("autor") or "").strip(),
        "isbn": (data.get("isbn") or "").replace("-", "").strip()
    }


def limpiar_isbn(isbn: str) -> str:
    """Normaliza ISBN dejando solo dígitos y X; acepta 10 o 13 caracteres."""
    if not isbn:
        return ""
    isbn = isbn.upper()
    isbn = re.sub(r"[^0-9X]", "", isbn)
    if len(isbn) in (10, 13):
        return isbn
    return ""


def buscar_en_google_books(isbn=None, titulo=None, autor=None):
    """Consulta Google Books para completar datos. Prioriza ISBN si existe."""
    if isbn:
        q = f"isbn:{isbn}"
    elif titulo and autor:
        q = f'intitle:"{titulo}" inauthor:"{autor}"'
    elif titulo:
        q = f'intitle:"{titulo}"'
    else:
        return None

    params = {"q": q, "maxResults": 5}
    try:
        r = requests.get(GOOGLE_BOOKS_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    items = data.get("items")
    if not items:
        return None

    # Tomamos el primer resultado; si quieres puedes hacer lógica más fina
    info = items[0].get("volumeInfo", {})

    autores = info.get("authors") or []
    autores_str = ", ".join(autores) if autores else ""

    # Algunos libros no tienen industryIdentifiers bien formados
    isbn_gb = ""
    for ident in info.get("industryIdentifiers", []):
        if ident.get("type") in ("ISBN_13", "ISBN_10"):
            isbn_gb = ident.get("identifier", "").strip()
            break

   # Portada
    image_links = info.get("imageLinks", {}) or {}
    portada = (
        image_links.get("thumbnail")
        or image_links.get("smallThumbnail")
        or ""
    )

    return {
        "titulo": (info.get("title") or "").strip(),
        "autor": autores_str.strip(),
        "isbn": isbn_gb or (isbn or "").strip(),
        "editorial": (info.get("publisher") or "").strip(),
        "anio": int(str(info.get("publishedDate", "0"))[:4]) if info.get("publishedDate") else 0,
        "portada": portada,  # url de la imagen de portada de google books
    }
    

def identificar_libro_por_imagen(image_bytes: bytes):
    """
    1) Usa OpenAI para leer la portada.
    2) Usa Google Books para completar.
    3) Devuelve (data, error) donde data es dict o None.
    """

    # 1️⃣ Extraer posibles datos desde la portada con IA
    ia_data = _call_openai_for_cover(image_bytes)
    if ia_data is None:
        return None, "No se pudo interpretar la portada con IA."

    titulo_ia = ia_data.get("titulo", "")
    autor_ia = ia_data.get("autor", "")
    isbn_ia = ia_data.get("isbn", "")

    # 2️⃣ Consultar Google Books con lo que tengamos
    gb_data = buscar_en_google_books(
        isbn=isbn_ia or None,
        titulo=titulo_ia or None,
        autor=autor_ia or None,
    )

    # 3️⃣ Combinar información (sin pisar buenos datos con vacíos)
    combinado = {
        "titulo": "",
        "autor": "",
        "isbn": "",
        "editorial": "",
        "anio": 0,
        "portada_url": "",
    }

    # Título
    if gb_data and gb_data.get("titulo"):
        combinado["titulo"] = gb_data["titulo"]
    else:
        combinado["titulo"] = titulo_ia

    # Autor
    if gb_data and gb_data.get("autor"):
        combinado["autor"] = gb_data["autor"]
    else:
        combinado["autor"] = autor_ia

    # ISBN
    if gb_data and gb_data.get("isbn"):
        combinado["isbn"] = gb_data["isbn"]
    else:
        combinado["isbn"] = isbn_ia

    # Editorial y año solo vienen de Google Books usualmente
    if gb_data:
        combinado["editorial"] = gb_data.get("editorial", "")
        combinado["anio"] = gb_data.get("anio", 0)

    # 4️⃣ Validar si realmente identificamos un libro
    tiene_algo = any([
        combinado["titulo"],
        combinado["autor"],
        combinado["isbn"],
    ])

    if not tiene_algo:
        # Nada confiable → consideramos libro no identificado
        return None, "Libro no identificado. No se encontraron datos confiables."

    return combinado, None
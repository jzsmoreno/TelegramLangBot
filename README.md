# TelegramLangBot

TelegramLangBot es un bot de Telegram que se conecta a Azure Chat OpenAI para ofrecer conversaciones interactivas y respuestas inteligentes. Este bot está diseñado para proporcionar asistencia y enriquecer la experiencia del usuario en chats.

## Características

- **Conversaciones interactivas**: Mantén diálogos fluidos y naturales.
- **Integración con Azure Chat OpenAI**: Aprovecha la inteligencia artificial para respuestas precisas.
- **Configuración sencilla**: Fácil de implementar y personalizar.

# Cómo ejecutar localmente

Para ejecutar el bot en tu máquina local necesitas:

1. **Python 3.10+** instalado.
2. Un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate.bat
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Crea un archivo `.env` con tus credenciales (ver **Variables de entorno**).
5. Ejecuta el bot:
   ```bash
   python -m TelegramLangBot.main
   ```

## Docker

El proyecto incluye un `Dockerfile` y un `docker-compose.yml`. Para construir y lanzar la imagen:

```bash
# Construye la imagen (solo la primera vez)
docker build -t telegram-lang-bot .

# O usa docker compose (concreta la construcción si es necesario)
docker compose up --build
```

El contenedor leerá las variables de entorno desde un archivo `.env` que debe estar en el mismo directorio que `docker-compose.yml`.

## Variables de entorno

| Variable | Descripción |
| -------- | ----------- |
| `OPENAI_API_KEY` | Clave API de OpenAI. |
| `AZURE_ENDPOINT` | URL del endpoint de Azure OpenAI. |
| `API_VERSION` | Versión de la API (ej. `2023-03-15-preview`). |
| `TOKEN` | Token del bot de Telegram. |

Puedes usar el archivo `.env.example` como plantilla.

## Desarrollo y Extensión

### Añadir nuevos comandos
1. Define una función async con la firma `(update: Update, context: ContextTypes.DEFAULT_TYPE)`.
2. Decora con `@restricted` si quieres limitarlo a los usuarios administradores.
3. Registra el handler:
   ```python
   command_handler = CommandHandler("nombre_comando", tu_función)
   application.add_handler(command_handler)
   ```

### Chain‑of‑Thought (CoT)
El bot soporta un modo CoT que se activa con `/cot on` y desactiva con `/cot off`. Cuando está activado, la respuesta incluye una explicación paso a paso antes de dar la conclusión.

### Pruebas unitarias
Puedes crear tests en `tests/`. Un ejemplo básico:
```python
# tests/test_main.py
import pytest
from telegram import Update

@pytest.mark.asyncio
async def test_get_user_id():
    # Mock Update y Context aquí...
    pass
```
Ejecuta con `pytest`.

## Contribuir
Si quieres contribuir, por favor abre un PR. Asegúrate de que las pruebas pasen y que la documentación esté actualizada.

---

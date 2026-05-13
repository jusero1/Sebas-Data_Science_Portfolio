# Política de Seguridad

## Versiones soportadas

| Versión | Soporte activo |
|---------|---------------|
| `main`  | ✅ Sí |

## Reportar una vulnerabilidad

**NO abras un issue público para reportar vulnerabilidades de seguridad.**

Envía un correo a **jssegurarod96@gmail.com** con el asunto `[SECURITY] <descripción breve>` incluyendo:

1. Descripción detallada de la vulnerabilidad.
2. Pasos para reproducirla (PoC si es posible).
3. Impacto potencial estimado.
4. Versión o commit afectado.

Recibirás acuse de recibo en menos de 72 horas. Las vulnerabilidades válidas serán corregidas en un plazo razonable y se te acreditará en el changelog (a menos que prefieras anonimato).

---

## Prácticas de seguridad implementadas

### Gestión de secretos
- Ningún secreto, API key ni contraseña está hardcodeado en el código.
- Todos los valores sensibles se leen desde variables de entorno (`os.getenv`).
- Se provee un `.env.example` para cada proyecto; el `.env` real está en `.gitignore`.

### Dependencias
- Las versiones de dependencias están fijadas exactamente en `requirements.txt`.
- Se recomienda ejecutar `pip-audit -r requirements.txt` antes de cada despliegue.
- Dependabot está configurado para recibir alertas de CVEs automáticamente.

### Contenedores Docker
- Imágenes base oficiales `python:3.11-slim` (mínima superficie de ataque).
- Construcción multi-etapa: el artefacto de runtime no incluye herramientas de build.
- La aplicación se ejecuta con un usuario sin privilegios (`uid=1000`, `gid=1000`).
- `HEALTHCHECK` definido en todos los Dockerfiles.
- `.dockerignore` excluye datos, modelos, archivos `.env` y código de tests.

### APIs (FastAPI)
- Validación estricta de entradas con Pydantic v2 (tipos, rangos, tamaños).
- CORS configurado con origins explícitos (no `*`), leído desde variable de entorno.
- Rate limiting en endpoints de inferencia con `slowapi` (10 req/min por IP).
- Cabeceras de seguridad HTTP en todas las respuestas:
  `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`,
  `Strict-Transport-Security`, `Referrer-Policy`.
- Los mensajes de error en producción no revelan detalles internos del sistema.

### Integridad del código
- `pre-commit` con hooks: `ruff`, `black`, `nbstripout`, detección de claves privadas.
- Los notebooks se almacenan **sin outputs** en el repositorio (`nbstripout`).
- CI/CD con GitHub Actions ejecuta linting, tests y escaneo de dependencias en cada push.

---

## Responsabilidad

Este repositorio es un portafolio educativo. No procesa datos reales de usuarios.
Los datasets usados son públicos (UCI, Kaggle) con licencias de uso académico/investigación.

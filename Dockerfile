# Carbon Capture screening app.
#
# Plain Dockerfile (not nixpacks): deterministic system dependencies for
# WeasyPrint, with a build-time smoke test so a broken PDF stack fails
# the BUILD loudly instead of failing silently on the user's click.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries for WeasyPrint (pango/cairo/harfbuzz text stack) plus a
# real font so the memo PDF renders text instead of blank boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libharfbuzz-subset0 \
    libffi8 \
    libglib2.0-0 \
    shared-mime-info \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Loud build-time check: WeasyPrint must load its native libraries here,
# in the same image that will serve traffic.
RUN python -c "from weasyprint import HTML; print('weasyprint native libs ok')"

COPY . .

EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]

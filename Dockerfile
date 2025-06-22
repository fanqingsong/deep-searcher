FROM m.daocloud.io/docker.io/library/python:3.10

WORKDIR /app

# Create directories
RUN mkdir -p /app/data /app/logs

# Configure pip to use Chinese mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Copy all necessary files for installation
COPY pyproject.toml ./
COPY LICENSE README.md ./
COPY deepsearcher/ ./deepsearcher/
COPY main.py ./
COPY examples/ ./examples/

# Install the project in editable mode with all dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/docs')"

CMD ["python", "main.py", "--enable-cors", "true"] 
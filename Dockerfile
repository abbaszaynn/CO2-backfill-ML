FROM python:3.11-slim
# In mainland China, build with a PyPI mirror, e.g.
#   docker build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple -t co2-backfill .
ARG PIP_INDEX_URL=https://pypi.org/simple
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -i "$PIP_INDEX_URL" -r requirements.txt
COPY app.py backfill_model.py co2_backfill_train.csv ./
COPY .streamlit .streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

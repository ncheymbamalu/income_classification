FROM python:3.9.13-slim

RUN mkdir income_classification
RUN cd income_classification

WORKDIR income_classification

ADD . .

RUN pip3 install -r requirements.txt -q

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
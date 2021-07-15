FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./api /api/api

RUN pip install transformers networkx tqdm torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

ENV PYTHONPATH /api/api
WORKDIR /api/api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0"]
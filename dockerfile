FROM python:3.8-buster
COPY . /app
EXPOSE 5000
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python","app.py"]
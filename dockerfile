FROM python:3.8-buster
COPY . /app
EXPOSE 5000
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip uninstall -y opencv-python
CMD ["python","app.py"]
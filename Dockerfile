# set base image (host OS)
FROM python:3.8

# set the working directory in the container
#WORKDIR /code
WORKDIR /app

# copy the dependencies file to the working directory
#COPY requirements.txt .
#COPY ./requirements.txt requirements.txt
COPY ./production/requirements.txt /app/requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the directory to the working directory
#COPY . .
COPY ./production /app

EXPOSE 8080

# command to run on container start
#CMD [ "python", "./app.py" ]
CMD [ "python", "./app.py" ]
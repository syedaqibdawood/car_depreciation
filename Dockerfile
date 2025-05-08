# Using an official lightweight Python image
FROM python:3.10-slim

# Setting work directory
WORKDIR /app

# Copying all files to the container
COPY . .

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Exposing the port Flask runs on
EXPOSE 5000

# Running the app
CMD ["python", "application.py"]
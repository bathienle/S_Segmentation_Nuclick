FROM python:3.8

# Create the directories
RUN mkdir -p weights/ app/

# Install the dependencies
RUN pip3 install numpy==1.20.1 scipy==1.6.1 matplotlib==3.3.4 torch==1.8.0 \
    torchvision==0.9.0 pillow==8.1.0 scikit-image==0.18.1 pandas==1.2.2

# Install Cytomine and the master thesis
RUN git clone https://github.com/cytomine-uliege/Cytomine-python-client.git
RUN git clone https://github.com/bathienle/master-thesis-code.git

WORKDIR /master-thesis-code/
RUN pip install .

WORKDIR /Cytomine-python-client/
RUN git checkout tags/v2.8.0 && pip install .

# Clean
RUN rm -r /master-thesis-code/ /Cytomine-python-client/

# Add the weight of the model
ADD weights/ /weights/

# Add the scripts
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]

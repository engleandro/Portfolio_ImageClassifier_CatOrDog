FROM python:3.10.0

LABEL AUTHOR="Leandro Alves <alves.engleandro@gmail.com>"
LABEL VERSION="0.0.0"

ENV WORK_DIR /usr/app
ENV IMAGES_URL https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
ENV ANNOTATIONS_URL https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
ENV DATASET_DIR /usr/app/dataset

RUN echo "Creating the Image Classifier image..."

WORKDIR /usr/app

COPY . .

#RUN make install

RUN apt-get update
#&& apt-get install -y wget
#&& apt-get install -y nvidia-cuda-toolkit

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r ./setup/packages

#RUN wget -P ${DATASET_DIR} ${IMAGES_URL}
#RUN wget -O ${DATASET_DIR}/images.tar.gz ${IMAGES_URL}
#RUN wget ${IMAGES_URL} --directory-prefix=${DATASET_DIR}
#RUN tar -xf ${DATASET_DIR}/images.tar.gz -C ${DATASET_DIR}

#RUN wget -P ${DATASET_DIR} ${ANNOTATIONS_URL}
#RUN wget -O ${DATASET_DIR}/annotations.tar.gz ${ANNOTATIONS_URL}
#RUN wget ${ANNOTATIONS_URL} --directory-prefix=${DATASET_DIR}
#RUN tar -xf ${DATASET_DIR}/annotations.tar.gz -C ${DATASET_DIR}

RUN export LD_LIBRARY_PATH=/usr/local/lib

VOLUME ["/model", "/logs"]

EXPOSE 6006
#EXPOSE 80

RUN python3 app.py

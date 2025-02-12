FROM continuumio/anaconda3:latest

RUN apt-get update \
    && apt-get install -y wget default-jre \
    && apt-get install -y build-essential \
    && git clone https://github.com/samleung314/BERN2.git \
    && conda env create -f ./BERN2/environment.yml

# Make RUN commands use the conda environment:
SHELL ["conda", "run", "-n", "bern2", "/bin/bash", "-c"]
RUN gcloud components install core gsutil gcloud-crc32c --quiet \
    && gcloud components update --quiet

WORKDIR /BERN2
RUN pip install -r requirements.txt
    
RUN mkdir -p ./resources \
    && gcloud storage cp -r gs://bern2-resources/* ./resources

WORKDIR /BERN2/resources/GNormPlusJava/CRF
RUN chmod +x ./configure \
    && ./configure --prefix="$HOME" \
    && make \
    && make install

WORKDIR /BERN2/scripts
RUN chmod +x ./run_bern2_cpu.sh
    
EXPOSE 8888
    
ENTRYPOINT ["conda", "run", "-n", "bern2", "/bin/bash", "-c"]
CMD ["./run_bern2_cpu.sh"]
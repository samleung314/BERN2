# stage: download /resources from gcs
FROM google/cloud-sdk:alpine AS downloader_stage
WORKDIR /resources
RUN gcloud storage cp -r gs://bern2-resources/* .

# stage: build service image
FROM continuumio/anaconda3:latest

RUN apt-get update \
    && apt-get install -y wget default-jre \
    && apt-get install -y build-essential

RUN git clone https://github.com/samleung314/BERN2.git \
    && conda env create -f ./BERN2/environment.yml

# Make RUN commands use the conda environment:
SHELL ["conda", "run", "-n", "bern2", "/bin/bash", "-c"]

WORKDIR /BERN2
RUN pip install -r requirements.txt

COPY --from=downloader_stage /resources /BERN2/resources

WORKDIR /BERN2/resources/GNormPlusJava
RUN tar -zxvf CRF++-0.58.tar.gz \
    && mv CRF++-0.58 CRF \
    && cd CRF \
    && ./configure --prefix="$HOME" \
    && make \
    && make install

WORKDIR /BERN2/scripts
RUN chmod +x ./run_bern2_cpu.sh
    
EXPOSE 8888
    
ENTRYPOINT ["conda", "run", "-n", "bern2", "/bin/bash", "-c"]
CMD ["./run_bern2_cpu.sh"]
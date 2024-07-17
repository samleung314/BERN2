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

# COPY resources_v1.1.b.tar.gz .

# RUN tar -zxvf resources_v1.1.b.tar.gz \
#     && rm -rf resources_v1.1.b.tar.gz

RUN mkdir -p ./resources \
    && gsutil -m cp -r gs://bern2-resources/* ./resources

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

# Expose port 8888 and mount /resources
# docker run -it \
#     -v /home/jupyter/BERN2/resources/:/app/BERN2/resources \
#     -p 8888:8888 \
#     bern2:latest /bin/bash
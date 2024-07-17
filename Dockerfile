FROM continuumio/anaconda3:latest

RUN apt-get update \
    && apt-get install -y wget default-jre \
    && apt-get install -y build-essential
    
WORKDIR /app
COPY ./environment.yml ./
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bern2", "/bin/bash", "-c"]

# RUN conda init bash \
# # && echo "source activate bern2" > ~/.bashrc \
#     && . ~/.bashrc

# RUN conda create -n bern2 python=3.7 -y \
#     && conda activate bern2 && \
#     && conda install pytorch==1.9.0 cudatoolkit=10.2 -c pytorch -y \
#     && conda install faiss-gpu libfaiss-avx2 -c conda-forge -y \
#     && conda install conda-forge::gcc -y

RUN git clone https://github.com/samleung314/BERN2.git \
    && cd ./BERN2 \
    && pip install -r requirements.txt

WORKDIR /app/BERN2
COPY resources_v1.1.b.tar.gz .

RUN tar -zxvf resources_v1.1.b.tar.gz \
    && rm -rf resources_v1.1.b.tar.gz \
    && cd resources/GNormPlusJava \
    && mv CRF++-0.58 CRF \
    && cd CRF \
    && ./configure --prefix="$HOME" \
    && make \
    make install

WORKDIR /app/BERN2/scripts
RUN chmod +x run_bern2_cpu.sh
    
EXPOSE 8888
    
ENTRYPOINT ["bash", "-c"]
CMD ["./run_bern2_cpu.sh"]

# Expose port 8888 and mount /resources
# docker run -it \
#     -v /home/jupyter/BERN2/resources/:/app/BERN2/resources \
#     -p 8888:8888 \
#     bern2:latest /bin/bash
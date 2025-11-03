FROM qiime2/core:2020.8
LABEL maintainer="jwlee230@unist.ac.kr"

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get update --allow-releaseinfo-change && apt-get upgrade -y && apt-get install -y make gcc g++ zlib1g-dev libbz2-dev liblzma-dev libncurses5-dev unzip libcurl4-openssl-dev libfontconfig1 cabextract && wget https://www.freedesktop.org/software/fontconfig/webfonts/webfonts.tar.gz && tar -zxvf webfonts.tar.gz && cd msfonts && cabextract *.exe && mkdir -p ~/.local/share/fonts && cp *.ttf *.TTF ~/.local/share/fonts/ && mkdir /Tools

# Python
ADD requirements.txt /
RUN pip install --no-cache-dir --requirement /requirements.txt

# R
RUN mkdir -p $HOME/.R && echo "MAKEFLAGS = -j" > $HOME/.R/Makevars && conda deactivate &&conda create -n ancombc_env -c conda-forge -c bioconda r-base bioconductor-ancombc -y

CMD ["/bin/bash"]

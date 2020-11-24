FROM qiime2/core:2020.8
LABEL maintainer="jwlee230@unist.ac.kr"

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

RUN apt-get update && apt-get upgrade -y && apt-get install -y make gcc g++ zlib1g-dev libbz2-dev liblzma-dev libncurses5-dev unzip libcurl4-openssl-dev libfontconfig1
RUN mkdir /Tools

# PIP
ADD requirements.txt /
RUN pip install --requirement /requirements.txt

CMD ["/bin/bash"]

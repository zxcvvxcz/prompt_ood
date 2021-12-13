FROM zxcvvxcz/nv-prompt:base
LABEL "purpose"="nsml"
LABEL maintainer="pch330 <pch330@snu.ac.kr>, <pch330@europa.snu.ac.kr>"
# ADD ~/share/naver_project/prompt_ood/.cache/huggingface/transformers /root/.cache/huggingface/transformers
COPY . /root
WORKDIR /root
RUN ["/bin/bash", "-c", "pwd"]
# CMD python main.py


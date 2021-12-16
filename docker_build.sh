# restore ownership
docker run --rm -it \
    --entrypoint /bin/sh \
    -e HOST_UID=`id -u` \
    -v /home/pch330/prompt_ood:/root \
    zxcvvxcz/nsml-cuda10.2:base \
    -c 'chown -R ${HOST_UID}:${HOST_UID} /root'

docker build -t zxcvvxcz/nsml-cuda10.2:gpt2-large -f Dockerfile .

# make container
docker stop test_container1
docker rm test_container1
docker run -itd --name test_container1 -p 8887:8887 --ipc=host --gpus all --restart=always zxcvvxcz/nsml-cuda10.2:gpt2-large
docker exec -it test_container1 bash
# docker run -itd --name prompt_nsml -p 8888:8888 --ipc=host -v /home/pch330/share/naver_project/prompt_ood:/root --gpus all --restart=always zxcvvxcz/nv-prompt:base

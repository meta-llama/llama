# LLaMA Docker Playground

A "Clean and Hygienic" LLaMA Playground, Play LLaMA with 10GB or 20GB of VRAM.

## How to use

To use this project, we need to do **two things**:

- the first thing is to download the model
  - (you can download the LLaMA models from anywhere)
- and the second thing is to build the image with the docker
  - (saves time compared to downloading from Docker Hub)

### Put the Models File in Right Place

Taking the smallest model as an example, you need to place the model related files like this:

```bash
.
└── models
    ├── 30B
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   ├── consolidated.02.pth
    │   ├── consolidated.03.pth
    │   └── params.json
    ├── 13B
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   └── params.json
    ├── 7B
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model
```

### Build the LLaMA Docker Playground

If you prefer to use the official authentic model, build the docker image with the following command:

```bash
docker build -t soulteary/llama:llama . -f docker/Dockerfile.llama
```

If you wish to use a model with lower memory requirements, build the docker image with the following command:

```bash
docker build -t soulteary/llama:pyllama . -f docker/Dockerfile.pyllama
```

If you wish to use a model with **the minimum memory** requirements, build the docker image with the following command:

```bash
docker build -t soulteary/llama:int8 . -f docker/Dockerfile.int8
```


### Play with the LLaMA

For official model docker images, use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -v `pwd`/models:/app/models -p 7860:7860 -it --rm soulteary/llama:llama
```

For lower memory requirements docker images, use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -v `pwd`/models:/llama_data -p 7860:7860 -it --rm soulteary/llama:pyllama
```

For **the minimum memory** requirements docker images, use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -v `pwd`/models:/llama_data -p 7860:7860 -it --rm soulteary/llama:int8
```


## Credits

- [facebookresearch/llama](https://github.com/facebookresearch/llama)
- [andrewssobral's pr](https://github.com/facebookresearch/llama/pull/126/files)
- [juncongmoo/pyllama](https://github.com/juncongmoo/pyllama)
- [tloen/llama-int8](https://github.com/tloen/llama-int8)

## License

Follow the rules of the game and be consistent with the original project.

# baseLess: minION sequence detection, no basecalling required

## Running on Nvidia Jetson Nano (2GB)
The Nvidia Jetson Nano is a single-board computer with dedicated GPU. Designed to be baseLess is light enough to run on 

Before first usage open a terminal in the baseless directory and build the image:

```shell
docker build -f Dockerfile_jetson -t baseless .
```

open the docker container interactively:
```shell
docker run -it --rm --runtime nvidia --network host baseless
```
Add any directories you may want to read/write by adding `-v /local/dir:/home/some_dir_name `

You can now start running inference! As the Jetson Nano has little memory to spare (2GB or 4GB, depending on your model), you may need to limit memory using the `--mem` option. We found that the 2GB model runs well with 512mb (`--mem 512`).

## Jetson nano tips
- To free up some memory for smoother running, switch off the GUI -- `sudo systemctl stop lxdm` or `sudo sysctl disable lxdm` to prevent the GUI starting at startup. You can also follow Zafer Arican's [advice](https://www.zaferarican.com/post/how-to-save-1gb-memory-on-jetson-nano-by-installing-lubuntu-desktop) and switch to lubuntu.
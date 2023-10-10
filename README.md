# Configure Docker Desktop
If you are using Docker Desktop, change the allocated disk space to **200GB** in the Docker preferences (Preferences -> Resources -> Disk).

# Run main experiment
1. Unzip test folder
```
mkdir test
tar -I pigz -xf test.tar.gz --directory tests
```
Ignore warnings like below:
```
tar: Ignoring unknown extended header keyword 'SCHILY.fflags'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.FinderInfo'
tar: Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.lastuseddate#PS'
...
```

2. Run the main experiment
```
make main
```

3. Plot the experiment results
```
python plot.py
```

4. Export the plots to host machine

First get the name of the container running under the NAMES column:
```
$ docker ps
CONTAINER ID   IMAGE      COMMAND       CREATED          STATUS          PORTS     NAMES
e705b63f2247   arborist   "/bin/bash"   37 minutes ago   Up 37 minutes             peaceful_hertz
```
Here the name is `peaceful_hertz`.

Then, copy the generated figures from the container to your host machine.
Here, we show how to copy the figures to the Downloads folder.
```
docker cp peaceful_hertz:/workspace/figures ~/Downloads/figures
```

# Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f test.tar.gz tests
```

# Building Docker for Multi-Platform

```
docker buildx build --load --platform linux/amd64,linux/arm64 -t arborist .
```

# Build for mac
```
docker buildx build --load --platform linux/arm64 -t arborist .
```

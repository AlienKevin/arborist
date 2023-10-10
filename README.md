# Configure Docker Desktop
If you are using Docker Desktop, change the allocated disk space to **200GB** in the Docker preferences (Preferences -> Resources -> Disk).

# Run main experiment
1. Pull and run the docker container for arborist:
```
docker run -ti alienkevin/arborist
```

2. Unzip test folder
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

3. Run the main experiment
```
make main
```

4. Plot the experiment results
```
python plot.py
```

5. Export results to host machine

    a. First get the name of the container running under the NAMES column:
    ```
    $ docker ps
    CONTAINER ID   IMAGE      COMMAND       CREATED          STATUS          PORTS     NAMES
    e705b63f2247   arborist   "/bin/bash"   37 minutes ago   Up 37 minutes             peaceful_hertz
    ```
    Here the name is `peaceful_hertz`.

    b. Then, copy the generated figures from the container to your host machine.
    Here, we show how to copy the figures to the Downloads folder.
    ```
    docker cp peaceful_hertz:/workspace/figures ~/Downloads/figures
    ```

    c. You can also copy the generated result spread sheets to your host machine for inspection.
    ```
    docker cp peaceful_hertz:/workspace/benchmark_summary.csv ~/Downloads/benchmark_summary.csv
    ```


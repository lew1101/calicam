# calicam

Generates projection matrix and calculates intrinsic and extrinsic parameters. This project was made for my [IB Math AA HL Extended Essay](https://github.com/lew1101/Fundamentals-of-Camera-Calibration).

## Cloning the Repo

Make sure you have `git` installed on your system. Then, clone the repo by command line:

```shell
git clone https://github.com/lew1101/calicam.git
```

## Installing Dependencies

Make sure you have `python3` (version 3.9 or higher) installed on your system. Then, install dependencies using `pip3`.

```shell
cd calicam

pip3 install -r requirements.txt
```

## Run

To run the program, simply execute `run.py` with arguments passed.

**Basic usage**:

```shell
python3 run.py calibration_points.csv -d data.csv -g
```

## Options

**Usage**: `calicam [-h] [-d DATA_PATH] [-g [BKGD_IMG]] [-t TITLE] [-s] [-o GRAPH_PATH] [--noprint] PATH`

**Options**:
    - `-h`, `--help` – show help message
    - `-d` _`DATA_PATH`_, `--data` _`DATA_PATH`_ – path to `csv` file with model verification data. CSV inputs are in the format: x,y,z,u,v where 3D point = (x, y, z) and 2D point = (u,v)
    - `-g` _`[BKGD_IMG]`_, `--graph` _`[BKGD_IMG]`_ – generate graph
    - `-t` _`TITLE`_, `--title` _`TITLE`_ – title of graph (ignored if `-g` is not passed)
    - `-s`, `--show` – show graph (only necessary if `-o` is passed)
    - `-o` _`GRAPH_PATH`_, `--out` `GRAPH_PATH` – graph output location
    - `--noprint` – don't print output

## Author

[Kenneth Lew](https://github.com/lew1101)

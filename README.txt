The entirety of the project is described in the semester project pdf. But here is a short version.

The purpose of this program is to generate pretty plots from the output of COSMO models.
It does this by reading in .nc files directly.

The main components of the program are the  visualize_climate_simulation.py, Visualization.py, VisParam.py and a json file.

Running the program functions in the following way:
visualize_climate_simulation.py pathToJsonFile.json


It is assumef that the nc files are named in the following way:
    "lffd" + YYYYmmddHHMMSS + ".nc"


to generate the video run this command in the folder where you generated the pictures
ffmpeg -framerate 20  -pattern_type glob -i '*.jpg' \
  -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" NAMEOFVIDEO.mp4


I recommend to set up a new virtual environment for python to run. The most easy way to this is using miniconda and the provided .yml files

This can also be done manually by installing the following packages with all their dependencies:

- xarray
- matplotlib
- django (I just it to parse dates, there might be something more lightweight out there)
- numpy
- h5netcdf
- cartopy
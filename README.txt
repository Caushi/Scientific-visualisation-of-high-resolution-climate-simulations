The purpose of this program is to generate pretty plots from the output of COSMO models.
It does this by reading in .nc files directly.

The main components of the program are the animate_Claudio.py, VisParam.py and a json file.
I recommend to set up a new virtual environment for your python to run. Install the following packages with all their dependencies:
- xarray
- matplotlib
- django (I just it to parse dates, there might be something more lightweight out there)
- numpy
- h5netcdf

Running the program functions in the following way:
python animate_Claudio.py pathToJsonFile.json


!!! THIS PROGRAM IS STILL UNDER ACTIVE DEVELOPMENT !!!
This means things are subject to change and might not work as intended.

STILL MISSING:
- choosing which variables to plot
- generating an animation automatically during the script
- hail and  lightning
- projection of the map
- automatic selection of soil lvls


I assume that the nc files are named in the follwoing way:
    "lffd" + YYYYmmddHHMMSS + ".nc"


to generate the video run this command in the folder where you generated the pictures
ffmpeg -framerate 20  -pattern_type glob -i '*.jpg' \
  -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" NAMEOFVIDEO.mp4

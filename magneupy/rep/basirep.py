import tempfile
import sh
from sh import basireps_mac
basireps = basireps_mac

tempfile.tempdir = '/var/tmp'

#open a temporary directory https://docs.python.org/3/library/tempfile.html
# use context manager to open three files for smb, fp, and bsr
# run the basireps command to write to files
# get neccessary info from those files
# exit context manager so the files close
# delete the temperorary directory


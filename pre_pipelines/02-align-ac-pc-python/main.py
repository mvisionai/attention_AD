#!/usr/bin/python

import sys, getopt
import os
import subprocess
import shlex
import shutil


def detectACPCProgram():
    try:
        cmd = subprocess.Popen(["acpcdetect", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (_, _) = cmd.communicate()
        return True
    except:
        return False

def runACPCDetect(niifile):
    try:
        input = "acpcdetect -no-tilt-correction -center-AC -nopng -noppm -i " + niifile
        args = shlex.split(input)
        cmd = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (stdoutdata, _) = cmd.communicate()
        return (True, None)
    except:
        return (False, stdoutdata)

def createDestinationDir(destinationdir):
    try: 
        if not os.path.exists(destinationdir):
            os.mkdir(destinationdir)
        else:
            shutil.rmtree(destinationdir)
            os.mkdir(destinationdir)
    except (IOError, OSError) as e:
        print(e.errno)
        print("error creating directory: ", destinationdir)
        sys.exit(2)

def createTmpFile(sourcefile, tmpFile):
    try:
        shutil.copyfile(sourcefile, tmpFile)
    except (IOError, OSError) as e:
        print(e.errno)
        print("error copying FROM: ", sourcefile, " TO: ", tmpFile)
        sys.exit(2)


def main(argv):
    inputFileText = ''
    try:
        opts, _ = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('main.py -i <inputFileText>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputFileText>')
            sys.exit()
        elif opt in ("-i", "--ifiletext"):
            inputFileText = arg

    if not os.path.isfile(inputFileText):
        print('the input is not a file')
        sys.exit(2)
    
    existsACPC = detectACPCProgram()
    if not existsACPC:
        print("Unable to check acpcdetect version")
        sys.exit(2)
    else:
        with open(inputFileText, 'r') as fp:
            #content = f.readlines()
            line = fp.readline()
            #for f in content:
            while line:
                # create tmp dir
                filename = os.path.basename(line.strip())
                basedir = os.path.dirname(line.strip())
                destinationdir = os.path.join(basedir, "0-AC-PC-Aligned")
                createDestinationDir(destinationdir)
                
                # create tmp file
                sourcefile = os.path.join(basedir, filename)
                tmpFile = os.path.join(destinationdir, filename)
                createTmpFile(sourcefile, tmpFile)

                # do the AC-PC alignment
                (result, msg) = runACPCDetect(tmpFile)
                if not result:
                    print("An error has ocurred using acpcdetect program")
                    print(msg)
                    sys.exit(2)
                os.remove(tmpFile)
                line = fp.readline()


if __name__ == "__main__":
    main(sys.argv[1:])
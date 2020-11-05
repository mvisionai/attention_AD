#!/usr/bin/python

import sys, getopt
import os
import shutil

def readAndCopyFilesToDestination(inputDir, outputDir):
    # creating destinations folders
    counter = 0
    for root,_,f_names in os.walk(inputDir):        
        for f in f_names:
            if f.lower().find("brain_") > -1:
                sourcefile = os.path.join(root, f)
                destinationfile = os.path.join(outputDir, 'NC_' + str(counter) + '.nii')

                try: 
                    if not os.path.exists(outputDir):
                        os.mkdir(outputDir)
                    shutil.copyfile(sourcefile, destinationfile)
                    counter = counter + 1
                except (IOError, OSError) as e:
                    print(e.errno)
                    print("error creating directory: ", outputDir)
                    sys.exit(2)

def main(argv):
    inputDir = ''
    outputDir = ''
    try:
        opts, _ = getopt.getopt(argv,"hi:o:",["idir=","odir="])
    except getopt.GetoptError:
        print('main.py -i <inputDir> -o <outputDir>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <inputDir> -o <outputDir>')
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputDir = arg
        elif opt in ("-o", "--odir"):
            outputDir = arg

    if not os.path.isdir(inputDir):
        print('the input dir is incorrect')
        sys.exit(2)

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
    else:
        shutil.rmtree(outputDir)
        os.mkdir(outputDir)

    readAndCopyFilesToDestination(inputDir, outputDir)
    

if __name__ == "__main__":
    main(sys.argv[1:])
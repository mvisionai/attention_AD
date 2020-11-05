#!/usr/bin/python

import sys, getopt
import os
import shutil
import re


def readAndCopyFilesToDestination(inputDir, outputDir):

    # creating destinations folders
    fnames = []    
    for root,_,f_names in os.walk(inputDir):        
        for f in f_names:
            if f.lower().find(".nii") > -1:
                sourcefile = os.path.join(root, f)
                match = re.search('(\d*)_S_(\d*)', sourcefile)
                forlderSubjectName = match.group(0)
                destinationdir = os.path.join(outputDir, forlderSubjectName)

                try: 
                    if not os.path.exists(destinationdir):
                        os.mkdir(destinationdir)
                        fnames.append(sourcefile)
                    else:
                        #shutil.rmtree(destinationdir)
                        os.mkdir(destinationdir)
                        fnames.append(sourcefile)
                except (IOError, OSError) as e:
                    print(e.errno)
                    print("error creating directory: ", destinationdir)
                    sys.exit(2)

    # copying files
    filesnames = []    
    for f_name in fnames:
        sourcefile = f_name
        f = os.path.basename(sourcefile)
        match = re.search('(\d*)_S_(\d*)', f_name)
        forlderSubjectName = match.group(0)
        destinationdir = os.path.join(outputDir, forlderSubjectName)
        filesCounter = len([d for d in os.listdir(destinationdir) if os.path.isfile(os.path.join(destinationdir, d))])
        
        if filesCounter == 0:
            destinationfile = os.path.join(destinationdir, f)

            try:
                if not os.path.exists(destinationdir):
                    os.mkdir(destinationdir)

                shutil.copyfile(sourcefile, destinationfile)
                filesnames.append(destinationfile)
            except (IOError, OSError) as e:
                print(e.errno)
                print("error copying FROM: ", sourcefile, " TO: ", destinationfile)
                sys.exit(2)
                
    return filesnames

def saveListOfFiles(fnames, outputDir):
    f= open(os.path.join(outputDir, "files-list.txt"), "w+")
    for i in fnames:
        f.write(i + "\r\n")
    f.close()


def main(argv):
    inputDir = ''
    outputDir = ''
    try:
        opts, _ = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
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

    if os.path.isdir(inputDir) == False:
        print('the input dir is incorrect')
        sys.exit(2)

    if os.path.isdir(outputDir) == False:
        print('the output dir is incorrect')
        sys.exit(2)

    fnames = readAndCopyFilesToDestination(inputDir, outputDir)
    
    if len(fnames) > 0:
        saveListOfFiles(fnames, outputDir)


if __name__ == "__main__":
    main(sys.argv[1:])

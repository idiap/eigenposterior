# This is the top-level Makefile for this package

SUBDIRS = pcabin

all : subdirs
.PHONY : all 

subdirs: 
	for dir in $(SUBDIRS); do echo $$dir; $(MAKE) -C $$dir; done

clean:
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done


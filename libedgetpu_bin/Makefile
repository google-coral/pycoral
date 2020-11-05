SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: all \
        deb \
        help

all: deb

deb:
	dpkg-buildpackage -rfakeroot -us -uc -tc -b
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a armhf -d
	dpkg-buildpackage -rfakeroot -us -uc -tc -b -a arm64 -d
	mkdir -p $(MAKEFILE_DIR)/../dist
	mv $(MAKEFILE_DIR)/../*.{deb,changes,buildinfo} \
	   $(MAKEFILE_DIR)/../dist

help:
	@echo "make all       - Build Debian packages for all platforms"
	@echo "make help      - Print help message"


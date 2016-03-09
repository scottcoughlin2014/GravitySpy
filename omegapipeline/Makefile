MATLAB_ROOT := /ldcg/matlab_r2015b
MATLAB_MCR := /ldcg/matlab_r2015b
MATLAB_ARCH := glnxa64
MCCFLAGS := -m -v -R -nodisplay -R -singleCompThread -N -p signal -p stats
MCC := $(MATLAB_ROOT)/bin/mcc $(MCCFLAGS)
INSTDIR = /home/scoughlin/opt/omega/newomega/
WSCAN.BIN = build/wscan_bin
WSCAN.SH = build/wscan
INSTDIR_BIN = $(INSTDIR)/bin/
INSTDIR_LIB = $(INSTDIR)/lib/wscan
FOO = $PATH

all:
	install -d $(INSTDIR_LIB) $(INSTDIR_BIN)
	mkdir -p build
	$(MCC)  wscan.m -a ./share -o wscan_bin -d build
	./gen-wrapper-script.sh $(MATLAB_MCR) $(MATLAB_ARCH) $(INSTDIR_LIB)/wscan_bin $(WSCAN.SH)
	cp build/wscan_bin $(INSTDIR_LIB)
	cp build/wscan $(INSTDIR_BIN)
	echo "export PATH=$(INSTDIR_BIN):$(value FOO)" >> $(INSTDIR)/envsetup.sh
clean:
	rm -rf build

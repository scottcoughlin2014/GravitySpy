all:
	/ldcg/matlab_r2015b/bin/mcc -m -v -R -nodisplay -nocache -R -singleCompThread -N -p signal -p stats wscan.m -a ./share
	chmod 775 wscan
	cp wscan /home/scoughlin/opt/omega/newomega/
clean:
	rm -rf run_wscan.sh wscan *.txt *.log 

#!/bin/bash

# wrapper script generating script for X-Pipeline standalone executables

# (c) 2007 Jameson Rollins <jrollins@phys.columbia.edu>
# Licensed under GPL v3 or later

usage() {
    echo "Usage: $(basename $0) MATLAB_ROOT MATLAB_ARCH TARGET_BIN OUTFILE"
}

MATLAB_ROOT="$1"
MATLAB_ARCH="$2"
BIN="$3"
OUTFILE="$4"

cat <<EOF > "$OUTFILE"
#!/bin/bash
# 
# This is a shell wrapper script for the following
# matlab-compiled standalone binary:
# $BIN

ulimit -c 0

if [[ \`hostname -f\` =~ atlas ]];
then
export MCR_CACHE_ROOT=/local/user/\${USER}/
fi
if [[ \`hostname -f\` =~ ldas ]];
then
export MCR_CACHE_ROOT=/usr1/\${USER}/
fi

ROOT=${MATLAB_ROOT}
ARCH=${MATLAB_ARCH}

LD_LIBRARY_PATH=\${ROOT}/runtime/\${ARCH}:\${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=\${ROOT}/bin/\${ARCH}:\${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=\${ROOT}/sys/os/\${ARCH}:\${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=\${ROOT}/sys/java/jre/\${ARCH}/jre/lib/amd64/native_threads:\${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=\${ROOT}/sys/java/jre/\${ARCH}/jre/lib/amd64/server:\${LD_LIBRARY_PATH}
LD_LIBRARY_PATH=\${ROOT}/sys/java/jre/\${ARCH}/jre/lib/amd64:\${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH

XAPPLRESDIR=\${ROOT}/X11/app-defaults
export XAPPLRESDIR

set -o pipefail
exec ${BIN} "\$@" | gawk 'BEGIN{IGNORECASE=1}/Fatal error/{print ; 1/0}{print}'
EOF
chmod 0755 "$OUTFILE"

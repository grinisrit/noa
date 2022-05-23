#!/bin/env sh

cd "src/TNL"

for F in $(find); do
	if [ -d "$F" ]; then
		echo "$F is a directory"
	else
		echo "Stylizing $F..."
		sed -i 's/#include <TNL/\#include \<noa\/3rdparty\/tnl-noa\/src\/TNL/g' "$F"
		sed -i 's/#include <mpark\/variant.*/\#include \<variant\>/g' "$F"
		sed -i 's/mpark::/std\:\:/g' "$F"
		sed -i 's/ ::TNL/ \:\:noa\:\:TNL/g' "$F"
		sed -i 's/namespace TNL/namespace noa\:\:TNL/g' "$F"
		sed -i 's/<async\//\<noa\/3rdparty\/async\//g' "$F"
	fi
done

#!/bin/env sh

cd "src/TNL"

for F in $(find); do
	if [ -d "$F" ]; then
		echo "$F is a directory"
	else
		echo "Stylizing $F..."
		# Replace include paths for TNL headers
		sed -i 's/#include <TNL/\#include \<noa\/3rdparty\/tnl-noa\/src\/TNL/g' "$F"
		# Replace mpark::variant with std::variant
		sed -i 's/#include <mpark\/variant.*/\#include \<variant\>/g' "$F"
		sed -i 's/mpark::/std::/g' "$F"
		# Put TNL namespace inside noa namespace
		sed -i 's/ ::TNL/ ::noa::TNL/g' "$F"
		sed -i 's/namespace TNL/namespace noa::TNL/g' "$F"
		# Change experimental/filesystem to filesystem
		sed -i 's/experimental\/filesystem/filesystem/g' "$F"
		sed -i 's/experimental::filesystem/filesystem/g' "$F"
		# Change 3rdparty library paths
		# async/* -> noa/3rdparty/async/*
		sed -i 's/<async\//\<noa\/3rdparty\/async\//g' "$F"
		# tinyxml.h -> noa/3rdparty/tinyxml2.hh (removing surrounding ifdef/endif)
		sed -i '1h;1!H;${g;s/#ifdef HAVE_TINYXML2\n\s*#include <tinyxml2.h>\n#endif/#include <noa\/3rdparty\/tinyxml2.hh>/;p};d' "$F"
	fi
done

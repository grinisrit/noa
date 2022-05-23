#! /usr/bin/env python3

import argparse
import datetime
import fnmatch
import os
import re
import sys
from string import Template

default_template = """\
// Copyright (c) 2004-${current_year} Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT
"""

# for each processing type, the detailed settings of how to process files of that type
TYPE_SETTINGS = {
    # All the languages with C style comments:
    "cpp": {
        "extensions": [".h", ".hpp", ".hxx", ".cpp", ".cu", ".cc", ".cxx"],
        "keepFirst": None,
        "blockCommentStartPattern": re.compile(r'^\s*/\*'),
        "blockCommentEndPattern": re.compile(r'\*/\s*$'),
        "lineCommentStartPattern": re.compile(r'^\s*//'),
        "lineCommentEndPattern": None,
    },
    "python": {
        "extensions": [".py"],
        "keepFirst": re.compile(r'^#!|^# +pylint|^# +-\*-|^# +coding|^# +encoding|^# +type|^# +flake8'),
        "blockCommentStartPattern": None,
        "blockCommentEndPattern": None,
        "lineCommentStartPattern": re.compile(r'^\s*#'),
        "lineCommentEndPattern": None,
    },
}

# maps each extension to its processing type. Filled from tpeSettings during initialization
ext2type = {}
name2type = {}
patterns = []


def parse_command_line(argv):
    """
    Parse command line argument. See -h option.
    :param argv: the actual program arguments
    :return: parsed arguments
    """
    import textwrap

    known_extensions = [ftype+":"+",".join(conf["extensions"]) for ftype, conf in TYPE_SETTINGS.items() if "extensions" in conf]
    # known_extensions = [ext for ftype in typeSettings.values() for ext in ftype["extensions"]]

    example = textwrap.dedent("""
      Known extensions: {0}

      If -t/--tmpl is specified, that header is added to (or existing header replaced for) all source files of known type
      If -t/--tmpl is not specified byt -y/--years is specified, all years in existing header files
        are replaced with the years specified

      Examples:
        {1} -t lgpl-v3 -y 2012-2014 -o ThisNiceCompany -n ProjectName -u http://the.projectname.com
        {1} -y 2012-2015
        {1} -y 2012-2015 -d /dir/where/to/start/
        {1} -t .copyright.tmpl -cy
        {1} -t .copyright.tmpl -cy -f some_file.cpp
    """).format(known_extensions, os.path.basename(argv[0]))
    parser = argparse.ArgumentParser(description="copyright header updater",
                                     epilog=example,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dir", dest="dir", default=None,
                        help="Directory to recursively process.")
    parser.add_argument("-f", "--files", dest="files", nargs='*', type=str,
                        help="The list of files to process. If not empty - will disable '--dir' option")
    parser.add_argument("--remove-header", action="store_true",
                        help="Remove the header from the files.")
    parser.add_argument("-t", "--tmpl", dest="tmpl", default=None,
                        help="Template name or file to use.")
    parser.add_argument("-y", "--years", dest="years", default=None,
                        help="Year or year range to use.")
    parser.add_argument("-cy", "--current-year", dest="current_year", action="store_true",
                        help="Use today's year.")
    parser.add_argument("-o", "--owner", dest="owner", default=None,
                        help="Name of copyright owner to use.")
    parser.add_argument("-n", "--projname", dest="projectname", default=None,
                        help="Name of project to use.")
    parser.add_argument("-u", "--projurl", dest="projecturl", default=None,
                        help="Url of project to use.")
    parser.add_argument("--dry", action="store_true", help="Only show what would get done, do not change any files")
    parser.add_argument("--safesubst", action="store_true",
                        help="Do not raise error if template variables cannot be substituted.")
    parser.add_argument("-E","--ext", type=str, nargs="*",
                        help="If specified, restrict processing to the specified extension(s) only")
    parser.add_argument("-x", "--exclude", type=str, nargs="*",
                        help="File path patterns to exclude")
    arguments = parser.parse_args(argv[1:])

    return arguments

def get_paths(fnpatterns, start_dir):
    """
    Retrieve files that match any of the glob patterns from the start_dir and below.
    :param fnpatterns: the file name patterns
    :param start_dir: directory where to start searching
    :return: generator that returns one path after the other
    """
    seen = set()
    for root, dirs, files in os.walk(start_dir):
        names = []
        for pattern in fnpatterns:
            names += fnmatch.filter(files, pattern)
        for name in names:
            path = os.path.join(root, name)
            if path in seen:
                continue
            seen.add(path)
            yield path

def get_files(fnpatterns, files):
    seen = set()
    names = []
    for f in files:
        file_name = os.path.basename(f)
        for pattern in fnpatterns:
            if fnmatch.filter([file_name], pattern):
                names += [f]

    for path in names:
        if path in seen:
            continue
        seen.add(path)
        yield path

def read_file(file, args, type_settings):
    """
    Read a file and return a dictionary with the following elements:
    :param file: the file to read
    :param args: the options specified by the user
    :return: a dictionary with the following entries or None if the file is not supported:
      - skip: number of lines at the beginning to skip (always keep them when replacing or adding something)
       can also be seen as the index of the first line not to skip
      - headStart: index of first line of detected header, or None if non header detected
      - headEnd: index of last line of detected header, or None
      - settings: the type settings
    """
    skip = 0
    head_start = None
    head_end = None
    filename, extension = os.path.splitext(file)
    # if we have no entry in the mapping from extensions to processing type, return None
    ftype = ext2type.get(extension)
    if not ftype:
        ftype = name2type.get(os.path.basename(file))
        if not ftype:
            return None
    settings = type_settings.get(ftype)
    if not os.access(file, os.R_OK):
        print(f"File {file} is not readable.", file=sys.stderr)
    with open(file, "r") as f:
        lines = f.readlines()
    # now iterate throw the lines and try to determine the various indies
    # first try to find the start of the header: skip over shebang or empty lines
    keep_first = settings.get("keepFirst")
    isBlockHeader = False
    block_comment_start_pattern = settings.get("blockCommentStartPattern")
    block_comment_end_pattern = settings.get("blockCommentEndPattern")
    line_comment_start_pattern = settings.get("lineCommentStartPattern")
    i = 0
    print(f"Processing file {file} as {ftype}")
    for line in lines:
        if (i == 0 or i == skip) and keep_first and keep_first.findall(line):
            skip = i + 1
        elif line.strip() == "":
            pass
        elif block_comment_start_pattern and block_comment_start_pattern.findall(line):
            head_start = i
            isBlockHeader = True
            break
        elif line_comment_start_pattern and line_comment_start_pattern.findall(line):
            head_start = i
            break
        elif not block_comment_start_pattern and \
                line_comment_start_pattern and \
                line_comment_start_pattern.findall(line):
            head_start = i
            break
        else:
            # we have reached something else, so no header in this file
            return {"type": ftype,
                    "lines": lines,
                    "skip": skip,
                    "headStart": None,
                    "headEnd": None,
                    "settings": settings,
                    }
        i = i + 1
    # now we have either reached the end, or we are at a line where a block start or line comment occurred
    # if we have reached the end, return default dictionary without info
    if i == len(lines):
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": head_start,
                "headEnd": head_end,
                "settings": settings,
                }
    # otherwise process the comment block until it ends
    if isBlockHeader:
        for j in range(i, len(lines)):
            if block_comment_end_pattern.findall(lines[j]):
                return {"type": ftype,
                        "lines": lines,
                        "skip": skip,
                        "headStart": head_start,
                        "headEnd": j,
                        "settings": settings,
                        }
        # if we went through all the lines without finding an end, maybe we have some syntax error or some other
        # unusual situation, so lets return no header
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": None,
                "headEnd": None,
                "settings": settings,
                }
    else:
        for j in range(i, len(lines)):
            if not line_comment_start_pattern.findall(lines[j]):
                return {"type": ftype,
                        "lines": lines,
                        "skip": skip,
                        "headStart": i,
                        "headEnd": j - 1,
                        "settings": settings,
                        }
        # if we went through all the lines without finding the end of the block, it could be that the whole
        # file only consisted of the header, so lets return the last line index
        return {"type": ftype,
                "lines": lines,
                "skip": skip,
                "headStart": i,
                "headEnd": len(lines) - 1,
                "settings": settings,
                }

def main():
    arguments = parse_command_line(sys.argv)

    type_settings = TYPE_SETTINGS

    for t in type_settings:
        settings = type_settings[t]
        exts = settings["extensions"]
        if "filenames" in settings:
            names = settings['filenames']
        else:
            names = []

        for ext in exts:
            ext2type[ext] = t
            patterns.append("*" + ext)

        for name in names:
            name2type[name] = t
            patterns.append(name)

    limit2exts = None
    if arguments.ext is not None and len(arguments.ext) > 0:
        limit2exts = arguments.ext

    error = False
    template_lines = None
    if arguments.dir is not None and arguments.files:
        print("Cannot use both '--dir' and '--files' options.", file=sys.stderr)
        error = True

    if arguments.years and arguments.current_year:
        print("Cannot use both '--years' and '--currentyear' options.", file=sys.stderr)
        error = True

    current_year = str(datetime.datetime.now().year)
    years = arguments.years
    if arguments.current_year:
        years = current_year

    remove_header = arguments.remove_header
    if remove_header is False:
        settings = {}
        settings["current_year"] = current_year
        if years:
            settings["years"] = years
        if arguments.owner:
            settings["owner"] = arguments.owner
        if arguments.projectname:
            settings["projectname"] = arguments.projectname
        if arguments.projecturl:
            settings["projecturl"] = arguments.projecturl
        # if we have a template name specified, try to get or load the template
        if arguments.tmpl:
            # check if we can interpret the option as file
            if os.path.isfile(arguments.tmpl):
                template_file = os.path.abspath(arguments.tmpl)
                print(f"Using file {template_file}")
                with open(template_file, "r") as f:
                    template_lines = f.readlines()
                    if args.safesubst:
                        template_lines = [Template(line).safe_substitute(settings) for line in template_lines]
                    else:
                        template_lines = [Template(line).substitute(settings) for line in template_lines]
            else:
                print(f"Template file does not exist, cannot proceed: {arguments.tmpl}", file=sys.stderr)
                error = True
        else:
            template_lines = [line + "\n" for line in default_template.splitlines()]
            if arguments.safesubst:
                template_lines = [Template(line).safe_substitute(settings) for line in template_lines]
            else:
                template_lines = [Template(line).substitute(settings) for line in template_lines]

    if error:
        return 1

    # now do the actual processing: if we did not get some error, we have a template loaded or
    # no template at all
    # if we have no template, then we will have the years.
    # now process all the files and either replace the years or replace/add the header
    paths = []
    if arguments.files:
        paths += get_files(patterns, arguments.files)
    if arguments.dir:
        paths += get_paths(patterns, arguments.dir)
    if not paths:
        print("No paths were given.")
        return 1

    for file in paths:
        file = os.path.normpath(file)
        # skip symbolic links (they may lead to a different source tree)
        if os.path.islink(file):
            continue
        if limit2exts is not None and not any([file.endswith(ext) for ext in limit2exts]):
            print("Skipping file with non-matching extension: {}".format(file))
            continue
        if arguments.exclude and any([fnmatch.fnmatch(file, pat) for pat in arguments.exclude]):
            print("Ignoring file {}".format(file))
            continue
        finfo = read_file(file, arguments, type_settings)
        if not finfo:
            print(f"File not supported {file}")
            continue
        lines = finfo["lines"]
#        LOGGER.debug(
#            "Info for the file: headStart=%s, headEnd=%s, skip=%s, len=%s,
#            finfo["headStart"], finfo["headEnd"], finfo["skip"], len(lines))
        # replace or add
        if not arguments.dry:
            with open(file, "w") as fw:
                if (fw is not None):
                    # if we found a header, replace it
                    # otherwise, add it after the lines to skip
                    head_start = finfo["headStart"]
                    head_end = finfo["headEnd"]
                    ftype = finfo["type"]
                    skip = finfo["skip"]
                    if head_start is not None and head_end is not None:
                        # first write the lines before the header
                        fw.writelines(lines[0:head_start])
                        if remove_header is False:
                            #  now write the new header from the template lines
                            fw.writelines(template_lines)
                        else:
                            # also remove blank lines after the (removed) header
                            while lines[head_end + 1].strip() == "":
                                head_end += 1
                        #  now write the rest of the lines
                        fw.writelines(lines[head_end + 1:])
                    else:
                        fw.writelines(lines[0:skip])
                        if remove_header is False:
                            fw.writelines(template_lines)
                            # ensure there is a blank line after the header
                            if lines[skip].strip():
                                fw.write("\n")
                        else:
                            # also remove blank lines after the (removed) header
                            while lines[skip].strip() == "":
                                skip += 1
                        fw.writelines(lines[skip:])

    return 0

if __name__ == "__main__":
    sys.exit(main())

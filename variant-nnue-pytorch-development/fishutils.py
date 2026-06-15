#!/usr/bin/env python3
#
# fishutils is a script for semi-automatic code changes in Stockfish.
# Copyright (C) 2017-2019 Fabian Fichter
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import re
import argparse
import logging
import traceback
import parser
from math import floor, ceil
from difflib import unified_diff

MODES = ("spsa", "function", "align")
ROUNDING = {'round': round, 'ceil': ceil, 'floor': floor}
LOGGING = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR}


def parse_args():
    arg_parser = argparse.ArgumentParser(description="Auto-apply parameter tweaks to source code.")
    arg_parser.add_argument("-s", "--src-dir", help="path to directory of Stockfish source code", type=str,
                            required=True)
    arg_parser.add_argument("-i", "--input-file", help="path to input file (e.g., containing tuning results)", type=str)
    arg_parser.add_argument("-m", "--mode", help="mode. Default: spsa", choices=MODES, default="spsa")
    mutex_group = arg_parser.add_mutually_exclusive_group()
    mutex_group.add_argument("-d", "--dry-run", help="show changes, do not write to files", action="store_true")
    mutex_group.add_argument("--low-memory", help="do not keep all files in memory", action="store_true")
    arg_parser.add_argument("-r", "--rounding", help="rounding function. Default: round", choices=ROUNDING.keys(),
                            default="round")
    arg_parser.add_argument("-l", "--log-level", help="logging level. Default: WARNING", choices=LOGGING.keys(),
                            default="WARNING")
    return arg_parser.parse_args()


def get_index(enum, entry):
    """Calculates enum value by counting commata."""
    m = re.search(entry + r"\s*=\s*(?P<v>(-)?\d+)", enum)
    if m is not None:
        return int(m.group("v"))
    else:
        return enum[:enum.find(entry)].count(",")


def is_source(f):
    return os.path.isfile(f) and f.endswith((".h", ".cpp"))


def find_index(s, l):
    """
    Finds position where a specified array entry is defined.

    Gets a code snippet containing the array definition
    and a list of indices as an input.

    Returns a position in the input string.
    """
    pos = s.find("=")
    if l:
        pos = s.find("{", pos) + 1
    for d in range(len(l)):
        for i in range(l[d]):
            pos_new = s.find(",", pos + 1)
            while s.count("{", 0, pos_new + 1) - s.count("}", 0, pos_new + 1) > d + 1:
                pos_new = s.find("}", pos_new + 1)
            while s.count("(", pos, pos_new + 1) - s.count(")", pos, pos_new + 1) > 0:
                pos_new = s.find(")", pos_new + 1)
            pos = s.find(",", pos_new)
    return pos


class SourceFile(object):
    def __init__(self, path, in_memory):
        self.path = path
        self.in_memory = in_memory
        self.orig = None
        self.curr = None
        if in_memory:
            with open(self.path) as f:
                self.orig = f.read()
                self.curr = self.orig

    def read_orig(self):
        assert self.in_memory, "File has to be in memory to read original content."
        return self.orig

    def read(self):
        if self.in_memory:
            return self.curr
        else:
            with open(self.path, "r") as f:
                return f.read()

    def write(self, s):
        if self.in_memory:
            self.curr = s
        else:
            with open(self.path, "w") as f:
                f.write(s)

    def diff(self):
        assert self.in_memory, "File has to be in memory to show diff."
        s = str()
        for line in unified_diff(self.orig.splitlines(True), self.curr.splitlines(True), fromfile=self.path,
                                 tofile=self.path):
            s += line
        return s

    def save(self):
        if self.in_memory:
            with open(self.path, "w") as f:
                f.write(self.curr)


class Repository(object):
    """
    Represents a Stockfish source directory.

    The path to the repository is a required argument.
    """

    def __init__(self, path, in_memory=True):
        self.src_path = os.path.abspath(path)
        self.in_memory = in_memory
        self.file_paths = [os.path.join(self.src_path, f) for f in os.listdir(self.src_path) if
                           is_source(os.path.join(self.src_path, f))]
        self.files = list()
        for f_path in self.file_paths:
            self.files.append(SourceFile(f_path, self.in_memory))

    def search_def(self, var_name, enum=False):
        """
        Searches for definition of variable or enum value.

        Returns a regex match object and a file handle.
        """

        for file in self.files:
            s = file.read()
            if enum:
                pattern = r"^\s*enum\s*" + r"[\w\s:]*{([^{}]*?(\s|,))?" + var_name + r"(\s|,|}).*?;"
            else:
                pattern = (r"^([\w \t]*\s+|)(?P<type>\w+)\s+"
                           + var_name
                           + r"((\[|\s)?(\[[^=;\\\{\}\[\]]*\])*[ \t]*)?=(?P<def>(?!=)[^;]*?);")
            m = re.search(pattern, s, re.DOTALL | re.MULTILINE)
            if m is not None:
                return m, file

        return None, None

    def parse_indices(self, s):
        """Returns an array of integers from string of index accesses ('[1][2][3]' -> [1, 2, 3])."""
        indices = []
        while s != "":
            p1 = s.find('[')
            p2 = s.find(']')
            if p1 == -1 or p2 == -1 or p1 + 1 >= p2:
                break
            i = s[p1 + 1:p2]
            try:
                i = int(i)
            except ValueError:
                m, _ = self.search_def(i, enum=True)
                if m is not None:
                    i = get_index(m.group(), i)
                else:
                    raise Exception("Index %s is neither int nor enum." % i)
            indices.append(i)
            s = s[p2 + 1:]
        return indices

    def process_spsa_match(self, match_input, f_round):
        """
        Processes a regex match object of a parsed input line.

        Replaces a hard-coded value in the code
        according to the content of the match object.
        """
        if match_input.group("name").startswith(("m", "e")):
            name = match_input.group("name")[1:]
            prefix = match_input.group("name")[0]
        else:
            name = match_input.group("name")
            prefix = "v"
        fullname = match_input.group("name") + match_input.group("indices")
        indices = self.parse_indices(match_input.group("indices"))
        match_definition, file = self.search_def(name)
        if match_definition is None:
            logging.error("%s: Definition not found." % fullname)
            return
        s2 = match_definition.group()
        i = find_index(s2, indices)
        new_val = str(int(f_round(float(match_input.group("value_new")))))

        assert (match_definition.group("type") == "Score") == bool(prefix in ("m", "e"))

        if prefix in ("m", "e"):
            pattern = r"^(?:(?!#|//).)*?(S|make_score)\((?P<m>\s*[-\d]+)\s*,(?P<e>\s*[-\d]+)\s*\)"
        elif match_definition.group("type") == "Value":
            pattern = r"^(?:(?!#|//).)*?(((V|Value)\((?P<v>\s*[-\d]+)\s*\))|(?P<enum>\w+))"
        elif match_definition.group("type") == "int":
            pattern = r"^(?:(?!#|//).)*?((?P<v>[ \t]*[-\d]+)|(?P<enum>\w+))"
        else:
            logging.error("%s: Unknown type %s." % (fullname, match_definition.group("type")))
            return

        match_value = re.search(pattern, s2[i:], re.MULTILINE)
        if not match_value:
            logging.error("%s: Not found." % fullname)
            return

        if match_value.group(prefix) is None:
            # Try to replace defining enum
            enum_name = match_value.group('enum')
            prefix = 'v'
            i = 0
            logging.debug("%s: Replacing defining enum '%s' instead of array entry itself." % (fullname, enum_name))
            match_definition, file = self.search_def(enum_name, enum=True)
            if match_definition is not None:
                logging.debug("Enum definition found in file %s: %s" % (file.path, match_value.group(0)))
                match_value = re.search(enum_name + r"\s*=\s*(?P<v>(-)?\d+)", match_definition.group(0),
                                        re.DOTALL | re.MULTILINE)
                assert match_value is not None
            else:
                logging.error("Replacing %s failed. Definition '%s' can not be handled." % (fullname, enum_name))
                return

        if match_input.group("value_old") is not None and int(match_value.group(prefix)) != int(
                float(match_input.group("value_old"))):
            logging.warning("%s: Value in file differs from start value: %s !=  %s" %
                            (fullname, int(match_value.group(prefix)), int(float(match_input.group("value_old")))))

        if len(match_value.group(prefix)) < len(new_val):
            logging.info("%s: Code lines might be not aligned." % fullname)

        file_content = (match_definition.string[:match_definition.start() + i + match_value.start(prefix)]
                        + new_val.rjust(len(match_value.group(prefix)))
                        + match_definition.string[match_definition.start() + i + match_value.end(prefix):])

        file.write(file_content)

    def process_function_match(self, match_input, f_round):
        varname, f_string = match_input.group('name'), match_input.group('func')
        match_definition, file = self.search_def(varname)
        if match_definition is None:
            logging.error("%s: Definition not found." % varname)
            return

        try:
            func = eval(parser.expr('lambda x: ' + f_string).compile())
        except Exception:
            logging.error("%s: Function '%s' could not be parsed." % (varname, f_string))
            raise

        def_string = match_definition.group('def')
        pattern = re.compile(r'\d+')
        try:
            new_def = pattern.sub(lambda x: str(int(f_round(func(float(x.group()))))), def_string)
        except Exception:
            logging.error("%s: Failed to apply function '%s'." % (varname, f_string))
            raise

        file_content = (match_definition.string[:match_definition.start('def')]
                        + new_def
                        + match_definition.string[match_definition.end('def'):])

        file.write(file_content)

    def align_array(self, match_input, _):
        var_name = match_input.group('name')
        match_definition, file = self.search_def(var_name)
        if match_definition is None:
            logging.error("%s: Definition not found." % var_name)
            return
        current = match_definition.group(0)
        previous = ""
        while current != previous:
            previous = current
            current = self.align(current)
        file_content = (match_definition.string[:match_definition.start()]
                        + current
                        + match_definition.string[match_definition.end():])

        file.write(file_content)

    @staticmethod
    def align(s):
        lines = s.splitlines()
        format_patterns = []
        for line in lines:
            format_patterns.append(re.sub(r'[^{}(),]*', '', line))

        for i in range(max(len(lines) - 1, 0)):
            if (len(format_patterns[i]) > 0 and format_patterns[i].rstrip(',') == format_patterns[i + 1].rstrip(',')
                    and (format_patterns[i][0] != '{' or len(os.path.commonprefix([lines[i], lines[i + 1]])) >= lines[
                        i].find(format_patterns[i][0]))):
                # Align adjacent lines
                pos_old = -1
                for j in format_patterns[i][:len(os.path.commonprefix([format_patterns[i], format_patterns[i + 1]]))]:
                    pos0 = lines[i].find(j, pos_old + 1)
                    pos1 = lines[i + 1].find(j, pos_old + 1)
                    if pos0 < pos1:
                        lines[i] = lines[i][:pos_old + 1] + (pos1 - pos0) * ' ' + lines[i][pos_old + 1:]
                    elif pos0 > pos1:
                        lines[i + 1] = lines[i + 1][:pos_old + 1] + (pos0 - pos1) * ' ' + lines[i + 1][pos_old + 1:]
                    pos_old = max(pos0, pos1)

        return "\n".join(lines)

    def diff(self):
        assert self.in_memory
        s = ""
        for f in self.files:
            s += f.diff()

        return s

    def save(self):
        for f in self.files:
            f.save()


class InputParser(object):
    def __init__(self, input_strings=None, repo=None, src_dir=None):
        self.process_method = None
        self.regex_pattern = None
        self.help_text = None
        self.input_strings = input_strings if input_strings else []
        assert repo is not None or src_dir is not None
        self.repo = repo if repo is not None else Repository(src_dir)

    def user_input(self):
        assert self.help_text is not None
        print(self.help_text)
        while True:
            try:
                s = input()
            except EOFError:
                sys.exit(1)
            if not s:
                break
            self.input_strings.append(s)

    def process(self, dry_run, f_round):
        for line in self.input_strings:
            line = line.rstrip()
            if not line:
                # skip empty lines
                continue
            match_input = self.read_line(line)
            if match_input:
                try:
                    self.process_match(match_input, f_round)
                except Exception:
                    logging.exception("Failed to process input line '%s'" % line)
            else:
                logging.warning("Skipping invalid input line: '%s'." % line)

        if dry_run:
            print(self.repo.diff())
        else:
            self.repo.save()

    def read_line(self, s):
        assert self.regex_pattern is not None
        return re.match(self.regex_pattern, s)

    def process_match(self, match_input, f_round):
        assert self.process_method is not None
        self.process_method(match_input, f_round)


class ResultParser(InputParser):
    """Parser object for processing result strings of SPSA tuning sessions."""

    def __init__(self, *args, **kwargs):
        super(ResultParser, self).__init__(*args, **kwargs)
        self.process_method = self.repo.process_spsa_match
        self.regex_pattern = r"^\s*(param|Parameter): (?P<name>\w*)(?P<indices>[[\]\w\s]*), (best|theta):" \
                             r" (?P<value_new>[\w.-]*)" \
                             r"(, start: (?P<value_old>[\w.-]*))?"
        self.help_text = "Give SPSA tuning results as reported by fishtest.\nQuit input by empty line.\nInput:"


class FunctionParser(InputParser):
    def __init__(self, *args, **kwargs):
        super(FunctionParser, self).__init__(*args, **kwargs)
        self.process_method = self.repo.process_function_match
        self.regex_pattern = r"^\s*(?P<name>\S+)\s*;\s*(?P<func>.+?)\s*$"
        self.help_text = "Give array and a formula f(x) separated by a semicolon.\n" \
                         "Example: 'razor_margin;2*x' doubles the razoring margins.\n" \
                         "Quit input by empty line.\nInput:"


class AlignParser(InputParser):
    def __init__(self, *args, **kwargs):
        super(AlignParser, self).__init__(*args, **kwargs)
        self.process_method = self.repo.align_array
        self.regex_pattern = r"^\s*(?P<name>\S+)\s*$"
        self.help_text = "Give the names of the arrays that are to be aligned, with one name per line.\n" \
                         "Quit input by empty line.\nInput:"


def main(args):
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=LOGGING[args.log_level])

    repo = Repository(args.src_dir, not args.low_memory)
    if len(repo.files) == 0:
        sys.exit("Error: Path %s does not contain any source files." % args.src_dir)

    input_strings = list()
    if args.input_file is not None:
        with open(args.input_file, "r") as f:
            input_strings = f.readlines()

    if args.mode == "spsa":
        parser_class = ResultParser
    elif args.mode == "function":
        parser_class = FunctionParser
    elif args.mode == "align":
        parser_class = AlignParser
    else:
        assert False, "This should never be reached. Check definition of modes."

    input_parser = parser_class(input_strings=input_strings, repo=repo)
    if not input_strings:
        input_parser.user_input()
    input_parser.process(dry_run=args.dry_run, f_round=ROUNDING[args.rounding])


if __name__ == "__main__":
    main(parse_args())

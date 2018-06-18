#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey);
#                 Arnab Ghoshal, Karel Vesely

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This is basically a file from Kaldi speech recognition toolkit (https://github.com/kaldi-asr/kaldi)
# See, for instance, this for more details: https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh

# Parse command-line options.
# To be sourced by another script (as in ". parse_options.sh").
# Option format is: --option-name arg
# and shell variable "option_name" gets set to value "arg."
# The exception is --help, which takes no arguments, but prints the
# $help_message variable (if defined).

###
### No we process the command line options
###

# DEPRECATED, TO BE REMOVED
ignore_unknown_params=${ignore_unknown_params:-false}
# 'die' - default behaviour of original script, see comments below
# 'ignore' - ignore (skip) this parameter
# 'set' - create new variable and set its value
unknown_params_action=${unknown_params_action:-'ignore'}
# Optionally, print found command line arguments
__print_parsed_args=${__print_parsed_args:-false}

while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    # If the enclosing script is called with --help option, print the help
    # message and exit.  Scripts should put help messages in $help_message
  --help|-h) if [ -z "$help_message" ]; then echo "No help found." 1>&2;
      else printf "$help_message\n" 1>&2 ; fi;
      exit 0 ;;
  --*=*) echo "$0: options to scripts must be of the form --name value, got '$1'"
       exit 1 ;;
    # If the first command-line argument begins with "--" (e.g. --foo-bar),
    # then work out the variable name as $name, which will equal "foo_bar".
  --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`;
    # Next we test whether the variable in question is undefned-- if so it's
    # an invalid option and we die.  Note: $0 evaluates to the name of the
    # enclosing script.
    # The test [ -z ${foo_bar+xxx} ] will return true if the variable foo_bar
    # is undefined.  We then have to wrap this test inside "eval" because
    # foo_bar is itself inside a variable ($name).
      #echo "found option with name \"$name\" and value \"$2\""
      eval '[ -z "${'$name'+xxx}" ]' &&  \
      [ "$ignore_unknown_params" = "true" ] && {
        shift 2;
        continue;
      }
      eval '[ -z "${'$name'+xxx}" ]' &&  \
      [ "$unknown_params_action" = "ignore" ] && {
        shift 2;
        continue;
      }
      eval '[ -z "${'$name'+xxx}" ]' &&  \
      [ "$unknown_params_action" = "die" ] && {
        echo "$0: invalid option $1";
        exit 1;
      }

      #eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;

      oldval="`eval echo \\$$name`";
    # Work out whether we seem to be expecting a Boolean argument.
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then
    was_bool=true;
      else
    was_bool=false;
      fi

    # Set the variable to the right value-- the escaped quotes make it work if
    # the option had spaces, like --cmd "queue.pl -sync y"
    # Why is dollar escaped? Read this - https://stackoverflow.com/a/9715377/575749.
    # Another option is to use declare $name=$2
      eval $name=\"\$2\";
      [ "$__print_parsed_args" = "true" ] && {
        echo "Found new command line argument: $name = ${!name}"
      }

    # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;
  *) break;
  esac
done


# Check for an empty argument to the --cmd option, which can easily occur as a
# result of scripting errors.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" 1>&2 && exit 1;


true; # so this script returns exit code 0.

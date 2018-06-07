#!/bin/bash

echo "Hello from shell process that's supposed to do something useful."
echo "Note how dummy.launcher variabled has been expanded."
echo "Also note that we replaced all '.' with '_' in variables names."

echo "$0 $*"                                        # For debugging purposes.
unknown_params_action='set'                         # For all parameters that are not set, set.
__print_parsed_args='true'
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options

FROM nvcr.io/nvidia/tensorflow:18.07-py3

# DLBS hash tag
ARG DLBS_HASHTAG='7c5ca5a6dfa4e2f7b8b4d81c60bd8be343dabd30'

# Config files need to reference these variables if they want to be compatible with the MLBox
ENV CONFIG_ROOT='/workspace/config'
ENV LOGS_ROOT='/workspace/logs'
ENV DATA_ROOT='/workspace/data'

# Clone DLBS
RUN git clone https://github.com/HewlettPackard/dlcookbook-dlbs.git /dlbs &&
    cd /dlbs && git reset --hard ${DLBS_HASHTAG}

# Initialize environment and run packaged configuration
ENTRYPOINT /bin/bash -c "source /dlbs/scripts/environment.sh && \
                         python /dlbs/python/dlbs/experimenter.py run --config /${CONFIG_ROOT}/config.json"

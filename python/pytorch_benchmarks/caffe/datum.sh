#!/bin/bash

protoc -I=. --python_out=. ./datum.proto
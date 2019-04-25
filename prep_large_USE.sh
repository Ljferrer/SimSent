#!/usr/bin/env bash
# VersionArg: model version number (highest number will be deployed)
VersionArg="${1:-001}"

# Make Dockerized Model
python SimSent/vectorizer/make_service_model.py -l -v "$VersionArg"

printf "\nReady to deploy large Universal Sentence Encoder for similarity server \n\n"

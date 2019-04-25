#!/usr/bin/env bash
printf "Dockerizing small Universal Sentence Encoder for query service\n"

# Update Docker TF Serving
docker pull tensorflow/serving
printf "\n"

# Construct Docker Run Instructions
PORT=8501

VECTORIZER_NAME="USE-lite-v2"

SHELL_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
MODEL_DIR="$SHELL_DIR/SimSent/vectorizer/service_models/$VECTORIZER_NAME/"

MOUNT_INSTRUCTIONS="type=bind,source=$MODEL_DIR,target=/models/$VECTORIZER_NAME"

# Run It
docker run -d -p "$PORT:$PORT" --mount "$MOUNT_INSTRUCTIONS" \
-e "MODEL_NAME=$VECTORIZER_NAME" -t tensorflow/serving

# Report Status
printf "\nUsing port $PORT for requests to $VECTORIZER_NAME\n\n"
docker ps
printf "\n"



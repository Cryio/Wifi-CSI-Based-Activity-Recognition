#!/bin/bash

# Check if both duration and iterations are provided as arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <duration_in_seconds> <number_of_iterations>"
    exit 1
fi

DURATION=$1
ITERATIONS=$2

# Validate that duration is a positive integer
if ! [[ "$DURATION" =~ ^[0-9]+$ ]] || [ "$DURATION" -le 0 ]; then
    echo "Error: Duration must be a positive integer."
    exit 1
fi

# Validate that iterations is a positive integer
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -le 0 ]; then
    echo "Error: Number of iterations must be a positive integer."
    exit 1
fi

echo "Running record_both.py for $ITERATIONS iterations, each for $DURATION seconds..."

# Loop for the specified number of iterations
for ((i=1; i<=ITERATIONS; i++)); do
    echo "Starting iteration $i of $ITERATIONS..."

    # Run the Python script in the background and capture its PID
    python3.11 record_both.py &
    PYTHON_PID=$!

    # Wait for the specified duration
    sleep $DURATION

    # Terminate the Python process gracefully
    echo "Stopping iteration $i..."
    kill -SIGTERM $PYTHON_PID

    # Wait for the process to terminate
    wait $PYTHON_PID 2>/dev/null

    # Optional: Add a short pause between iterations to avoid overlap
    sleep 1
done

echo "All $ITERATIONS iterations completed."

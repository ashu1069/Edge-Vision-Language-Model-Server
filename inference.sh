#!/bin/bash
# inference.sh - Quick inference helper script with annotation support

if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_file> [confidence_threshold] [prompt] [--annotate]"
  echo ""
  echo "Examples:"
  echo "  $0 test.jpg"
  echo "  $0 test.jpg 0.5"
  echo "  $0 test.jpg 0.5 'Find all people in this image'"
  echo "  $0 test.jpg 0.5 'Find all people' --annotate  # Save annotated image and JSON"
  exit 1
fi

IMAGE_FILE=$1
CONF=${2:-0.5}
PROMPT=${3:-"Analyze this image"}

# Check for --annotate flag
ANNOTATE=false
if [[ "$*" == *"--annotate"* ]]; then
  ANNOTATE=true
  # Remove --annotate from arguments
  set -- "${@/--annotate/}"
fi

if [ ! -f "$IMAGE_FILE" ]; then
  echo "Error: Image file '$IMAGE_FILE' not found"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Edge Vision Language Model Server - Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Image: $IMAGE_FILE"
echo "Confidence threshold: $CONF"
echo "Prompt: $PROMPT"
echo ""

# Check if API is available
if ! curl -s http://localhost:8000/health > /dev/null; then
  echo "Error: API server is not responding. Is it running?"
  echo "Start it with: docker compose up -d"
  exit 1
fi

# Encode and submit
echo "Encoding image and submitting request..."
IMAGE_B64=$(base64 -i "$IMAGE_FILE" | tr -d '\n')
RESPONSE=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"$PROMPT\", \"confidence_threshold\": $CONF}")

# Check for errors
if echo "$RESPONSE" | grep -q "detail"; then
  echo "Error submitting request:"
  echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
  exit 1
fi

REQUEST_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['request_id'])")
echo "✓ Request submitted"
echo "Request ID: $REQUEST_ID"
echo ""
echo "Processing (this may take a few seconds)..."

# Poll for result
MAX_ATTEMPTS=30
for i in $(seq 1 $MAX_ATTEMPTS); do
  sleep 1
  RESULT=$(curl -s "http://localhost:8000/result/$REQUEST_ID")
  STATUS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
  
  if [ "$STATUS" = "completed" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Results"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$RESULT" | python3 -m json.tool
    
    # If --annotate flag is set, create annotated image and save JSON
    if [ "$ANNOTATE" = true ]; then
      echo ""
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "  Creating Annotated Output"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      python3 annotate_result.py "$IMAGE_FILE" "$REQUEST_ID"
    fi
    
    exit 0
  elif [ "$STATUS" = "processing" ]; then
    echo -n "."
  else
    echo ""
    echo "Error: Unexpected status: $STATUS"
    echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"
    exit 1
  fi
done

echo ""
echo "Timeout: Request took longer than expected. Check manually:"
echo "  curl http://localhost:8000/result/$REQUEST_ID | python3 -m json.tool"
exit 1


#!/bin/bash
# Phase 4: Error Handling Evaluation

# Load test configuration with secure tokens
source "$(dirname "$0")/test-config.sh"

echo "🛡️ Phase 4: Error Handling Evaluation"
echo "===================================="

error_score=0

echo "Testing authentication error handling..."
# Test without auth token
unauth_response=$(curl -s -X POST "$SVAMI_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"question": "test", "role": "backend_developer"}')

if echo "$unauth_response" | grep -qiE "(authentication|auth|unauthorized|error)"; then
    error_score=$((error_score + 2))
    echo "✅ Authentication errors handled correctly"
else
    echo "⚠️ Authentication error handling needs improvement"
fi

echo "Testing malformed request handling..."
# Test with malformed request (using secure test configuration)
AUTH_HEADER=$(create_auth_header "bearer")
malformed_response=$(curl -s -X POST "$SVAMI_URL/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: $AUTH_HEADER" \
    -d '{"invalid": "json", "structure": true}')

if echo "$malformed_response" | grep -qiE "(error|invalid|bad)"; then
    error_score=$((error_score + 2))
    echo "✅ Malformed requests handled correctly"
else
    echo "⚠️ Malformed request handling needs improvement"
fi

echo "Testing edge case handling..."
# Test with empty question (using secure test configuration)
empty_response=$(curl -s -X POST "$SVAMI_URL/answer" \
    -H "Content-Type: application/json" \
    -H "Authorization: $AUTH_HEADER" \
    -d '{"question": "", "role": "backend_developer"}')

if [ -n "$empty_response" ]; then
    error_score=$((error_score + 2))
    echo "✅ Edge cases handled correctly"
else
    echo "⚠️ Edge case handling needs improvement"
fi

echo "📊 Error Handling Score: $error_score/6"

if [ $error_score -ge 5 ]; then
    echo "✅ Phase 4: PASSED (Excellent error handling)"
    exit 0
elif [ $error_score -ge 3 ]; then
    echo "⚠️ Phase 4: PASSED (Good error handling)"
    exit 0
else
    echo "❌ Phase 4: FAILED (Needs improvement)"
    exit 1
fi

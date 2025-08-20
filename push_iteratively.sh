DISTORTIONS=("fast" "reversed" "narrowband" "tone_vocoded" "noise_vocoded" "sinewave" "glimpsed" "sculpted")
for DIST in "${DISTORTIONS[@]}"; do
    echo "pushing distortion: $DIST"
    python push_to_hub.py $DIST
done
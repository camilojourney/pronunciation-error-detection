#!/bin/bash
# Quick workflow using uv package manager

set -e  # Exit on error

echo "=========================================="
echo "PRONUNCIATION ERROR CLASSIFICATION"
echo "Using uv package manager"
echo "=========================================="

# Step 1: Install dependencies
echo -e "\n[1/5] Installing dependencies with uv..."
uv sync

# Step 2: Download NLTK data
echo -e "\n[2/5] Downloading NLTK data..."
uv run python -c "import nltk; nltk.download('punkt')"

# Step 3: Quick test
echo -e "\n[3/5] Running quick test..."
uv run python test_setup.py

if [ $? -eq 0 ]; then
    echo -e "\n✅ Quick test passed!"

    # Step 4: Train classifier (optional - commented out by default)
    # echo -e "\n[4/5] Training classifier on full dataset..."
    # uv run python train_classifier.py

    # Step 5: Evaluate model (optional - commented out by default)
    # echo -e "\n[5/5] Evaluating model..."
    # uv run python evaluate_model.py

    echo -e "\n=========================================="
    echo "✅ SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Download L2-ARCTIC dataset (see SETUP_GUIDE.md)"
    echo "  2. Run: uv run python train_classifier.py"
    echo "  3. Run: uv run python evaluate_model.py"
    echo "  4. Render presentation: quarto render nlp_presentation_final.qmd"
    echo ""
else
    echo -e "\n❌ Quick test failed!"
    echo "Check SETUP_GUIDE.md for troubleshooting"
    exit 1
fi

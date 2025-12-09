#!/usr/bin/env bash
# Script to add activation alias to .bashrc
# Run with: ./activate.sh

ALIAS_LINE="alias activate='source $(pwd)/venv/bin/activate'"

if grep -q "alias activate=" ~/.bashrc; then
    echo "Alias 'activate' already exists in ~/.bashrc"
else
    echo "$ALIAS_LINE" >> ~/.bashrc
    echo "✓ Added alias to ~/.bashrc"
    echo "Run 'source ~/.bashrc' or restart your terminal to use it"
fi

echo ""
echo "You can now type 'activate' from anywhere to activate this venv"
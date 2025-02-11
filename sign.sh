#!/bin/bash
set -e

# Configuration and path setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_PATH="dist/mac-computer-use.app"
IDENTITY=""
NOTARIZATION_PROFILE="maccomputeruse"
DMG_NAME="mac-computer-use-0.0.0.dmg"  # Using hyphens instead of spaces
DMG_PATH="${SCRIPT_DIR}/releases/${DMG_NAME}"
ENTITLEMENTS_PATH="entitlements.plist"

# Check prerequisites
if [ ! -d "$APP_PATH" ]; then
    echo "Error: $APP_PATH not found. Please run build.sh first."
    exit 1
fi

# If DMG exists in root directory, move it to releases
if [ -f "${SCRIPT_DIR}/${DMG_NAME}" ]; then
    echo "Moving DMG from root directory to releases..."
    mkdir -p "${SCRIPT_DIR}/releases"
    mv "${SCRIPT_DIR}/${DMG_NAME}" "${DMG_PATH}"
fi

# Check for DMG
if [ ! -f "${DMG_PATH}" ]; then
    echo "Error: DMG not found. Please run build.sh first."
    echo "Expected path: ${DMG_PATH}"
    exit 1
fi

# First, remove any existing signatures
echo "Removing existing signatures..."
codesign --remove-signature "$APP_PATH" || true

# Sign all binaries within the app bundle first
echo "Signing internal binaries..."
find "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources" -type f \( \
    -name "*.so" -o \
    -name "*.dylib" -o \
    -path "*/Contents/MacOS/*" -o \
    -name "Python" -o \
    -name "python*" \) | while read -r file; do
    echo "Signing $file"
    codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS_PATH" \
        --sign "$IDENTITY" \
        --timestamp "$file"
done

# Sign the frameworks
echo "Signing frameworks..."
find "$APP_PATH" -type d -path "*/Contents/Frameworks/*.framework" | while read -r framework; do
    echo "Signing $framework"
    codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS_PATH" \
        --sign "$IDENTITY" \
        --timestamp "$framework"
done

# Sign the main app bundle
echo "Signing main application..."
codesign --force --options runtime \
    --entitlements "$ENTITLEMENTS_PATH" \
    --deep \
    --sign "$IDENTITY" \
    --timestamp "$APP_PATH"

# Verify the signature
echo "Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

# Sign the DMG
echo "Signing DMG..."
codesign --force --sign "$IDENTITY" --timestamp "$DMG_PATH"

# Notarize the DMG
echo "Notarizing DMG..."
xcrun notarytool submit --wait --keychain-profile "$NOTARIZATION_PROFILE" "$DMG_PATH"

# Staple the notarization
echo "Stapling notarization..."
xcrun stapler staple "$DMG_PATH"

echo "Done! Signed and notarized DMG is at $DMG_PATH" 
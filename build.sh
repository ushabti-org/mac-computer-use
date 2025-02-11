#!/bin/bash
set -e  # Exit on error

echo "=== Starting build process ==="
echo "Current directory: $(pwd)"

echo "=== Cleaning previous builds ==="
rm -rf build dist releases
echo "Cleaned build, dist, and releases directories"

echo "=== Activating virtual environment ==="
source venv/bin/activate
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Ensure entitlements.plist exists
echo "=== Checking entitlements.plist ==="
if [ ! -f "entitlements.plist" ]; then
    echo "Creating entitlements.plist..."
    cat > entitlements.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
EOF
    echo "Created entitlements.plist"
else
    echo "entitlements.plist already exists"
fi

echo "=== Installing PyInstaller ==="
pip install pyinstaller

echo "=== Building app bundle ==="
echo "Using spec file: mac-computer-use.spec"
pyinstaller mac-computer-use.spec

echo "=== Verifying app bundle ==="
if [ ! -d "dist/mac-computer-use.app" ]; then
    echo "Error: App bundle not created successfully"
    echo "Contents of dist directory:"
    ls -la dist/
    exit 1
else
    echo "App bundle created successfully"
fi

echo "=== Creating DMG ==="
# Create DMG directory and ensure absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DMG_NAME="mac-computer-use-0.0.0.dmg"
DMG_PATH="${SCRIPT_DIR}/releases/${DMG_NAME}"
APP_PATH="${SCRIPT_DIR}/dist/mac-computer-use.app"

echo "Script directory: ${SCRIPT_DIR}"
echo "DMG name: ${DMG_NAME}"
echo "DMG path: ${DMG_PATH}"
echo "App path: ${APP_PATH}"

echo "=== Cleaning up existing DMG files ==="
rm -f "${SCRIPT_DIR}/mac-computer-use-0.0.0.dmg"
rm -f "${SCRIPT_DIR}/mac-computer-use 0.0.0.dmg"
rm -f "${SCRIPT_DIR}/releases/mac-computer-use-0.0.0.dmg"
rm -f "${SCRIPT_DIR}/releases/mac-computer-use 0.0.0.dmg"
rm -f "${DMG_PATH}"
echo "Cleaned up existing DMG files"

echo "=== Creating releases directory ==="
mkdir -p "${SCRIPT_DIR}/releases"
echo "Created releases directory"

echo "=== Running create-dmg ==="
create-dmg \
    --volname "Mac Computer Use" \
    --sandbox-safe \
    --no-internet-enable \
    --verbose \
    --debug \
    "${DMG_PATH}" \
    "${APP_PATH}" || {
        echo "Error: create-dmg failed with status $?"
        echo "Full command output above"
        echo "Contents of current directory:"
        ls -la
        echo "Contents of releases directory:"
        ls -la releases/
        exit 1
    }

echo "=== Checking for DMG in root directory ==="
if [ -f "${SCRIPT_DIR}/mac-computer-use 0.0.0.dmg" ]; then
    echo "Found DMG in root directory, moving to releases..."
    mv "${SCRIPT_DIR}/mac-computer-use 0.0.0.dmg" "${DMG_PATH}"
fi

echo "=== Verifying final DMG location ==="
if [ ! -f "${DMG_PATH}" ]; then
    echo "Error: DMG was not created successfully"
    echo "Checking contents of releases directory:"
    ls -la releases/
    echo "Checking root directory:"
    ls -la "${SCRIPT_DIR}"
    exit 1
else
    echo "DMG created successfully at: ${DMG_PATH}"
    echo "DMG size: $(ls -lh "${DMG_PATH}" | awk '{print $5}')"
fi

echo "=== Build completed successfully! ==="
echo "App bundle: dist/mac-computer-use.app"
echo "DMG file: ${DMG_PATH}"
echo "You can now run ./sign.sh to sign and notarize the DMG"

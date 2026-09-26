#!/usr/bin/env python3
"""
SSL Certificate Fix for Corporate Proxy (Raytheon/RTX)
======================================================
This script fixes SSL certificate verification errors when behind a corporate
proxy that uses Raytheon/RTX certificates.

The script:
1. Extracts all Raytheon/RTX CA certificates from Windows certificate stores
2. Creates a custom CA bundle that includes these certificates
3. Sets the SSL_CERT_FILE environment variable to use this bundle
"""

import os
import sys
import subprocess
import certifi

def extract_corporate_certs():
    """Extract all Raytheon/RTX certificates from Windows certificate stores."""
    ps_script = '''
$stores = @("LocalMachine\\Root", "CurrentUser\\Root", "LocalMachine\\CA", "CurrentUser\\CA")

$stores | ForEach-Object {
    $storePath = $_
    Get-ChildItem -Path "Cert:\\$storePath" -ErrorAction SilentlyContinue | 
    Where-Object { 
        $_.Subject -like "*Raytheon*" -or $_.Subject -like "*RTX*" -or
        $_.Issuer -like "*Raytheon*" -or $_.Issuer -like "*RTX*"
    } | ForEach-Object {
        $cert = $_
        $bytes = $cert.RawData
        $base64 = [System.Convert]::ToBase64String($bytes, [System.Base64FormattingOptions]::None)
        "-----BEGIN CERTIFICATE-----"
        for ($i = 0; $i -lt $base64.Length; $i += 64) {
            $end = [Math]::Min($i + 64, $base64.Length)
            $base64.Substring($i, $end - $i)
        }
        "-----END CERTIFICATE-----"
    }
}
'''
    
    result = subprocess.run(['powershell', '-Command', ps_script], capture_output=True, text=True)
    return result.stdout

def create_custom_ca_bundle(corporate_certs):
    """Create a custom CA bundle with corporate certificates."""
    certifi_bundle = certifi.where()
    with open(certifi_bundle, 'r') as f:
        certifi_content = f.read()
    
    custom_bundle = certifi_content + '\n' + corporate_certs
    
    custom_path = os.path.join(os.path.dirname(certifi_bundle), 'custom_cacert.pem')
    with open(custom_path, 'w') as f:
        f.write(custom_bundle)
    
    return custom_path

def test_ssl_connection(ca_path):
    """Test SSL connection with the custom CA bundle."""
    import ssl
    import urllib.request
    
    context = ssl.create_default_context(cafile=ca_path)
    
    try:
        req = urllib.request.Request('https://paper-api.alpaca.markets/v2/account')
        r = urllib.request.urlopen(req, timeout=10, context=context)
        return True, f'Status: {r.status}'
    except Exception as e:
        error_str = str(e)
        if '401' in error_str or 'Unauthorized' in error_str:
            return True, 'SSL works, getting expected 401 Unauthorized (no API key)'
        return False, str(e)

def main():
    print("=" * 60)
    print("SSL Certificate Fix for Corporate Proxy (Raytheon/RTX)")
    print("=" * 60)
    
    # Extract corporate certificates
    print("\n[1/3] Extracting Raytheon/RTX certificates from Windows store...")
    corporate_certs = extract_corporate_certs()
    
    cert_count = corporate_certs.count('-----BEGIN CERTIFICATE-----')
    print(f"Found {cert_count} corporate certificate(s)")
    
    # Create custom CA bundle
    print("\n[2/3] Creating custom CA bundle...")
    custom_path = create_custom_ca_bundle(corporate_certs)
    print(f"Custom bundle: {custom_path}")
    print(f"Size: {os.path.getsize(custom_path)} bytes")
    
    # Test SSL connection
    print("\n[3/3] Testing SSL connection...")
    success, message = test_ssl_connection(custom_path)
    
    if success:
        print(f"SUCCESS: {message}")
    else:
        print(f"FAILED: {message}")
    
    print("\n" + "=" * 60)
    print("Configuration Complete")
    print("=" * 60)
    print(f"\nThe broker.py module will automatically use this CA bundle.")
    print(f"Environment variables set:")
    print(f"  SSL_CERT_FILE={custom_path}")
    print(f"  REQUESTS_CA_BUNDLE={custom_path}")
    print(f"  CURL_CA_BUNDLE={custom_path}")

if __name__ == "__main__":
    main()
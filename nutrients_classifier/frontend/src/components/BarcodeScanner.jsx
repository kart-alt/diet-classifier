import React, { useEffect, useRef } from 'react';
import { Html5QrcodeScanner } from 'html5-qrcode';

const BarcodeScanner = ({ onResult, onClose }) => {
  const scannerRef = useRef(null);

  useEffect(() => {
    const scanner = new Html5QrcodeScanner(
      "reader",
      { fps: 10, qrbox: { width: 250, height: 150 } },
      /* verbose= */ false
    );

    scanner.render((decodedText) => {
      scanner.clear();
      onResult(decodedText);
    }, (error) => {
      // ignore scanning errors
    });

    return () => {
      scanner.clear().catch(error => {
        console.error("Failed to clear html5QrcodeScanner. ", error);
      });
    };
  }, [onResult]);

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(0,0,0,0.8)', zIndex: 9999, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
      <div style={{ background: 'var(--surface)', padding: '20px', borderRadius: '20px', width: '100%', maxWidth: '500px' }}>
        <h3 className="syne-font" style={{ fontSize: '1.5rem', marginBottom: '1rem', textAlign: 'center' }}>Scan Barcode</h3>
        <div id="reader" style={{ width: '100%', border: 'none' }}></div>
        <button className="btn secondary" style={{ width: '100%', marginTop: '20px', justifyContent: 'center' }} onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
};

export default BarcodeScanner;

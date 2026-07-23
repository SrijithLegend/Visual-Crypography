🔐 AI-Driven Steganographic Visual Cryptography for Secure Transactions
An end-to-end cryptographic and generative AI system designed to secure financial transactions. This system combines Elliptic Curve Digital Signatures (ECDSA), 2-of-2 Visual Cryptography (VSS), and Conditional Generative Adversarial Networks (cGANs) to encode, split, and hide signed QR code payloads inside realistic synthetic images.

📌 Overview
Traditional transaction security relies heavily on digital-only verification. This project introduces a hybrid physical-digital security pipeline:

Digital Signature & QR Encoding: A transaction payload is signed using ECDSA (P-256) and encoded into a QR code.

Visual Cryptography (2-of-2 VSS): The QR matrix is decomposed into two noise-like transparency shares (shareA and shareB). Neither share reveals information individually, but physically or digitally stacking them reveals the original QR code via bitwise pixel expansion.

Generative Steganography (cGAN): A Conditional GAN conditioned on both shares simultaneously (128x128x2) embeds the visual cryptography data into a synthetic 3-channel RGB image.

🏗 System Architecture
[ Transaction Data ] 
         │
         ▼
  [ ECDSA Sign ] ──► [ SHA-256 Digest ]
         │
         ▼
  [ QR Code Generator ]
         │
         ▼
  [ 2-of-2 Visual Cryptography Split ]
      ┌──┴──────────┐
      ▼             ▼
  [ Share A ]   [ Share B ]
      └──┬──────────┘
         ▼ (Stacked 2-Channel Input)
  [ cGAN Generator (U-Net) ]
         │
         ▼
  [ Synthetic Stego Image ]
🛠 Features
Authenticity & Non-Repudiation: Powered by PyCryptodome using the P-256 ECC curve and FIPS-186-3 DSS.

Information-Theoretic Security: The 2-of-2 visual secret sharing scheme ensures individual shares contain zero mutual information regarding the underlying QR payload.

Deep Learning Integration: Built with TensorFlow/Keras, featuring a U-Net Generator with skip connections and a PatchGAN-style Discriminator.

📦 Prerequisites & Installation
Ensure you have Python 3.9+ installed. Install the required dependencies:

Bash
pip install tensorflow numpy pillow qrcode pycryptodome matplotlib
💻 Usage
Run the main pipeline script to generate signatures, construct shares, train the cGAN, and render the final synthetic output:

Bash
python main.py
Script Execution Flow
Key Generation: Automatically creates or loads signing_key.pem.

Transaction Creation: Signs demo payload {"sender": "Alice", "receiver": "Bob", "amount": 10000.0}.

QR Generation & Splitting: Generates transaction_qr.png, splits it into shareA.png and shareB.png, and saves an overlay test reconstructed_qr.png.

Model Training & Generation: Trains the cGAN on the stacked dual-channel share input and displays the synthesized stego image alongside the input shares.

📑 Technical References
Visual Cryptography: Naor, M., & Shamir, A. (1994). Visual Cryptography. Advances in Cryptology – EUROCRYPT '94.

Conditional GANs: Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix).

📜 License
Distributed under the MIT License. See LICENSE for more information.